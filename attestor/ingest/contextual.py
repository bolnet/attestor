# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Contextual embedding — Anthropic-style chunk-context prefix at ingest.

Anthropic's Sept 2024 *Contextual Retrieval* research
(https://www.anthropic.com/news/contextual-retrieval) showed that
prefixing each chunk with 1-2 sentences of LLM-generated document-level
context BEFORE embedding cut retrieval failures by ~49%. This module
implements that pre-pass for Attestor's write path.

Where it fires
--------------

``AgentMemory.add()`` runs the orchestration object below before the
vector store add. When ``stack.ingest.contextual_embedding.enabled`` is
``False`` (the shipping default), this module is bypassed entirely and
the codepath is byte-identical to ``main`` — preserving backward
compatibility.

What it does
------------

1. Pull the last ``session_window`` memories from the same session/
   namespace as document context.
2. Call a small LLM to write a 1-2 sentence prefix situating the new
   content in that document.
3. Cap the prefix at ``max_prefix_chars`` (anti-runaway).
4. Embed the prefixed payload ``[CTX] <prefix>\\n\\n<content>`` instead
   of the raw content.
5. Store the prefix on the memory's ``metadata['_context_prefix']`` so
   it survives the document store round-trip and is auditable.

Failure isolation
-----------------

The LLM call is the only new external dependency. Any exception from
it (network down, API rate-limit, malformed response) is caught and
logged at WARNING; the embedder degrades to the raw payload and ingest
proceeds. The doc store is the source of truth and never sees the
contextual layer.

Caching
-------

The prefix is purely a function of (content, session_history). The
process-local cache keys by content_hash so re-ingesting the same
content (in a different namespace, or after a restart of a long-lived
service) reuses the prior prefix without paying for another LLM call.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("attestor.ingest.contextual")


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextPrefix:
    """Result of one LLM contextualization call.

    ``prefix`` is empty string on degraded paths (LLM failure / disabled).
    ``degraded`` carries the explicit signal so callers can branch
    without checking string emptiness.
    """

    prefix: str
    degraded: bool = False


# ─── Prompt template ────────────────────────────────────────────────────


_CONTEXTUAL_PROMPT = """You are tagging a memory snippet with the \
short context that would help a search engine find it.

The user is having an ongoing conversation. Several earlier turns are \
shown below as DOCUMENT CONTEXT. A new snippet (NEW MEMORY) has just \
been written. Write 1-2 sentences situating the NEW MEMORY in the \
DOCUMENT CONTEXT — what topic / thread / event is it part of?

Rules:
- 1-2 sentences only.
- Reference the topic / event / thread, NOT the snippet's literal text.
- Keep it under {max_chars} characters.
- Plain prose. No bullets, no labels, no quotation marks.
- If the document context is empty, briefly state what topic the \
snippet itself is about.

DOCUMENT CONTEXT (most recent first):
{document_context}

NEW MEMORY:
{content}

Context (1-2 sentences, ≤{max_chars} chars):"""


_LABEL_PREFIXES = (
    "context:",
    "context (1-2 sentences):",
    "context (1-2 sentences,",
    "summary:",
)


def _format_history(session_history: list[str], window: int) -> str:
    """Truncate to the last ``window`` items and render newest-first."""
    if not session_history:
        return "(no prior turns in this session)"
    last = session_history[-window:]
    # Newest-first so the LLM sees the freshest signal at the top of
    # the prompt window — matters when many turns push the oldest out
    # of the model's local-attention sweet spot.
    return "\n".join(f"- {turn}" for turn in reversed(last))


def _strip_label_echo(text: str) -> str:
    """Some models echo the 'Context:' label — strip whatever the prompt
    framed itself with."""
    lower = text.lower()
    for label in _LABEL_PREFIXES:
        if lower.startswith(label):
            return text[len(label):].strip()
    return text


def build_context_prefix(
    *,
    content: str,
    session_history: list[str],
    client: Any,
    model: str,
    max_prefix_chars: int = 200,
    session_window: int = 5,
) -> ContextPrefix:
    """One LLM call returning a 1-2 sentence prefix for ``content``.

    Pure function over an injected ``client`` (any object exposing
    ``chat.completions.create(**kwargs)`` — the OpenAI SDK shape, which
    is what the rest of the Attestor codebase already speaks). The
    caller owns client construction so this module stays free of YAML
    / env reads.

    Returns a :class:`ContextPrefix` with ``prefix=""`` and
    ``degraded=True`` when the LLM call raises. Never escapes an
    exception — the goal is failure isolation: a flaky LLM must never
    block ingest.
    """
    if not content.strip():
        return ContextPrefix(prefix="")

    document_context = _format_history(session_history, session_window)
    prompt = _CONTEXTUAL_PROMPT.format(
        document_context=document_context,
        content=content,
        max_chars=max_prefix_chars,
    )

    try:
        # max_tokens generous (300) — the model still emits ~50-80 tokens
        # for a 1-2 sentence prefix. We hard-cap the *characters* on the
        # output, not the tokens, so the cap is consistent across
        # tokenizers.
        response = client.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "contextual embedding LLM call failed (%s: %s); "
            "falling back to raw content for this write",
            type(e).__name__, e,
        )
        return ContextPrefix(prefix="", degraded=True)

    text = _strip_label_echo(text)
    if len(text) > max_prefix_chars:
        text = text[:max_prefix_chars].rstrip()

    return ContextPrefix(prefix=text)


# ─── Orchestration ──────────────────────────────────────────────────────


@dataclass
class ContextualEmbedder:
    """Applies the contextual prefix at ingest time.

    Mutable on purpose: holds a process-local ``_cache`` so re-ingesting
    the same content_hash within a process reuses the prefix. The cfg
    + client are otherwise immutable references.

    Public API: ``prepare(content, session_history) -> (payload, prefix)``.
    """

    cfg: Any  # ContextualEmbeddingCfg — duck-typed to avoid import cycle
    client: Any | None = None
    _cache: dict[str, str] = field(default_factory=dict)

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.strip().encode()).hexdigest()

    def prepare(
        self,
        content: str,
        session_history: list[str],
    ) -> tuple[str, str | None]:
        """Build the embedding payload + (optionally) the prefix to stash
        on Memory.metadata.

        Returns ``(payload, prefix_or_None)``:
          - When the embedder is disabled OR the LLM call degrades,
            ``payload == content`` and the second element is ``None``.
          - When successful, ``payload == "[CTX] <prefix>\\n\\n<content>"``
            and the second element is the raw prefix text for metadata.
        """
        # Fast path: feature disabled.
        if not getattr(self.cfg, "enabled", False):
            return content, None

        if self.client is None:
            logger.debug(
                "contextual embedder enabled but no client wired; "
                "falling back to raw content"
            )
            return content, None

        # Cache: same content_hash → reuse prior prefix.
        cache_on = bool(getattr(self.cfg, "cache", True))
        if cache_on:
            chash = self._content_hash(content)
            cached = self._cache.get(chash)
            if cached is not None:
                return self._format_payload(content, cached), cached

        result = build_context_prefix(
            content=content,
            session_history=session_history,
            client=self.client,
            model=getattr(self.cfg, "model", "anthropic/claude-haiku-4.5"),
            max_prefix_chars=int(getattr(self.cfg, "max_prefix_chars", 200)),
            session_window=int(getattr(self.cfg, "session_window", 5)),
        )

        if result.degraded or not result.prefix:
            return content, None

        if cache_on:
            self._cache[self._content_hash(content)] = result.prefix

        return self._format_payload(content, result.prefix), result.prefix

    @staticmethod
    def _format_payload(content: str, prefix: str) -> str:
        """The wire format for the contextualized embedding."""
        return f"[CTX] {prefix}\n\n{content}"
