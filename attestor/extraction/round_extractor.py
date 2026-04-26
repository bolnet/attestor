"""Round-level fact extraction (Phase 3.3, roadmap §A.2).

Two functions:

  extract_user_facts(turn, recent_context, model, client) -> list[ExtractedFact]
      Speaker-locked to the USER. Returns durable facts the user stated
      (preferences, plans, identity attributes...).

  extract_agent_facts(turn, recent_context, model, client) -> list[ExtractedFact]
      Speaker-locked to the ASSISTANT. Returns recommendations,
      decisions, commitments — anything an auditor must replay.

Both share:
  - JSON-only output, parsed with markdown-fence tolerance
  - Strict per-fact validation (drops malformed facts rather than crash)
  - source_span citations clamped to the turn's content range
  - Confidence clamped to [0.0, 1.0]

LLM client is injectable so tests can use a stub. By default, builds an
OpenRouter-backed OpenAI client from OPENROUTER_API_KEY (or OPENAI_API_KEY).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from attestor.conversation.turns import ConversationTurn
from attestor.extraction.prompts import (
    format_agent_fact_prompt,
    format_user_fact_prompt,
)

logger = logging.getLogger("attestor.extraction.round")

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 2048

USER_FACT_CATEGORIES = {
    "preference", "career", "project", "technical",
    "personal", "location", "relationship", "event", "financial",
}
AGENT_FACT_CATEGORIES = {
    "recommendation", "decision", "commitment",
    "constraint", "calculation", "refusal",
}


@dataclass(frozen=True)
class ExtractedFact:
    """One atomic fact emitted by the extractor.

    Bound to a single source turn; the (start, end) span lets the audit
    pipeline highlight which portion of the turn produced this fact.
    """

    text: str
    category: str
    entity: Optional[str]
    confidence: float
    source_span: List[int]
    speaker: str  # "user" | "assistant" | "<agent_id>"

    @property
    def source_start(self) -> int:
        return self.source_span[0]

    @property
    def source_end(self) -> int:
        return self.source_span[1]


# ──────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────


def default_llm_client():
    """Build an OpenRouter-backed OpenAI client. Raises if no key set.

    Tests inject a stub via the `client` parameter on the extract_*
    functions, so this is only invoked when a real call is needed.
    """
    from openai import OpenAI

    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
        )
    oa_key = os.environ.get("OPENAI_API_KEY")
    if oa_key:
        return OpenAI(api_key=oa_key)
    raise RuntimeError(
        "No LLM API key set. Provide OPENROUTER_API_KEY or OPENAI_API_KEY, "
        "or pass client= explicitly."
    )


def _call_llm(
    prompt: str,
    *,
    client: Any,
    model: str,
    max_tokens: int,
) -> str:
    """Invoke the LLM and return the raw response text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


# ──────────────────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _parse_facts_payload(raw: str) -> List[dict]:
    """Parse the LLM response into a list of fact dicts.

    Tolerates: markdown fences, pure-array shape, the documented
    {"facts": [...]} envelope. Returns [] on any parse failure rather
    than raising — extractor failures must never break ingest.
    """
    text = _strip_markdown_fences(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("extractor JSON parse failed: %s; raw=%r", e, raw[:200])
        return []
    if isinstance(parsed, list):
        return [f for f in parsed if isinstance(f, dict)]
    if isinstance(parsed, dict):
        facts = parsed.get("facts", [])
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict)]
    return []


def _validate_fact(
    raw: dict,
    *,
    speaker: str,
    content_len: int,
    allowed_categories: set,
) -> Optional[ExtractedFact]:
    """Coerce a raw fact dict into an ExtractedFact, or return None.

    Validates:
      - text is non-empty
      - category is in the allowed set (else "general")
      - confidence is a float in [0, 1] (else 0.5)
      - source_span is [start, end] both ints, clamped to [0, content_len]
        and start <= end (else [0, 0])
    """
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    category = raw.get("category")
    if not isinstance(category, str) or category not in allowed_categories:
        category = "general"

    entity_raw = raw.get("entity")
    entity = entity_raw if isinstance(entity_raw, str) and entity_raw else None

    conf_raw = raw.get("confidence", 0.5)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    span_raw = raw.get("source_span", [0, content_len])
    if (
        isinstance(span_raw, list) and len(span_raw) == 2
        and all(isinstance(x, int) for x in span_raw)
    ):
        start = max(0, min(content_len, span_raw[0]))
        end = max(start, min(content_len, span_raw[1]))
    else:
        start, end = 0, content_len

    return ExtractedFact(
        text=text.strip(),
        category=category,
        entity=entity,
        confidence=confidence,
        source_span=[start, end],
        speaker=speaker,
    )


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def extract_user_facts(
    turn: ConversationTurn,
    recent_context: str = "(none)",
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[Any] = None,
) -> List[ExtractedFact]:
    """Extract durable facts from a USER turn.

    Speaker-locked: refuses to extract from anything but the user message
    via the IMPORTANT line in the prompt. Returns [] if the turn is not
    a user turn, the LLM returns empty, or parsing fails — never raises
    on extraction failure.
    """
    if not turn.is_user:
        logger.warning(
            "extract_user_facts called on a non-user turn (role=%r); "
            "returning empty", turn.role,
        )
        return []
    prompt = format_user_fact_prompt(
        ts=turn.ts.isoformat(),
        user_message=turn.content,
        recent_context_summary=recent_context or "(none)",
    )
    cli = client or default_llm_client()
    raw_text = _call_llm(prompt, client=cli, model=model, max_tokens=max_tokens)
    raw_facts = _parse_facts_payload(raw_text)

    facts: List[ExtractedFact] = []
    for rf in raw_facts:
        f = _validate_fact(
            rf, speaker=turn.speaker,
            content_len=len(turn.content),
            allowed_categories=USER_FACT_CATEGORIES,
        )
        if f is not None:
            facts.append(f)
    return facts


def extract_agent_facts(
    turn: ConversationTurn,
    recent_context: str = "(none)",
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[Any] = None,
) -> List[ExtractedFact]:
    """Extract durable statements from an ASSISTANT turn.

    Speaker-locked to the assistant message. Categories are
    recommendation / decision / commitment / constraint / calculation
    / refusal — the audit categories an auditor cares about.
    """
    if not turn.is_assistant:
        logger.warning(
            "extract_agent_facts called on a non-assistant turn (role=%r); "
            "returning empty", turn.role,
        )
        return []
    prompt = format_agent_fact_prompt(
        ts=turn.ts.isoformat(),
        assistant_message=turn.content,
        recent_context_summary=recent_context or "(none)",
    )
    cli = client or default_llm_client()
    raw_text = _call_llm(prompt, client=cli, model=model, max_tokens=max_tokens)
    raw_facts = _parse_facts_payload(raw_text)

    facts: List[ExtractedFact] = []
    for rf in raw_facts:
        f = _validate_fact(
            rf, speaker=turn.speaker,
            content_len=len(turn.content),
            allowed_categories=AGENT_FACT_CATEGORIES,
        )
        if f is not None:
            facts.append(f)
    return facts
