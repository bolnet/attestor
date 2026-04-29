"""Critique-and-revise answerer (Phase 3 PR-E, +3-5% LME-S).

A single greedy decode often produces a confident-but-wrong answer
because the answerer commits before sanity-checking against the
retrieved context. A two-pass cycle — answer, critique, then revise
only when the critic flags an issue — catches a meaningful fraction
of those failures with bounded extra cost (~3x answerer at most;
1x extra in the common case where the critic says ``pass``).

This module is answerer-side only — it does NOT touch the retrieval
pipeline or ``RetrievalCfg``. The orchestrator runs unchanged.

Configuration lives in ``configs/attestor.yaml`` under the top-level
``stack.critique_revise`` block (peer of ``stack.self_consistency``,
not nested inside ``retrieval``):

    stack:
      critique_revise:
        enabled: false
        critic_model: null      # null → models.verifier
        revise_model: null      # null → models.answerer
        max_revisions: 1        # hard cap; loader rejects > 1

Trace events emitted (when ``ATTESTOR_TRACE=1``):

  - ``answer.critique.verdict`` — verdict, reason, revised
  - ``answer.critique.revised`` — initial_length, final_length,
    n_revisions

Cost: 1 answer + 1 critique + at most 1 revise per question. Disabled
by default — gate behind the YAML knob and flip per bench run only.

This PR caps ``max_revisions`` at 1: literature shows diminishing
returns past one revision and the cost compounds. The cap is enforced
in the YAML loader (``_parse_yaml`` raises SystemExit on > 1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("attestor.longmemeval_critique")


# ──────────────────────────────────────────────────────────────────────
# Result shape
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CritiqueResult:
    """Three-step critique-revise output.

    All fields are populated even on degraded paths so downstream
    tracing has full visibility into what happened. Immutable per the
    project coding-style rules.

    Fields:
        initial_answer: Raw output of the first (greedy) answerer call.
        critique:       The critic LLM's self-critique text. Empty
                        string when the critic call failed or no
                        critique was produced.
        final_answer:   The revised answer when the critic said
                        ``revise``; otherwise equal to ``initial_answer``.
        revised:        True iff a revise call fired AND produced a
                        non-empty answer that was actually used.
        n_revisions:    Number of revise calls that succeeded (0 or 1
                        in this PR).
    """

    initial_answer: str
    critique: str
    final_answer: str
    revised: bool
    n_revisions: int


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────


_CRITIQUE_PROMPT = (
    "You just produced this answer to a memory-recall question. Now check "
    "it against the source context for factual accuracy.\n\n"
    "Question: {question}\n\n"
    "Context (retrieved memories):\n{context}\n\n"
    "Your initial answer: {initial_answer}\n\n"
    "Decide:\n"
    "  - If the answer is fully grounded in the context AND addresses the\n"
    "    question accurately, output exactly:\n"
    "      VERDICT: pass\n"
    "      REASON: <one short sentence>\n"
    "  - If the answer is wrong, partial, or unsupported by context, output:\n"
    "      VERDICT: revise\n"
    "      REASON: <what's wrong>\n"
    "      FIX: <what the correct answer should say>\n\n"
    "Output:"
)


_REVISE_PROMPT = (
    "Your initial answer to this question was incorrect or incomplete. "
    "A reviewer flagged the issue. Produce a corrected answer.\n\n"
    "Question: {question}\n\n"
    "Context (retrieved memories):\n{context}\n\n"
    "Initial answer: {initial_answer}\n\n"
    "Reviewer feedback: {critique}\n\n"
    "Corrected answer:"
)


# ──────────────────────────────────────────────────────────────────────
# Critique parsing
# ──────────────────────────────────────────────────────────────────────


# Match `VERDICT: pass` / `verdict = revise` / `Verdict :REVISE` etc.
# Case-insensitive; tolerant of whitespace and punctuation between the
# label and the value.
_VERDICT_RE = re.compile(
    r"verdict\s*[:=]?\s*(pass|revise)\b",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"reason\s*[:=]?\s*(.+?)(?:\n[A-Z]+\s*[:=]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FIX_RE = re.compile(
    r"fix\s*[:=]?\s*(.+?)(?:\n[A-Z]+\s*[:=]|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_verdict(text: str) -> str:
    """Extract the VERDICT value from critique output.

    Returns "pass", "revise", or "pass" as the safe default when the
    output can't be parsed unambiguously. Defaulting to ``pass`` is
    deliberate: a malformed critique should NEVER cause us to discard
    the initial answer in favor of an unconstrained revise — that path
    risks making a worse answer worse. When in doubt, keep the
    original.
    """
    if not text:
        return "pass"
    m = _VERDICT_RE.search(text)
    if not m:
        return "pass"
    return m.group(1).lower()


def _parse_reason(text: str) -> str:
    """Best-effort extraction of the REASON line from critique output."""
    if not text:
        return ""
    m = _REASON_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _parse_fix(text: str) -> str:
    """Best-effort extraction of the FIX line from critique output."""
    if not text:
        return ""
    m = _FIX_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


# ──────────────────────────────────────────────────────────────────────
# Message helpers
# ──────────────────────────────────────────────────────────────────────


def _question_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Best-effort extraction of the user's question from the chat
    messages — used to fill the critique/revise prompt's ``{question}``
    placeholder. If the schema doesn't match, falls back to a generic
    placeholder so the prompt still renders."""
    if not messages:
        return "(no question available)"
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content[:8000]
    return str(messages[-1].get("content", ""))[:8000]


# ──────────────────────────────────────────────────────────────────────
# Per-step LLM calls
# ──────────────────────────────────────────────────────────────────────


def _initial_answer(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
) -> str:
    """Greedy initial answer — same shape as the legacy single-sample
    path. Returns the stripped content text, or ``""`` on any error."""
    from attestor.llm_trace import traced_create

    response = traced_create(
        client,
        role="answerer",
        model=model,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    return text.strip()


def _critique(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
    initial_answer: str,
) -> str:
    """Critic call. Returns the stripped critique text or ``""`` on
    error. Caller is responsible for parsing the verdict."""
    from attestor.llm_trace import traced_create

    prompt = _CRITIQUE_PROMPT.format(
        question=question,
        context=context,
        initial_answer=initial_answer,
    )
    response = traced_create(
        client,
        role="critic",
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    return text.strip()


def _revise(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
    initial_answer: str,
    critique: str,
) -> str:
    """Reviser call. Fires only when the critic returned VERDICT=revise.
    Returns the stripped revised answer or ``""`` on error."""
    from attestor.llm_trace import traced_create

    prompt = _REVISE_PROMPT.format(
        question=question,
        context=context,
        initial_answer=initial_answer,
        critique=critique,
    )
    response = traced_create(
        client,
        role="reviser",
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def answer_with_critique_revise(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    question: Optional[str] = None,
    context: str = "",
    critic_client: Optional[Any] = None,
    critic_model: Optional[str] = None,
    revise_client: Optional[Any] = None,
    revise_model: Optional[str] = None,
    max_revisions: int = 1,
) -> CritiqueResult:
    """Three-step critique-and-revise pipeline.

    Steps:
      1. Initial answer — single greedy decode using ``client``/``model``;
         identical to the legacy single-sample path.
      2. Critique       — feed (question, context, initial_answer) into
         the critic. Output is structured as
         ``VERDICT: pass|revise\\nREASON: ...`` (with optional ``FIX:``).
      3. Revise (cond.) — only when ``VERDICT=revise``: feed the
         critique back into the reviser to produce a corrected answer.

    Args:
        client:        OpenAI-compatible answerer client. Used for
                       step 1 and (when ``revise_client`` is None)
                       step 3.
        model:         Answerer model id for step 1.
        messages:      Chat-completion messages forwarded verbatim to
                       step 1. The user's question is also extracted
                       from this list for the critique/revise prompts
                       when ``question`` is not provided explicitly.
        question:      Optional explicit question text for the critic
                       prompt. Defaults to the last ``user`` message
                       in ``messages``.
        context:       Retrieved-context block to embed in the
                       critique/revise prompts. Empty string is
                       acceptable but degrades the critic's ability
                       to ground-check.
        critic_client: Optional critic LLM client. Defaults to ``client``.
        critic_model:  Required when critique-revise is enabled.
                       Should normally come from
                       ``stack.critique_revise.critic_model`` or fall
                       back to ``stack.models.verifier``.
        revise_client: Optional reviser client. Defaults to ``client``.
        revise_model:  Reviser model id. Defaults to ``model`` when None.
        max_revisions: Hard-capped at 1 in this PR. Values > 1 are
                       silently clamped here; the YAML loader rejects
                       them up front so this is defensive only.

    Returns:
        ``CritiqueResult`` populated with all four fields. ``revised``
        is True iff a revise call actually changed the final answer;
        ``n_revisions`` is 0 or 1.

    Defensive contract: on ANY LLM failure (initial, critic, reviser
    network errors, malformed output, etc.), returns a CritiqueResult
    with ``initial_answer`` as ``final_answer`` and ``revised=False``.
    Never raises and never returns an empty final answer when the
    initial answer succeeded.
    """
    # Defensive cap — YAML loader already rejects > 1, but keep the
    # public surface honest for direct callers.
    if max_revisions > 1:
        max_revisions = 1
    if max_revisions < 0:
        max_revisions = 0

    effective_critic_model = critic_model or model
    effective_revise_model = revise_model or model
    effective_critic_client = critic_client or client
    effective_revise_client = revise_client or client

    # ── Step 1: initial answer ───────────────────────────────────────
    try:
        initial = _initial_answer(client=client, model=model, messages=messages)
    except Exception as exc:  # noqa: BLE001 — never abort the user-facing path
        logger.debug("critique_revise initial answer failed: %s", exc)
        return CritiqueResult(
            initial_answer="",
            critique="",
            final_answer="",
            revised=False,
            n_revisions=0,
        )

    # If the initial answer is empty there's nothing for the critic to
    # check — short-circuit and return early.
    if not initial:
        return CritiqueResult(
            initial_answer="",
            critique="",
            final_answer="",
            revised=False,
            n_revisions=0,
        )

    # If max_revisions is 0 we skip critique entirely (the knob is
    # effectively a no-op; identical to the single-sample path but
    # honors the contract).
    if max_revisions == 0:
        return CritiqueResult(
            initial_answer=initial,
            critique="",
            final_answer=initial,
            revised=False,
            n_revisions=0,
        )

    # ── Step 2: critique ─────────────────────────────────────────────
    effective_question = question or _question_from_messages(messages)
    try:
        critique_text = _critique(
            client=effective_critic_client,
            model=effective_critic_model,
            question=effective_question,
            context=context,
            initial_answer=initial,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("critique_revise critic call failed: %s", exc)
        # Critic failed → keep initial as the final answer.
        return CritiqueResult(
            initial_answer=initial,
            critique="",
            final_answer=initial,
            revised=False,
            n_revisions=0,
        )

    verdict = _parse_verdict(critique_text)
    reason = _parse_reason(critique_text)
    _emit_verdict_event(verdict=verdict, reason=reason, revised=verdict == "revise")

    if verdict != "revise":
        # Critic said pass (or output was unparseable → defaulted to
        # pass). Final answer = initial answer.
        return CritiqueResult(
            initial_answer=initial,
            critique=critique_text,
            final_answer=initial,
            revised=False,
            n_revisions=0,
        )

    # ── Step 3: revise ───────────────────────────────────────────────
    try:
        revised_text = _revise(
            client=effective_revise_client,
            model=effective_revise_model,
            question=effective_question,
            context=context,
            initial_answer=initial,
            critique=critique_text,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("critique_revise revise call failed: %s", exc)
        # Reviser failed → keep initial as the final answer.
        return CritiqueResult(
            initial_answer=initial,
            critique=critique_text,
            final_answer=initial,
            revised=False,
            n_revisions=0,
        )

    if not revised_text:
        # Empty revision → fall back to initial. Better a confident
        # initial than a blank revised answer.
        return CritiqueResult(
            initial_answer=initial,
            critique=critique_text,
            final_answer=initial,
            revised=False,
            n_revisions=0,
        )

    _emit_revised_event(
        initial_length=len(initial),
        final_length=len(revised_text),
        n_revisions=1,
    )
    return CritiqueResult(
        initial_answer=initial,
        critique=critique_text,
        final_answer=revised_text,
        revised=True,
        n_revisions=1,
    )


# ──────────────────────────────────────────────────────────────────────
# Trace events (best-effort, no-ops when ATTESTOR_TRACE is unset)
# ──────────────────────────────────────────────────────────────────────


def _emit_verdict_event(*, verdict: str, reason: str, revised: bool) -> None:
    try:
        from attestor import trace as _tr
        if not _tr.is_enabled():
            return
        _tr.event(
            "answer.critique.verdict",
            verdict=verdict,
            reason=reason,
            revised=revised,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the call
        pass


def _emit_revised_event(
    *, initial_length: int, final_length: int, n_revisions: int,
) -> None:
    try:
        from attestor import trace as _tr
        if not _tr.is_enabled():
            return
        _tr.event(
            "answer.critique.revised",
            initial_length=initial_length,
            final_length=final_length,
            n_revisions=n_revisions,
        )
    except Exception:  # noqa: BLE001
        pass
