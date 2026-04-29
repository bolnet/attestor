"""Self-consistency K-sample answerer (Phase 3 PR-B, +3-6% LME-S).

A single greedy decode from the answerer model can lock onto a wrong
answer when the question is ambiguous or the retrieved context is
noisy. K independent samples drawn at non-zero temperature plus a
tie-break heuristic (majority vote on a normalized fingerprint, or a
delegated judge LLM) almost always elects the correct answer when the
corpus contains it. Classic technique from Wang et al. 2022.

This module is answerer-side only — it does NOT touch the retrieval
pipeline or ``RetrievalCfg``. The orchestrator runs unchanged; we
just call the answerer K times instead of once and reduce the K
samples to one final answer.

Configuration lives in ``configs/attestor.yaml`` under the
top-level ``stack.self_consistency`` block (peer of ``stack.retrieval``,
not nested inside it):

    stack:
      self_consistency:
        enabled: false
        k: 5
        temperature: 0.7
        voter: majority           # majority | judge_pick
        judge_model: null         # null → models.judge

Trace events emitted (when ``ATTESTOR_TRACE=1``):

  - ``answer.self_consistency.samples`` — k, model, temperature,
    sample_lengths
  - ``answer.self_consistency.elected`` — voter, chosen_fingerprint,
    vote_breakdown

Cost: K × answerer cost. Disabled by default — gate behind the YAML
knob and flip per bench run only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("attestor.longmemeval_consistency")

# Strategies the loader + dispatcher accept. New entries added here
# automatically flow into the YAML validator.
VALID_VOTERS = ("majority", "judge_pick")


# ──────────────────────────────────────────────────────────────────────
# Result shape
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsistencyResult:
    """K answerer samples + the elected final answer.

    Immutable per the project coding-style rules. ``vote_breakdown``
    keys are normalized fingerprints (see ``_fingerprint``); values are
    raw counts. ``voter`` records which strategy actually produced
    ``chosen`` — judge_pick can fall back to majority on judge failure,
    in which case ``voter`` is reported as ``"majority"``.
    """

    samples: List[str]                       # all K raw answers (post-strip)
    chosen: str                              # the elected final answer
    voter: str                               # "majority" | "judge_pick"
    vote_breakdown: Dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Fingerprinting
# ──────────────────────────────────────────────────────────────────────

# Trailing punctuation we strip before fingerprinting so "Bob." and "Bob"
# collapse. Internal punctuation (e.g. "Bob, the CTO") is left intact.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!?\-—–'\"`]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\s\.,;:!?\-—–'\"`]+")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def _fingerprint(answer: str) -> str:
    """Canonicalize an answer string for majority-vote bucketing.

    Steps: lowercase → strip leading/trailing punctuation+whitespace →
    collapse internal whitespace → return. Two answers that differ
    only in casing, surrounding punctuation, or whitespace collapse to
    the same fingerprint.

    Empty strings (after canonicalization) return ``""`` — callers
    should treat the empty fingerprint as "no answer".
    """
    if not answer:
        return ""
    s = answer.lower()
    s = _LEADING_PUNCT_RE.sub("", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _WS_COLLAPSE_RE.sub(" ", s).strip()
    return s


# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────


def _sample_once(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> str:
    """Draw a single answerer sample. Returns the stripped content text,
    or ``""`` on any error. Wrapped in ``traced_create`` so each draw
    emits a ``chat.completion`` event for cost tracking."""
    from attestor.llm_trace import traced_create

    response = traced_create(
        client,
        role="answerer",
        model=model,
        temperature=temperature,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Voters
# ──────────────────────────────────────────────────────────────────────


def _majority_choice(samples: List[str]) -> tuple[str, Dict[str, int]]:
    """Pick the most-frequent fingerprint; tiebreak by first-seen.

    Returns ``(chosen_surface_form, vote_breakdown)``. The chosen
    surface form is the first sample whose fingerprint matched the
    winning bucket — preserves user-facing capitalization/punctuation.
    """
    if not samples:
        return "", {}

    # Count fingerprints in stable insertion order (first-seen wins
    # tiebreaks via dict ordering on Python 3.7+).
    counts: Dict[str, int] = {}
    first_surface: Dict[str, str] = {}
    for s in samples:
        fp = _fingerprint(s)
        if not fp:
            continue
        counts[fp] = counts.get(fp, 0) + 1
        if fp not in first_surface:
            first_surface[fp] = s

    if not counts:
        return "", {}

    # max() on dict.items() with a key is stable on first-seen — Python
    # iterates in insertion order, and max returns the first item with
    # the maximum value. That's exactly the tiebreak we want.
    winning_fp, _ = max(counts.items(), key=lambda kv: kv[1])
    return first_surface[winning_fp], counts


_JUDGE_PROMPT = (
    "You are picking the best of {n} candidate answers to the question below. "
    "Output a single integer 0..{n_minus_1} indicating the index of the best "
    "candidate — nothing else, no explanation, no punctuation.\n\n"
    "Question:\n{question}\n\n"
    "Candidates:\n{candidates}\n\n"
    "Best index:"
)


def _judge_choice(
    *,
    samples: List[str],
    question: str,
    judge_client: Any,
    judge_model: str,
) -> int:
    """Ask a judge LLM to pick the best of K candidate samples by index.

    Returns the chosen index. Raises on judge error or unparseable
    output — callers handle the fallback to majority.
    """
    from attestor.llm_trace import traced_create

    candidates = "\n".join(f"[{i}] {s}" for i, s in enumerate(samples))
    prompt = _JUDGE_PROMPT.format(
        n=len(samples),
        n_minus_1=len(samples) - 1,
        question=question,
        candidates=candidates,
    )
    response = traced_create(
        judge_client,
        role="self_consistency_judge",
        model=judge_model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    # Extract the first integer from the response.
    m = re.search(r"-?\d+", text)
    if not m:
        raise ValueError(f"judge returned unparseable index: {text!r}")
    idx = int(m.group(0))
    if idx < 0 or idx >= len(samples):
        raise ValueError(f"judge index {idx} out of range [0, {len(samples)})")
    return idx


def _question_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Best-effort extraction of the user's question from the chat
    messages — used only for the judge prompt context. Defensive: if
    the schema doesn't match, falls back to a generic placeholder."""
    if not messages:
        return "(no question available)"
    # Last user message wins.
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content[:4000]  # cap to keep judge prompt bounded
    return str(messages[-1].get("content", ""))[:4000]


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def answer_with_self_consistency(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    k: int = 5,
    temperature: float = 0.7,
    voter: str = "majority",
    judge_client: Optional[Any] = None,
    judge_model: Optional[str] = None,
) -> ConsistencyResult:
    """Run the answerer model K times and elect the consensus answer.

    Args:
        client: OpenAI-compatible chat-completions client (the answerer
            client). Each of the K samples calls
            ``client.chat.completions.create``.
        model: Answerer model id.
        messages: Messages list passed verbatim to every sample call.
        k: Number of independent samples to draw. K=0 returns an empty
            result without making any LLM call. K=1 effectively reduces
            to a single greedy decode.
        temperature: Per-sample temperature. Non-zero temperature is
            essential — at T=0 every sample collapses to the greedy
            decode and the vote degenerates.
        voter: "majority" (default) buckets samples by normalized
            fingerprint and picks the most-frequent answer with a
            first-sample tiebreak. "judge_pick" delegates to a judge
            LLM that picks the best of the K candidates by index;
            judge failures fall back to majority.
        judge_client: Optional judge LLM client for ``voter="judge_pick"``.
            Defaults to ``client`` (reuse the answerer client).
        judge_model: Required when ``voter="judge_pick"``. Should
            normally come from ``stack.models.judge``.

    Returns:
        ``ConsistencyResult`` with the K raw samples, the elected
        ``chosen`` answer, the actual voter strategy that produced it
        (judge_pick can fall back to "majority" on judge error), and
        the vote breakdown for diagnostics.

    Defensive contract: on any error (K=0, all samples empty, model
    unreachable, all samples fail), returns an empty ``chosen`` with
    whatever samples did succeed. Never raises — except for invalid
    voter strategies, which surface as ``ValueError`` because they're
    a programming error.
    """
    if voter not in VALID_VOTERS:
        raise ValueError(
            f"unknown voter strategy {voter!r}; expected one of {VALID_VOTERS}"
        )

    if k <= 0:
        return ConsistencyResult(samples=[], chosen="", voter=voter, vote_breakdown={})

    samples: List[str] = []
    for i in range(k):
        try:
            text = _sample_once(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 — sampling errors must not abort the loop
            logger.debug("self_consistency sample %d failed: %s", i, exc)
            continue
        if text:
            samples.append(text)

    # Telemetry: emit a samples event regardless of whether we found a
    # winner. Useful for cost accounting + debugging when all samples
    # came back empty.
    _emit_samples_event(
        k=k, model=model, temperature=temperature, samples=samples,
    )

    if not samples:
        return ConsistencyResult(samples=[], chosen="", voter=voter, vote_breakdown={})

    if voter == "judge_pick" and judge_model:
        try:
            idx = _judge_choice(
                samples=samples,
                question=_question_from_messages(messages),
                judge_client=judge_client or client,
                judge_model=judge_model,
            )
            chosen = samples[idx]
            breakdown = {_fingerprint(s): 1 for s in samples}
            _emit_elected_event(
                voter="judge_pick", chosen=chosen, breakdown=breakdown,
            )
            return ConsistencyResult(
                samples=samples, chosen=chosen,
                voter="judge_pick", vote_breakdown=breakdown,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to majority
            logger.debug("self_consistency judge_pick failed; falling back: %s", exc)
            # Fall through to majority path below.

    # Majority vote path (default + judge_pick fallback).
    chosen, breakdown = _majority_choice(samples)
    _emit_elected_event(
        voter="majority", chosen=chosen, breakdown=breakdown,
    )
    return ConsistencyResult(
        samples=samples, chosen=chosen,
        voter="majority", vote_breakdown=breakdown,
    )


# ──────────────────────────────────────────────────────────────────────
# Trace events (best-effort, no-ops when ATTESTOR_TRACE is unset)
# ──────────────────────────────────────────────────────────────────────


def _emit_samples_event(
    *, k: int, model: str, temperature: float, samples: List[str],
) -> None:
    try:
        from attestor import trace as _tr
        if not _tr.is_enabled():
            return
        _tr.event(
            "answer.self_consistency.samples",
            k=k,
            model=model,
            temperature=temperature,
            sample_lengths=[len(s) for s in samples],
            sample_count=len(samples),
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the call
        pass


def _emit_elected_event(
    *, voter: str, chosen: str, breakdown: Dict[str, int],
) -> None:
    try:
        from attestor import trace as _tr
        if not _tr.is_enabled():
            return
        _tr.event(
            "answer.self_consistency.elected",
            voter=voter,
            chosen_fingerprint=_fingerprint(chosen),
            vote_breakdown=dict(breakdown),
        )
    except Exception:  # noqa: BLE001
        pass
