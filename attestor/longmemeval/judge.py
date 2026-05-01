"""LLM-as-judge for LongMemEval.

Two judge variants:
  - ``judge_answer``: CORRECT/WRONG vs gold answer (all categories).
  - ``judge_personalization``: separate rubric for RECOMMENDATION-mode
    answers — credits tailoring to stored user facts, not literal match.

Also defines the per-sample judge concurrency cap (``_JUDGE_CONCURRENCY``)
that ``runner._process_sample`` uses to construct an ``asyncio.Semaphore``
per running event loop. The constant lives here because the limit is a
judge-side concern (provider burst limits, judge cost dominance).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from attestor.longmemeval.prompts import (
    JUDGE_PROMPT,
    PERSONALIZATION_JUDGE_PROMPT,
)


# ---------------------------------------------------------------------------
# Defaults — answerer/judge model identities
# ---------------------------------------------------------------------------
#
# These resolve from ``configs/attestor.yaml`` at import time so callers
# get the canonical YAML values without an extra wiring step. They live
# in this module (not in ``runner.py``) because ``judge_answer`` /
# ``judge_personalization`` use ``DEFAULT_MODEL`` as a function default;
# putting the constants alongside the consumers avoids a runner→judge
# import cycle at module load.


def _default_model() -> str:
    """Resolve the LME benchmark default model from
    ``configs/attestor.yaml``."""
    from attestor.config import get_stack
    return get_stack().models.benchmark_default


def _default_judges() -> tuple[str, ...]:
    """Resolve the dual-judge panel: the primary judge from the YAML
    plus the verifier (cross-family) — matches the canonical recommended
    ``Judge=gpt-4.1, Verifier=claude-sonnet-4-6`` pairing."""
    from attestor.config import get_stack
    s = get_stack()
    return (s.models.judge, s.models.verifier)


DEFAULT_MODEL = _default_model()
# Default dual-judge. Second judge anchors out answerer-judge collusion.
DEFAULT_JUDGES = _default_judges()
DEFAULT_PARALLEL = 4


# Robust label extraction — works on clean JSON, malformed JSON, or plain text.
_LABEL_FALLBACK_RE = re.compile(r"\b(CORRECT|WRONG)\b", re.IGNORECASE)


def _parse_judge_response(raw: str) -> tuple[str, str]:
    """Parse a judge response into ``(label, reasoning)``.

    Strategy:
      1. Try strict JSON parse.
      2. Extract JSON blob between the first ``{`` and last ``}`` and retry.
      3. Fall back to regex over the raw text; prefer the LAST label mention
         so trailing verdicts override in-reasoning quotations.

    Defaults to ``("WRONG", raw)`` if nothing matches — bias is conservative
    so bad judge output never inflates accuracy.
    """
    if not raw or not raw.strip():
        return "WRONG", ""
    text = raw.strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    candidates = [text]
    lb, rb = text.find("{"), text.rfind("}")
    if 0 <= lb < rb:
        candidates.append(text[lb : rb + 1])

    for blob in candidates:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip().upper()
            reasoning = str(obj.get("reasoning", "")).strip()
            if label in {"CORRECT", "WRONG"}:
                return label, reasoning

    matches = _LABEL_FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].upper(), text
    return "WRONG", text


@dataclass(frozen=True)
class JudgeResult:
    """Output of ``judge_answer`` — normalized label + reasoning + raw."""

    label: str  # "CORRECT" | "WRONG"
    correct: bool
    reasoning: str
    raw: str
    judge_model: str


def judge_personalization(
    question: str,
    expected: str,
    generated: str,
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 300,
) -> JudgeResult:
    """LLM judge for personalization quality on RECOMMENDATION-mode samples.

    Same shape as ``judge_answer`` so reporting paths can treat them
    uniformly. Robust JSON parsing — bad output defaults to WRONG so
    bad judge output never inflates the personalization score.
    """
    # Lazy import — keeps the import graph linear (judge.py → fixtures.py
    # for clients; runner.py also imports both judge.py and fixtures.py).
    from attestor.longmemeval.fixtures import _chat, _get_client, _get_client_for_model

    prompt = PERSONALIZATION_JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        generated=generated,
        context=context,
    )
    if api_key is not None:
        client = _get_client(api_key)
        clean_model = model
    else:
        client, clean_model = _get_client_for_model(model)
    raw = _chat(client, clean_model, prompt, max_tokens=max_tokens, role="judge")
    label, reasoning = _parse_judge_response(raw)
    return JudgeResult(
        label=label,
        correct=label == "CORRECT",
        reasoning=reasoning,
        raw=raw,
        judge_model=f"{model}__personalization",
    )


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    category: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 300,
) -> JudgeResult:
    """Use an LLM to score an AI answer against the gold answer.

    Robust against JSON drift — always returns a well-formed ``JudgeResult``.
    """
    # Lazy import — see ``judge_personalization`` for rationale.
    from attestor.longmemeval.fixtures import _chat, _get_client, _get_client_for_model

    prompt = JUDGE_PROMPT.format(
        category=category,
        question=question,
        expected=expected,
        generated=generated,
    )
    if api_key is not None:
        client = _get_client(api_key)
        clean_model = model
    else:
        client, clean_model = _get_client_for_model(model)
    raw = _chat(client, clean_model, prompt, max_tokens=max_tokens, role="judge")
    label, reasoning = _parse_judge_response(raw)
    return JudgeResult(
        label=label,
        correct=label == "CORRECT",
        reasoning=reasoning,
        raw=raw,
        judge_model=model,
    )


# Per-sample bound on concurrent judge LLM calls. Two reasons it's
# capped at 2: (a) judge providers' burst limits trip easily under
# unbounded fanout (Voyage / OpenRouter saw 429s during the 133q run,
# 2026-04-30), and (b) the judge prompt is small but the model is the
# most expensive in the pipeline — pacing reduces wallclock variance
# and cost spikes without materially extending wallclock at K=3.
#
# Constructed per-sample (not module-level) so the semaphore binds to
# the running event loop. Tests use ``asyncio.run`` per-case which
# replaces the loop, and a module-level semaphore would leak across
# loops and silently fail. The actual ``asyncio.Semaphore(...)`` call
# lives in ``runner._process_sample``; only the cap value is owned here.
_JUDGE_CONCURRENCY = 2


def _judgement_to_dict(j: JudgeResult) -> dict:
    return {
        "label": j.label,
        "correct": j.correct,
        "reasoning": j.reasoning,
        "judge_model": j.judge_model,
    }


def _safe_judge_dict(
    jm: str, result: JudgeResult | BaseException
) -> dict:
    """Normalize one judge outcome — JudgeResult or an exception — into a dict.

    Errors do NOT inflate accuracy: they are recorded as WRONG with a reason.
    """
    if isinstance(result, BaseException):
        return {
            "label": "WRONG",
            "correct": False,
            "reasoning": f"judge_error: {type(result).__name__}: {result}",
            "judge_model": jm,
        }
    return _judgement_to_dict(result)
