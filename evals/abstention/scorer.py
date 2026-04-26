"""AbstentionBench scorer (Phase 9.4.2).

Confusion matrix over the abstention DECISION (positive class = "abstain"):

  - Sample is unanswerable (gold: should abstain):
      response abstained → true_positive
      response answered  → false_negative   (confabulation)

  - Sample is answerable (gold: should answer):
      response abstained → false_positive   (over-abstention)
      response answered  → true_negative    (always); also checked for
                                              substring(expected_answer)
                                              for the answer-accuracy
                                              secondary metric

Why F1 as primary metric:
  Always-abstain → recall=1.0, precision=0.0 → F1=0.0
  Always-answer  → recall=0.0, precision=undef→0.0 → F1=0.0
  Both pathological extremes score zero. F1 only rewards calibration.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, List, Optional

from evals.abstention.detector import Detector, is_abstention
from evals.abstention.types import (
    AbstentionMetrics, AbstentionPrediction, AbstentionRunReport,
    AbstentionSample,
)

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _answer_matches(response: str, expected: Optional[str]) -> bool:
    """Substring match — same lenient rule BEAM uses."""
    if not expected:
        return True  # answerable case with no gold = data issue, don't crash
    return _normalize(expected) in _normalize(response)


# ── Per-prediction scoring ────────────────────────────────────────────────


def score_prediction(
    sample: AbstentionSample,
    pred: AbstentionPrediction,
    *,
    detector: Optional[Detector] = None,
) -> AbstentionPrediction:
    """Annotate prediction with abstained + correct fields.

    ``correct`` semantics here mean "did the model do the right thing?":
      - unanswerable + abstained  → True
      - unanswerable + answered   → False
      - answerable   + abstained  → False
      - answerable   + answered   → True iff substring(expected) hits

    A prediction with an error is always wrong and counts as having
    answered (not abstained) — a runtime crash is failure, not an
    intentional abstention.
    """
    if pred.error is not None:
        return replace(pred, abstained=False, correct=False,
                       answerable=sample.answerable, category=sample.category)

    abstained = (detector or is_abstention)(pred.response)

    if sample.answerable:
        if abstained:
            correct = False
        else:
            correct = _answer_matches(pred.response, sample.expected_answer)
    else:
        correct = abstained

    return replace(
        pred,
        abstained=abstained,
        correct=correct,
        answerable=sample.answerable,
        category=sample.category,
    )


# ── Aggregation ───────────────────────────────────────────────────────────


def _confusion(predictions: Iterable[AbstentionPrediction]) -> AbstentionMetrics:
    tp = fp = tn = fn = 0
    for p in predictions:
        # Unscored predictions get treated as wrong-answer (errors,
        # missing) — defensive, the runner should always populate these.
        abstained = bool(p.abstained)
        if not p.answerable:
            if abstained:
                tp += 1
            else:
                fn += 1
        else:
            if abstained:
                fp += 1
            else:
                tn += 1
    return AbstentionMetrics(
        true_positive=tp, false_positive=fp,
        true_negative=tn, false_negative=fn,
    )


def aggregate(
    samples: List[AbstentionSample],
    predictions: List[AbstentionPrediction],
    *,
    detector: Optional[Detector] = None,
) -> AbstentionRunReport:
    """Score every (sample, prediction) pair and produce the run report.

    Pairs by sample_id. A missing prediction is recorded as wrong (we
    surface bugs rather than silently shrink totals).
    """
    pred_by_id = {p.sample_id: p for p in predictions}
    scored: List[AbstentionPrediction] = []

    for s in samples:
        p = pred_by_id.get(s.sample_id)
        if p is None:
            scored.append(AbstentionPrediction(
                sample_id=s.sample_id, response="",
                abstained=False, correct=False,
                answerable=s.answerable, category=s.category,
                error="missing prediction",
            ))
            continue
        scored.append(score_prediction(s, p, detector=detector))

    overall = _confusion(scored)

    by_category: dict = {}
    grouped: dict = defaultdict(list)
    for p in scored:
        grouped[p.category or "general"].append(p)
    for cat, ps in grouped.items():
        by_category[cat] = _confusion(ps)

    # Secondary metric: of answerable cases that the model attempted,
    # how often did the substring match? Surfaces "abstention is fine
    # but the actual answers are bad" regressions.
    answer_total = sum(1 for p in scored if p.answerable and not p.abstained)
    answer_correct = sum(1 for p in scored
                         if p.answerable and not p.abstained and p.correct)
    answer_accuracy = (answer_correct / answer_total) if answer_total else 0.0

    return AbstentionRunReport(
        overall=overall,
        by_category=by_category,
        answer_accuracy=answer_accuracy,
        answer_correct=answer_correct,
        answer_total=answer_total,
        predictions=tuple(scored),
    )
