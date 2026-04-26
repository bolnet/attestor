"""BEAM scorer (Phase 9.3.2).

Long-context QA scoring is two-tier by convention:

  - exact_match  — strict. Used for needle-style benchmarks where the
                   gold is a specific short string and any deviation is
                   an error.
  - substring    — lenient. The gold appears anywhere inside the
                   prediction (after lowercase + whitespace normalize).
                   Standard for SQuAD-style answers and most public
                   long-context benchmarks.

Both metrics get reported. The primary metric for the gate is
configurable; substring is the safer default because it tolerates the
verbosity that recall-pack-fed answerers tend to produce.

Bucket aggregation slices accuracy by token-count buckets (see
DEFAULT_BUCKETS in types.py), which is the whole point of a 1M
benchmark — you want to see where the curve breaks.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

from evals.beam.types import (
    BeamPrediction, BeamRunReport, BeamSample,
    DEFAULT_BUCKETS, bucket_for,
)


# ── Normalization ─────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used to make exact/substring comparisons resilient to the formatting
    quirks of free-text generation. Same recipe used by SQuAD F1 and
    most long-context benchmarks.
    """
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


# ── Per-prediction scoring ────────────────────────────────────────────────


def exact_match(predicted: str, gold: str) -> bool:
    return _normalize(predicted) == _normalize(gold)


def substring_match(predicted: str, gold: str) -> bool:
    """Gold appears anywhere inside the (normalized) prediction."""
    g = _normalize(gold)
    if not g:
        # Empty gold trivially matches; flag as a data issue but don't crash.
        return True
    return g in _normalize(predicted)


def score_prediction(
    pred: BeamPrediction,
    gold: str,
    *,
    metric: str = "substring",
) -> BeamPrediction:
    """Return a new BeamPrediction with ``correct`` populated."""
    if pred.error is not None:
        return replace(pred, correct=False)
    if metric == "exact":
        ok = exact_match(pred.predicted_answer, gold)
    elif metric == "substring":
        ok = substring_match(pred.predicted_answer, gold)
    else:
        raise ValueError(
            f"unknown metric {metric!r}; expected 'exact' or 'substring'"
        )
    return replace(pred, correct=ok)


# ── Aggregation ───────────────────────────────────────────────────────────


def _bucket_stats(predictions: Iterable[BeamPrediction]) -> dict:
    counts: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in predictions:
        b = p.bucket or "unknown"
        counts[b]["total"] += 1
        if p.correct:
            counts[b]["correct"] += 1
    return {
        b: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": (
                round(100.0 * v["correct"] / v["total"], 2)
                if v["total"] else 0.0
            ),
        }
        for b, v in counts.items()
    }


def _category_stats(predictions: Iterable[BeamPrediction]) -> dict:
    counts: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in predictions:
        c = p.category or "general"
        counts[c]["total"] += 1
        if p.correct:
            counts[c]["correct"] += 1
    return {
        c: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": (
                round(100.0 * v["correct"] / v["total"], 2)
                if v["total"] else 0.0
            ),
        }
        for c, v in counts.items()
    }


def aggregate(
    samples: List[BeamSample],
    predictions: List[BeamPrediction],
    *,
    metric: str = "substring",
    buckets: Tuple[Tuple[str, int, int], ...] = DEFAULT_BUCKETS,
) -> BeamRunReport:
    """Score every (sample, prediction) pair and produce the run report.

    Pairs by ``sample_id``. A sample with no prediction is recorded as
    incorrect (the runner promises one prediction per sample; missing
    is a bug we want to surface, not silently drop).
    """
    pred_by_id = {p.sample_id: p for p in predictions}
    scored: List[BeamPrediction] = []
    for s in samples:
        p = pred_by_id.get(s.sample_id)
        if p is None:
            scored.append(BeamPrediction(
                sample_id=s.sample_id, predicted_answer="",
                bucket=bucket_for(s.token_count, buckets),
                category=s.category, correct=False,
                error="missing prediction",
            ))
            continue
        # Bucket and category come from the sample — they're properties
        # of the data, not the prediction. Overwriting any value the
        # answerer happened to set keeps the report authoritative.
        with_meta = replace(
            p,
            bucket=bucket_for(s.token_count, buckets),
            category=s.category,
        )
        scored.append(score_prediction(with_meta, s.answer, metric=metric))

    correct = sum(1 for p in scored if p.correct)
    total = len(scored)
    accuracy = round(100.0 * correct / total, 2) if total else 0.0

    return BeamRunReport(
        total=total,
        correct=correct,
        accuracy=accuracy,
        by_bucket=_bucket_stats(scored),
        by_category=_category_stats(scored),
        predictions=tuple(scored),
    )
