"""Phase 9.3 — BEAM scorer + runner tests.

Pure unit tests. No dataset, no DB, no LLM. Verifies normalization,
exact/substring matching, bucket assignment, aggregation, and the
runner's per-sample error isolation.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from evals.beam.runner import (
    DefaultDatasetLoader, _run_one, run, summarize,
)
from evals.beam.scorer import (
    _normalize, aggregate, exact_match,
    score_prediction, substring_match,
)
from evals.beam.types import (
    DEFAULT_BUCKETS, BeamPrediction, BeamRunReport, BeamSample,
    bucket_for,
)


# ── Bucket assignment ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_bucket_for_small_context() -> None:
    assert bucket_for(500) == "1k"


@pytest.mark.unit
def test_bucket_for_8k_range() -> None:
    assert bucket_for(8_000) == "8k"


@pytest.mark.unit
def test_bucket_for_million_token_context() -> None:
    assert bucket_for(900_000) == "1M"


@pytest.mark.unit
def test_bucket_for_overflow_returns_overflow() -> None:
    """Above the highest bucket → labeled, not silently dropped."""
    assert bucket_for(5_000_000) == "overflow"


# ── Normalization + matchers ──────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert _normalize("  Hello, World! ") == "hello world"


@pytest.mark.unit
def test_normalize_collapses_whitespace() -> None:
    assert _normalize("foo\n\tbar    baz") == "foo bar baz"


@pytest.mark.unit
def test_exact_match_after_normalization() -> None:
    assert exact_match("Paris.", "paris") is True
    assert exact_match("paris, france", "paris") is False


@pytest.mark.unit
def test_substring_match_after_normalization() -> None:
    assert substring_match(
        "The capital is Paris, France.", "paris",
    ) is True


@pytest.mark.unit
def test_substring_match_rejects_when_gold_absent() -> None:
    assert substring_match(
        "The capital is Berlin.", "paris",
    ) is False


# ── score_prediction ──────────────────────────────────────────────────────


def _pred(text: str = "", *, error: str = None,
          sample_id: str = "s1", bucket: str = "1k",
          category: str = "needle") -> BeamPrediction:
    return BeamPrediction(
        sample_id=sample_id, predicted_answer=text,
        bucket=bucket, category=category, error=error,
    )


@pytest.mark.unit
def test_score_prediction_substring_passes() -> None:
    p = score_prediction(_pred("answer is paris"), "Paris", metric="substring")
    assert p.correct is True


@pytest.mark.unit
def test_score_prediction_exact_strict() -> None:
    p_loose = score_prediction(_pred("answer is paris"), "paris", metric="exact")
    assert p_loose.correct is False
    p_tight = score_prediction(_pred("paris"), "paris", metric="exact")
    assert p_tight.correct is True


@pytest.mark.unit
def test_score_prediction_error_marks_incorrect() -> None:
    """A prediction with an error string is always wrong, regardless of
    what predicted_answer happens to contain."""
    p = score_prediction(_pred("paris", error="ingest failed"), "paris")
    assert p.correct is False


@pytest.mark.unit
def test_score_prediction_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        score_prediction(_pred("p"), "p", metric="bm25")


# ── aggregate ─────────────────────────────────────────────────────────────


def _sample(sid: str, *, gold: str, tokens: int = 1500,
            cat: str = "needle") -> BeamSample:
    return BeamSample(
        sample_id=sid, context="haystack", query="?", answer=gold,
        token_count=tokens, category=cat,
    )


@pytest.mark.unit
def test_aggregate_overall_accuracy() -> None:
    samples = [_sample("a", gold="paris"), _sample("b", gold="berlin")]
    preds = [_pred("paris", sample_id="a"), _pred("rome", sample_id="b")]
    rep = aggregate(samples, preds)
    assert rep.total == 2
    assert rep.correct == 1
    assert rep.accuracy == 50.0


@pytest.mark.unit
def test_aggregate_buckets_by_token_count() -> None:
    samples = [
        _sample("small", gold="x", tokens=500),       # 1k bucket
        _sample("large", gold="x", tokens=200_000),   # 128k bucket
    ]
    preds = [_pred("x", sample_id="small"), _pred("y", sample_id="large")]
    rep = aggregate(samples, preds)
    assert rep.by_bucket["1k"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    assert rep.by_bucket["128k"] == {"correct": 0, "total": 1, "accuracy": 0.0}


@pytest.mark.unit
def test_aggregate_categorizes_by_sample_category() -> None:
    samples = [
        _sample("a", gold="x", cat="needle"),
        _sample("b", gold="y", cat="multi_hop"),
        _sample("c", gold="z", cat="needle"),
    ]
    preds = [
        _pred("x", sample_id="a", category="needle"),
        _pred("y", sample_id="b", category="multi_hop"),
        _pred("wrong", sample_id="c", category="needle"),
    ]
    rep = aggregate(samples, preds)
    assert rep.by_category["needle"]["accuracy"] == 50.0
    assert rep.by_category["multi_hop"]["accuracy"] == 100.0


@pytest.mark.unit
def test_aggregate_records_missing_prediction_as_wrong() -> None:
    """If the runner dropped a sample, the report must reflect it as a
    failure with an explanatory error — not silently shrink ``total``."""
    samples = [_sample("a", gold="x"), _sample("missing", gold="y")]
    preds = [_pred("x", sample_id="a")]
    rep = aggregate(samples, preds)
    assert rep.total == 2
    miss = next(p for p in rep.predictions if p.sample_id == "missing")
    assert miss.correct is False
    assert "missing prediction" in (miss.error or "")


@pytest.mark.unit
def test_aggregate_propagates_bucket_when_missing_in_prediction() -> None:
    """A bare prediction (bucket='') should be re-bucketed from the sample."""
    samples = [_sample("a", gold="x", tokens=100_000)]
    preds = [BeamPrediction(sample_id="a", predicted_answer="x")]
    rep = aggregate(samples, preds)
    assert rep.predictions[0].bucket == "128k"


# ── _run_one (per-sample error isolation) ────────────────────────────────


class _FakeMem:
    closed = False

    def close(self) -> None:
        self.__class__.closed = True


@pytest.mark.unit
def test_run_one_records_ingest_error() -> None:
    def bad_ingest(s, m): raise RuntimeError("ingest blew up")
    def answer(s, m): return ""

    p = _run_one(_sample("a", gold="x"),
                 mem_factory=lambda: _FakeMem(),
                 ingest=bad_ingest, answer=answer)
    assert p.error is not None and "RuntimeError" in p.error


@pytest.mark.unit
def test_run_one_records_answer_error() -> None:
    def good_ingest(s, m): pass
    def bad_answer(s, m): raise ValueError("no answerer")

    p = _run_one(_sample("a", gold="x"),
                 mem_factory=lambda: _FakeMem(),
                 ingest=good_ingest, answer=bad_answer)
    assert p.error is not None and "ValueError" in p.error


@pytest.mark.unit
def test_run_one_closes_mem_even_on_error() -> None:
    _FakeMem.closed = False

    def good_ingest(s, m): pass
    def bad_answer(s, m): raise RuntimeError("boom")

    _run_one(_sample("a", gold="x"),
             mem_factory=lambda: _FakeMem(),
             ingest=good_ingest, answer=bad_answer)
    assert _FakeMem.closed is True


# ── DefaultDatasetLoader ─────────────────────────────────────────────────


@pytest.mark.unit
def test_default_loader_fails_loud() -> None:
    """Operators MUST wire their own loader; the default explodes."""
    with pytest.raises(NotImplementedError, match="not configured"):
        DefaultDatasetLoader()()


# ── End-to-end run() with stubs ──────────────────────────────────────────


@pytest.mark.unit
def test_run_end_to_end_with_stubs() -> None:
    samples = [
        _sample("a", gold="paris"),
        _sample("b", gold="berlin"),
    ]

    def loader(): return samples
    def ingest(s, m): pass
    def answer(s, m): return s.answer  # perfect answerer

    summary = run(
        mem_factory=lambda: _FakeMem(),
        ingest=ingest, answer=answer,
        loader=loader,
    )
    assert summary.benchmark == "beam"
    assert summary.primary_metric == 100.0
    assert summary.total == 2


@pytest.mark.unit
def test_run_sample_limit_truncates() -> None:
    samples = [_sample(f"s{i}", gold="x") for i in range(10)]

    def loader(): return samples
    def ingest(s, m): pass
    def answer(s, m): return "x"

    summary = run(
        mem_factory=lambda: _FakeMem(),
        ingest=ingest, answer=answer,
        loader=loader, sample_limit=3,
    )
    assert summary.total == 3


# ── summarize ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_summarize_uses_buckets_for_per_category() -> None:
    """For a 1M benchmark, the actionable signal is per-context-length
    accuracy. summarize wires the bucket breakdown into per_category."""
    rep = BeamRunReport(
        total=4, correct=3, accuracy=75.0,
        by_bucket={
            "1k":   {"correct": 2, "total": 2, "accuracy": 100.0},
            "128k": {"correct": 1, "total": 2, "accuracy": 50.0},
        },
        by_category={"needle": {"correct": 3, "total": 4, "accuracy": 75.0}},
    )
    s = summarize(rep)
    assert s.benchmark == "beam"
    assert s.primary_metric == 75.0
    assert s.per_category == {"1k": 100.0, "128k": 50.0}


@pytest.mark.unit
def test_summarize_omits_zero_total_buckets() -> None:
    rep = BeamRunReport(
        total=2, correct=2, accuracy=100.0,
        by_bucket={
            "1k":   {"correct": 2, "total": 2, "accuracy": 100.0},
            "1M":   {"correct": 0, "total": 0, "accuracy": 0.0},
        },
    )
    s = summarize(rep)
    assert "1k" in s.per_category
    assert "1M" not in s.per_category  # no signal
