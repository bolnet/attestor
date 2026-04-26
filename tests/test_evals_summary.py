"""Phase 9.2 — BenchmarkSummary round-trip tests."""

from __future__ import annotations

import pytest

from evals.summary import BenchmarkSummary


@pytest.mark.unit
def test_summary_to_from_dict_round_trip() -> None:
    s = BenchmarkSummary(
        benchmark="longmemeval",
        primary_metric=72.5,
        primary_metric_name="accuracy_pct",
        per_category={"single-session": 80.0, "multi-session": 65.0},
        total=500,
        metadata={"answer_model": "openai/gpt-4.1-mini"},
    )
    d = s.to_dict()
    back = BenchmarkSummary.from_dict(d)
    assert back == s


@pytest.mark.unit
def test_summary_from_dict_handles_missing_optionals() -> None:
    """Old baselines may lack per_category/metadata. Loader should default."""
    s = BenchmarkSummary.from_dict({
        "benchmark": "regression",
        "primary_metric": 0.95,
    })
    assert s.benchmark == "regression"
    assert s.primary_metric == 0.95
    assert s.primary_metric_name == "accuracy"  # default
    assert s.per_category == {}
    assert s.metadata == {}


@pytest.mark.unit
def test_summary_is_frozen() -> None:
    s = BenchmarkSummary(benchmark="b", primary_metric=1.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        s.benchmark = "other"  # type: ignore[misc]
