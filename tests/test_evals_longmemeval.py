"""Phase 9.2.1 — LME wrapper tests.

Pure unit tests for the summary extraction. No actual LME run.
"""

from __future__ import annotations

import pytest

from evals.longmemeval.runner import (
    _category_accuracies, _overall_accuracy, summarize,
)


# ── _overall_accuracy ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_overall_accuracy_averages_judges() -> None:
    by_judge = {
        "judge_a": {"correct": 80, "total": 100, "accuracy": 80.0},
        "judge_b": {"correct": 75, "total": 100, "accuracy": 75.0},
    }
    assert _overall_accuracy(by_judge) == pytest.approx(77.5)


@pytest.mark.unit
def test_overall_accuracy_skips_zero_total_judges() -> None:
    """A judge that scored zero samples shouldn't drag the average down."""
    by_judge = {
        "judge_a": {"correct": 80, "total": 100, "accuracy": 80.0},
        "judge_b": {"correct": 0, "total": 0, "accuracy": 0.0},
    }
    assert _overall_accuracy(by_judge) == pytest.approx(80.0)


@pytest.mark.unit
def test_overall_accuracy_empty_returns_zero() -> None:
    assert _overall_accuracy({}) == 0.0


# ── _category_accuracies ──────────────────────────────────────────────────


@pytest.mark.unit
def test_category_accuracies_averages_per_category_per_judge() -> None:
    by_category = {
        "single-session": {
            "judge_a": {"correct": 90, "total": 100, "accuracy": 90.0},
            "judge_b": {"correct": 80, "total": 100, "accuracy": 80.0},
        },
        "multi-session": {
            "judge_a": {"correct": 60, "total": 100, "accuracy": 60.0},
            "judge_b": {"correct": 50, "total": 100, "accuracy": 50.0},
        },
    }
    out = _category_accuracies(by_category)
    assert out["single-session"] == pytest.approx(85.0)
    assert out["multi-session"] == pytest.approx(55.0)


@pytest.mark.unit
def test_category_accuracies_omits_zero_sample_categories() -> None:
    """A category every judge scored zero on yields no signal — drop it."""
    by_category = {
        "single-session": {
            "judge_a": {"correct": 90, "total": 100, "accuracy": 90.0},
        },
        "abstention": {
            "judge_a": {"correct": 0, "total": 0, "accuracy": 0.0},
        },
    }
    out = _category_accuracies(by_category)
    assert "single-session" in out
    assert "abstention" not in out


# ── summarize: dataclass + dict input ─────────────────────────────────────


@pytest.mark.unit
def test_summarize_from_dict_form() -> None:
    """summarize must accept the JSON-loaded form, not just the dataclass."""
    report_dict = {
        "total": 100,
        "answer_model": "openai/gpt-4.1-mini",
        "judge_models": ["judge_a", "judge_b"],
        "by_judge": {
            "judge_a": {"correct": 80, "total": 100, "accuracy": 80.0},
            "judge_b": {"correct": 75, "total": 100, "accuracy": 75.0},
        },
        "by_category": {
            "single-session": {
                "judge_a": {"correct": 40, "total": 50, "accuracy": 80.0},
                "judge_b": {"correct": 38, "total": 50, "accuracy": 76.0},
            },
            "abstention": {
                "judge_a": {"correct": 40, "total": 50, "accuracy": 80.0},
                "judge_b": {"correct": 37, "total": 50, "accuracy": 74.0},
            },
        },
    }
    s = summarize(report_dict)
    assert s.benchmark == "longmemeval"
    assert s.primary_metric == pytest.approx(77.5)
    assert s.primary_metric_name == "accuracy_pct"
    assert s.total == 100
    assert s.metadata["answer_model"] == "openai/gpt-4.1-mini"
    assert s.metadata["judges"] == ["judge_a", "judge_b"]
    assert s.per_category["single-session"] == pytest.approx(78.0)
    assert s.per_category["abstention"] == pytest.approx(77.0)


@pytest.mark.unit
def test_summarize_from_dataclass_form() -> None:
    """summarize must accept the in-memory dataclass form too."""

    class _FakeReport:
        total = 50
        answer_model = "openai/gpt-4.1-mini"
        judge_models = ("judge_a",)
        by_judge = {"judge_a": {"correct": 30, "total": 50, "accuracy": 60.0}}
        by_category = {
            "temporal": {
                "judge_a": {"correct": 30, "total": 50, "accuracy": 60.0},
            },
        }

    s = summarize(_FakeReport())
    assert s.primary_metric == pytest.approx(60.0)
    assert s.per_category == {"temporal": pytest.approx(60.0)}
