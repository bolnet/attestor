"""Phase 9.2.2 — baseline comparator tests.

Pure logic tests for the regression gate. Verifies primary-metric drift
detection, per-category drift, threshold overrides, and first-run
behavior (no baseline = no block).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.baseline import (
    DEFAULT_THRESHOLD, GateReport, compare_summary,
    compare_to_baseline, load_baseline, write_baseline,
)
from evals.summary import BenchmarkSummary


# ── compare_summary: pure logic ───────────────────────────────────────────


@pytest.mark.unit
def test_compare_summary_no_baseline_does_not_regress() -> None:
    cur = BenchmarkSummary(benchmark="lme", primary_metric=72.0)
    d = compare_summary(cur, baseline=None)
    assert d.regressed is False
    assert d.delta == 72.0  # treated as new score
    assert "no baseline" in d.note


@pytest.mark.unit
def test_compare_summary_improvement_does_not_regress() -> None:
    base = BenchmarkSummary(benchmark="lme", primary_metric=70.0)
    cur = BenchmarkSummary(benchmark="lme", primary_metric=75.0)
    d = compare_summary(cur, base)
    assert d.regressed is False
    assert d.delta == pytest.approx(5.0)


@pytest.mark.unit
def test_compare_summary_small_drop_below_threshold_does_not_block() -> None:
    """Drift within the threshold is noise. No regression flag."""
    base = BenchmarkSummary(benchmark="lme", primary_metric=70.0)
    cur = BenchmarkSummary(benchmark="lme", primary_metric=68.5)  # -1.5 < 2.0
    d = compare_summary(cur, base, threshold=2.0)
    assert d.regressed is False


@pytest.mark.unit
def test_compare_summary_drop_exceeding_threshold_regresses() -> None:
    base = BenchmarkSummary(benchmark="lme", primary_metric=70.0)
    cur = BenchmarkSummary(benchmark="lme", primary_metric=67.0)  # -3.0 > 2.0
    d = compare_summary(cur, base, threshold=2.0)
    assert d.regressed is True
    assert "dropped 3.00pt" in d.note


@pytest.mark.unit
def test_compare_summary_per_category_regressions_recorded() -> None:
    """Per-category drift is reported even when the primary metric holds."""
    base = BenchmarkSummary(
        benchmark="lme", primary_metric=70.0,
        per_category={"single": 80.0, "multi": 60.0},
    )
    cur = BenchmarkSummary(
        benchmark="lme", primary_metric=70.0,
        per_category={"single": 80.0, "multi": 50.0},
    )
    d = compare_summary(cur, base, threshold=2.0)
    assert d.regressed is False  # primary metric unchanged
    assert len(d.per_category_regressions) == 1
    cat, base_v, cur_v, delta = d.per_category_regressions[0]
    assert cat == "multi" and delta == pytest.approx(-10.0)


# ── load_baseline / write_baseline ────────────────────────────────────────


@pytest.mark.unit
def test_load_baseline_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "nonexistent.json") == {}


@pytest.mark.unit
def test_write_then_load_baseline_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "v4-baseline.json"
    summaries = [
        BenchmarkSummary(benchmark="lme", primary_metric=72.0),
        BenchmarkSummary(
            benchmark="abstention", primary_metric=85.0,
            per_category={"unknown": 90.0},
        ),
    ]
    write_baseline(p, summaries, default_threshold=1.5,
                   thresholds={"abstention": 0.5})

    raw = json.loads(p.read_text())
    assert raw["default_threshold"] == 1.5
    assert raw["thresholds"]["abstention"] == 0.5

    loaded = load_baseline(p)
    assert set(loaded) == {"lme", "abstention"}
    assert loaded["lme"].primary_metric == 72.0
    assert loaded["abstention"].per_category["unknown"] == 90.0


# ── compare_to_baseline (the actual CI gate entry point) ─────────────────


@pytest.mark.unit
def test_gate_blocks_on_primary_regression(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    write_baseline(p, [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    # New run dropped below the default 2.0pt threshold
    rep = compare_to_baseline(
        [BenchmarkSummary(benchmark="lme", primary_metric=65.0)], p,
    )
    assert rep.blocked is True
    assert rep.deltas[0].regressed is True


@pytest.mark.unit
def test_gate_does_not_block_on_improvement(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    write_baseline(p, [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    rep = compare_to_baseline(
        [BenchmarkSummary(benchmark="lme", primary_metric=75.0)], p,
    )
    assert rep.blocked is False


@pytest.mark.unit
def test_gate_per_benchmark_threshold_override(tmp_path: Path) -> None:
    """Per-benchmark thresholds let unstable benchmarks have a looser
    gate without weakening the strict ones."""
    p = tmp_path / "baseline.json"
    write_baseline(p,
        [
            BenchmarkSummary(benchmark="strict", primary_metric=70.0),
            BenchmarkSummary(benchmark="loose", primary_metric=70.0),
        ],
        default_threshold=2.0,
        thresholds={"loose": 5.0},
    )

    # Both drop 3pt; only "strict" should regress (default 2pt threshold)
    rep = compare_to_baseline(
        [
            BenchmarkSummary(benchmark="strict", primary_metric=67.0),
            BenchmarkSummary(benchmark="loose", primary_metric=67.0),
        ], p,
    )
    by_b = {d.benchmark: d for d in rep.deltas}
    assert by_b["strict"].regressed is True
    assert by_b["loose"].regressed is False
    assert rep.blocked is True


@pytest.mark.unit
def test_gate_first_run_with_no_baseline_does_not_block(tmp_path: Path) -> None:
    """No baseline file → bootstrap mode. Record scores, never block."""
    rep = compare_to_baseline(
        [BenchmarkSummary(benchmark="lme", primary_metric=72.0)],
        tmp_path / "nope.json",
    )
    assert rep.blocked is False
    assert rep.deltas[0].current == 72.0


@pytest.mark.unit
def test_gate_block_on_category_regression_flag(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    write_baseline(p, [
        BenchmarkSummary(
            benchmark="lme", primary_metric=70.0,
            per_category={"multi": 60.0},
        ),
    ])

    cur = [BenchmarkSummary(
        benchmark="lme", primary_metric=70.0,
        per_category={"multi": 50.0},
    )]
    rep_default = compare_to_baseline(cur, p)
    assert rep_default.blocked is False  # primary unchanged → not blocked

    rep_strict = compare_to_baseline(cur, p, block_on_category_regression=True)
    assert rep_strict.blocked is True


@pytest.mark.unit
def test_gate_report_to_dict_serializable() -> None:
    rep = GateReport(
        deltas=(),
        blocked=False,
    )
    d = rep.to_dict()
    json.dumps(d)  # shouldn't raise
    assert d["blocked"] is False
