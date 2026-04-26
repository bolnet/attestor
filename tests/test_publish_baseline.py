"""Phase 9.6.2 — publish_baseline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.baseline import load_baseline, write_baseline
from evals.publish_baseline import (
    _format_diff, main, merge_summaries,
)
from evals.summary import BenchmarkSummary


def _write_summary(dirpath: Path, name: str, metric: float, **kw) -> None:
    s = BenchmarkSummary(benchmark=name, primary_metric=metric, **kw)
    (dirpath / f"{name}_summary.json").write_text(json.dumps(s.to_dict()))


# ── merge_summaries ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_merge_replaces_existing_benchmark() -> None:
    existing = {"lme": BenchmarkSummary(benchmark="lme", primary_metric=70.0)}
    new = [BenchmarkSummary(benchmark="lme", primary_metric=75.0)]
    out = merge_summaries(existing, new)
    assert len(out) == 1
    assert out[0].primary_metric == 75.0


@pytest.mark.unit
def test_merge_preserves_unmentioned_benchmarks() -> None:
    """A partial run shouldn't silently delete other benchmarks."""
    existing = {
        "lme":        BenchmarkSummary(benchmark="lme", primary_metric=70.0),
        "abstention": BenchmarkSummary(benchmark="abstention", primary_metric=85.0),
    }
    new = [BenchmarkSummary(benchmark="lme", primary_metric=72.0)]
    out = {s.benchmark: s for s in merge_summaries(existing, new)}
    assert out["lme"].primary_metric == 72.0
    assert out["abstention"].primary_metric == 85.0  # preserved


@pytest.mark.unit
def test_merge_adds_new_benchmark() -> None:
    existing = {"lme": BenchmarkSummary(benchmark="lme", primary_metric=70.0)}
    new = [BenchmarkSummary(benchmark="beam", primary_metric=60.0)]
    out = {s.benchmark: s for s in merge_summaries(existing, new)}
    assert "lme" in out and "beam" in out


@pytest.mark.unit
def test_merge_with_only_filters_promotions() -> None:
    """--only restricts which benchmarks are promoted."""
    existing = {"lme": BenchmarkSummary(benchmark="lme", primary_metric=70.0)}
    new = [
        BenchmarkSummary(benchmark="lme",  primary_metric=80.0),
        BenchmarkSummary(benchmark="beam", primary_metric=60.0),
    ]
    out = {s.benchmark: s for s in merge_summaries(existing, new, only=("lme",))}
    assert out["lme"].primary_metric == 80.0
    assert "beam" not in out  # filtered out


# ── _format_diff ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_diff_marks_replacement() -> None:
    txt = _format_diff(
        {"lme": BenchmarkSummary(benchmark="lme", primary_metric=70.0)},
        [BenchmarkSummary(benchmark="lme", primary_metric=75.0)],
    )
    assert "~ lme" in txt and "+5.00" in txt


@pytest.mark.unit
def test_format_diff_marks_addition() -> None:
    txt = _format_diff(
        {},
        [BenchmarkSummary(benchmark="abstention", primary_metric=85.0)],
    )
    assert "+ abstention" in txt and "85.00" in txt


@pytest.mark.unit
def test_format_diff_marks_removal() -> None:
    """Showing removals is important — operator may have intended a
    partial run and accidentally pruned the baseline."""
    txt = _format_diff(
        {"old": BenchmarkSummary(benchmark="old", primary_metric=50.0)},
        [],
    )
    assert "- old" in txt and "removing" in txt


@pytest.mark.unit
def test_format_diff_empty_says_so() -> None:
    assert "(no benchmarks)" in _format_diff({}, [])


# ── CLI: main ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_main_writes_new_baseline(tmp_path: Path) -> None:
    summaries = tmp_path / "out"
    summaries.mkdir()
    _write_summary(summaries, "lme", 75.0)
    baseline = tmp_path / "v4-baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    code = main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
    ])
    assert code == 0
    loaded = load_baseline(baseline)
    assert loaded["lme"].primary_metric == 75.0


@pytest.mark.unit
def test_main_dry_run_does_not_modify(tmp_path: Path) -> None:
    summaries = tmp_path / "out"
    summaries.mkdir()
    _write_summary(summaries, "lme", 90.0)  # huge promotion
    baseline = tmp_path / "v4-baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    code = main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
        "--dry-run",
    ])
    assert code == 0
    loaded = load_baseline(baseline)
    assert loaded["lme"].primary_metric == 70.0  # unchanged


@pytest.mark.unit
def test_main_preserves_threshold_overrides(tmp_path: Path) -> None:
    """Promoting a baseline should not nuke configured per-benchmark
    thresholds — operators tune those once and rarely revisit."""
    summaries = tmp_path / "out"
    summaries.mkdir()
    _write_summary(summaries, "abstention", 85.0)
    baseline = tmp_path / "v4-baseline.json"
    write_baseline(
        baseline,
        [BenchmarkSummary(benchmark="abstention", primary_metric=80.0)],
        default_threshold=2.0,
        thresholds={"abstention": 1.0, "regression": 0.5},
    )

    main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
    ])
    raw = json.loads(baseline.read_text())
    assert raw["thresholds"]["abstention"] == 1.0
    assert raw["thresholds"]["regression"] == 0.5
    assert raw["default_threshold"] == 2.0


@pytest.mark.unit
def test_main_default_threshold_override(tmp_path: Path) -> None:
    summaries = tmp_path / "out"
    summaries.mkdir()
    _write_summary(summaries, "lme", 70.0)
    baseline = tmp_path / "v4-baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
        "--default-threshold", "3.5",
    ])
    raw = json.loads(baseline.read_text())
    assert raw["default_threshold"] == 3.5


@pytest.mark.unit
def test_main_only_flag_filters(tmp_path: Path) -> None:
    summaries = tmp_path / "out"
    summaries.mkdir()
    _write_summary(summaries, "lme",  72.0)
    _write_summary(summaries, "beam", 60.0)
    baseline = tmp_path / "v4-baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
        "--only", "lme",
    ])
    loaded = load_baseline(baseline)
    assert loaded["lme"].primary_metric == 72.0
    assert "beam" not in loaded  # filtered


@pytest.mark.unit
def test_main_returns_one_when_no_summaries(tmp_path: Path) -> None:
    """Empty summaries dir = nothing to do; exit 1 so CI scripts notice."""
    summaries = tmp_path / "empty"
    summaries.mkdir()
    baseline = tmp_path / "v4-baseline.json"
    code = main([
        "--summaries-dir", str(summaries),
        "--baseline", str(baseline),
    ])
    assert code == 1
    assert not baseline.exists()
