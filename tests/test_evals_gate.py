"""Phase 9.5.3 — gate CLI tests.

Pure unit tests; no GitHub Actions, no live runs. Verifies summary
loading from a directory, exit codes, and the integration with the
baseline comparator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.baseline import GateReport, write_baseline
from evals.gate import format_report, load_summaries, main
from evals.summary import BenchmarkSummary


def _write_summary(dirpath: Path, name: str,
                   metric: float, **extra) -> Path:
    p = dirpath / f"{name}_summary.json"
    s = BenchmarkSummary(benchmark=name, primary_metric=metric, **extra)
    p.write_text(json.dumps(s.to_dict()))
    return p


# ── load_summaries ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_summaries_from_directory(tmp_path: Path) -> None:
    _write_summary(tmp_path, "longmemeval", 70.0)
    _write_summary(tmp_path, "abstention", 85.0)
    out = load_summaries(tmp_path)
    benches = {s.benchmark for s in out}
    assert benches == {"longmemeval", "abstention"}


@pytest.mark.unit
def test_load_summaries_missing_dir_returns_empty(tmp_path: Path) -> None:
    out = load_summaries(tmp_path / "no_such_dir")
    assert out == []


@pytest.mark.unit
def test_load_summaries_skips_invalid_json(tmp_path: Path) -> None:
    """A garbage file shouldn't crash the gate — log + skip."""
    _write_summary(tmp_path, "good", 80.0)
    (tmp_path / "broken_summary.json").write_text("{not json")
    out = load_summaries(tmp_path)
    assert {s.benchmark for s in out} == {"good"}


@pytest.mark.unit
def test_load_summaries_skips_wrong_shape(tmp_path: Path) -> None:
    """A JSON file that isn't a BenchmarkSummary shape — also skip."""
    _write_summary(tmp_path, "good", 80.0)
    (tmp_path / "bad_summary.json").write_text(
        json.dumps({"some": "other shape"})
    )
    out = load_summaries(tmp_path)
    assert {s.benchmark for s in out} == {"good"}


@pytest.mark.unit
def test_load_summaries_ignores_non_summary_files(tmp_path: Path) -> None:
    """Only ``*_summary.json`` are picked up — raw reports stay out."""
    _write_summary(tmp_path, "lme", 70.0)
    (tmp_path / "lme_report.json").write_text("anything")
    (tmp_path / "notes.txt").write_text("ignore")
    assert {s.benchmark for s in load_summaries(tmp_path)} == {"lme"}


# ── format_report ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_report_empty_says_default_pass() -> None:
    txt = format_report(GateReport())
    assert "PASS" in txt
    assert "default" in txt.lower()


@pytest.mark.unit
def test_format_report_blocked_marker_appears() -> None:
    from evals.baseline import BenchmarkDelta
    rep = GateReport(
        deltas=(BenchmarkDelta(
            benchmark="lme", baseline=70.0, current=65.0,
            delta=-5.0, threshold=2.0, regressed=True,
            note="primary metric dropped 5.00pt",
        ),),
        blocked=True,
    )
    txt = format_report(rep)
    assert "BLOCKED" in txt
    assert "REGRESSED" in txt
    assert "lme" in txt


# ── main: exit codes ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_main_exits_zero_on_pass(tmp_path: Path) -> None:
    summaries_dir = tmp_path / "out"
    summaries_dir.mkdir()
    _write_summary(summaries_dir, "lme", 70.0)
    baseline = tmp_path / "baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    code = main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(baseline),
    ])
    assert code == 0


@pytest.mark.unit
def test_main_exits_one_on_regression(tmp_path: Path) -> None:
    summaries_dir = tmp_path / "out"
    summaries_dir.mkdir()
    _write_summary(summaries_dir, "lme", 60.0)  # dropped 10pt
    baseline = tmp_path / "baseline.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    code = main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(baseline),
    ])
    assert code == 1


@pytest.mark.unit
def test_main_writes_report_when_requested(tmp_path: Path) -> None:
    summaries_dir = tmp_path / "out"
    summaries_dir.mkdir()
    _write_summary(summaries_dir, "lme", 70.0)
    baseline = tmp_path / "baseline.json"
    report_out = tmp_path / "gate.json"
    write_baseline(baseline,
                   [BenchmarkSummary(benchmark="lme", primary_metric=70.0)])

    main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(baseline),
        "--report-out", str(report_out),
    ])
    assert report_out.exists()
    payload = json.loads(report_out.read_text())
    assert payload["blocked"] is False


@pytest.mark.unit
def test_main_passes_when_baseline_missing(tmp_path: Path) -> None:
    """Bootstrap mode: no baseline file → never block."""
    summaries_dir = tmp_path / "out"
    summaries_dir.mkdir()
    _write_summary(summaries_dir, "lme", 30.0)  # would be huge regression

    code = main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(tmp_path / "no_baseline_yet.json"),
    ])
    assert code == 0


@pytest.mark.unit
def test_main_passes_when_no_summaries(tmp_path: Path) -> None:
    """Empty summaries dir → no benchmarks ran → gate passes by default."""
    code = main([
        "--summaries-dir", str(tmp_path / "empty_or_missing"),
        "--baseline", str(tmp_path / "no_baseline.json"),
    ])
    assert code == 0


@pytest.mark.unit
def test_main_block_on_category_flag(tmp_path: Path) -> None:
    """--block-on-category-regression escalates per-category drift to a block."""
    summaries_dir = tmp_path / "out"
    summaries_dir.mkdir()
    _write_summary(
        summaries_dir, "lme", 70.0,
        per_category={"multi": 50.0},  # baseline had 60.0
    )
    baseline = tmp_path / "baseline.json"
    write_baseline(baseline, [
        BenchmarkSummary(
            benchmark="lme", primary_metric=70.0,
            per_category={"multi": 60.0},
        ),
    ])

    code_default = main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(baseline),
    ])
    assert code_default == 0  # primary unchanged

    code_strict = main([
        "--summaries-dir", str(summaries_dir),
        "--baseline", str(baseline),
        "--block-on-category-regression",
    ])
    assert code_strict == 1
