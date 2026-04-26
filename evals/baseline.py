"""Baseline comparison + regression gate (Phase 9.2.2).

The CI gate works like this:

  1. Each runner emits a BenchmarkSummary (one per benchmark).
  2. ``compare_to_baseline`` diffs the new summary against
     ``docs/bench/v4-baseline.json``.
  3. If any score regresses by more than ``threshold`` (default 2.0
     points), the gate returns BLOCKED and CI fails the merge.

Why a separate module:
  - Pure logic — no I/O of any kind beyond JSON read/write.
  - Importable from CI scripts and the `attestor evals run` CLI.
  - The threshold is configurable per-benchmark in the baseline file
    itself, so an unstable benchmark can have a looser gate without
    weakening the strict ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from evals.summary import BenchmarkSummary


DEFAULT_THRESHOLD = 2.0  # absolute points (e.g., accuracy %)


# ── Result types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkDelta:
    """One benchmark's drift relative to baseline."""
    benchmark: str
    baseline: float
    current: float
    delta: float           # current - baseline; negative = regression
    threshold: float
    regressed: bool        # delta < -threshold
    per_category_regressions: tuple = ()  # tuple of (cat, baseline, current, delta)
    note: str = ""         # explanation when regressed=True


@dataclass(frozen=True)
class GateReport:
    """Aggregate verdict across all benchmarks."""
    deltas: tuple = ()
    blocked: bool = False
    threshold: float = DEFAULT_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "threshold": self.threshold,
            "deltas": [
                {
                    "benchmark": d.benchmark,
                    "baseline": d.baseline,
                    "current": d.current,
                    "delta": d.delta,
                    "threshold": d.threshold,
                    "regressed": d.regressed,
                    "per_category_regressions": [
                        {"category": c, "baseline": b, "current": cur, "delta": dl}
                        for (c, b, cur, dl) in d.per_category_regressions
                    ],
                    "note": d.note,
                }
                for d in self.deltas
            ],
        }


# ── I/O ───────────────────────────────────────────────────────────────────


def load_baseline(path: Path | str) -> Dict[str, BenchmarkSummary]:
    """Load a baseline JSON file. Missing file → empty baseline (first run).

    Schema:
        {
          "version": "1",
          "default_threshold": 2.0,
          "benchmarks": [ <BenchmarkSummary.to_dict()>, ... ],
          "thresholds": { "longmemeval": 2.0, "abstention": 1.0, ... }
        }
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    items = raw.get("benchmarks", []) or []
    return {b["benchmark"]: BenchmarkSummary.from_dict(b) for b in items}


def load_threshold_overrides(path: Path | str) -> Dict[str, float]:
    """Per-benchmark threshold overrides from the baseline file."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: float(v) for k, v in (raw.get("thresholds") or {}).items()}


def load_default_threshold(path: Path | str) -> float:
    p = Path(path)
    if not p.exists():
        return DEFAULT_THRESHOLD
    raw = json.loads(p.read_text())
    return float(raw.get("default_threshold", DEFAULT_THRESHOLD))


def write_baseline(
    path: Path | str,
    summaries: Iterable[BenchmarkSummary],
    *,
    default_threshold: float = DEFAULT_THRESHOLD,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Persist a new baseline. Used after a verified-good release run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "version": "1",
        "default_threshold": default_threshold,
        "thresholds": dict(thresholds or {}),
        "benchmarks": [s.to_dict() for s in summaries],
    }, indent=2))


# ── Comparison ────────────────────────────────────────────────────────────


def _per_category_regressions(
    baseline: BenchmarkSummary,
    current: BenchmarkSummary,
    threshold: float,
) -> List[tuple]:
    out = []
    for cat, base_v in baseline.per_category.items():
        cur_v = current.per_category.get(cat)
        if cur_v is None:
            continue
        delta = cur_v - base_v
        if delta < -threshold:
            out.append((cat, base_v, cur_v, delta))
    return out


def compare_summary(
    current: BenchmarkSummary,
    baseline: Optional[BenchmarkSummary],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> BenchmarkDelta:
    """Compare one current summary against its baseline."""
    if baseline is None:
        # No prior baseline → can't regress. Record the new score for
        # visibility but never block.
        return BenchmarkDelta(
            benchmark=current.benchmark,
            baseline=0.0, current=current.primary_metric,
            delta=current.primary_metric,
            threshold=threshold, regressed=False,
            note="no baseline (first run)",
        )
    delta = current.primary_metric - baseline.primary_metric
    regressed = delta < -threshold
    cat_regressions = tuple(
        _per_category_regressions(baseline, current, threshold)
    )
    note = ""
    if regressed:
        note = (
            f"primary metric dropped {abs(delta):.2f}pt "
            f"(threshold {threshold}pt)"
        )
    elif cat_regressions:
        cats = ", ".join(c for (c, _, _, _) in cat_regressions)
        note = f"per-category regressions in: {cats}"
    return BenchmarkDelta(
        benchmark=current.benchmark,
        baseline=baseline.primary_metric,
        current=current.primary_metric,
        delta=delta,
        threshold=threshold,
        regressed=regressed,
        per_category_regressions=cat_regressions,
        note=note,
    )


def compare_to_baseline(
    summaries: Iterable[BenchmarkSummary],
    baseline_path: Path | str,
    *,
    block_on_category_regression: bool = False,
) -> GateReport:
    """Diff every summary against its baseline; produce the gate verdict.

    If ``block_on_category_regression`` is True, a regression in any
    per-category score also blocks (default: only primary_metric).
    """
    baseline = load_baseline(baseline_path)
    overrides = load_threshold_overrides(baseline_path)
    default_threshold = load_default_threshold(baseline_path)

    deltas = []
    for s in summaries:
        thr = overrides.get(s.benchmark, default_threshold)
        deltas.append(compare_summary(s, baseline.get(s.benchmark), threshold=thr))

    blocked = any(d.regressed for d in deltas)
    if block_on_category_regression:
        blocked = blocked or any(d.per_category_regressions for d in deltas)

    return GateReport(
        deltas=tuple(deltas),
        blocked=blocked,
        threshold=default_threshold,
    )
