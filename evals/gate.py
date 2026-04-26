"""CI regression gate (Phase 9.5.1).

Load BenchmarkSummary JSONs from a directory, diff them against
``docs/bench/v4-baseline.json``, and exit with a non-zero status if any
benchmark regressed beyond the configured threshold.

Designed to be the single entry point CI calls:

    python -m evals.gate \\
        --summaries-dir bench-output/ \\
        --baseline docs/bench/v4-baseline.json \\
        --report-out bench-output/gate.json

The gate is permissive on bootstrap: if the baseline file is missing,
no regression is possible and the gate passes. That keeps the first
release from blocking itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from evals.baseline import (
    DEFAULT_THRESHOLD, GateReport, compare_to_baseline,
)
from evals.summary import BenchmarkSummary

logger = logging.getLogger("attestor.evals.gate")


# ── Loading summaries ────────────────────────────────────────────────────


def load_summaries(directory: Path | str) -> List[BenchmarkSummary]:
    """Load every ``*_summary.json`` file from a directory.

    Each file must be a single BenchmarkSummary (the shape every runner
    emits via ``summary.to_dict()``). Files with a different shape are
    skipped with a warning so a stray report file doesn't break the
    gate.
    """
    p = Path(directory)
    if not p.exists():
        logger.warning("summaries dir %s does not exist; treating as empty", p)
        return []
    out: List[BenchmarkSummary] = []
    for f in sorted(p.glob("*_summary.json")):
        try:
            raw = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            logger.warning("skipping %s: invalid JSON (%s)", f, e)
            continue
        try:
            out.append(BenchmarkSummary.from_dict(raw))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("skipping %s: bad summary shape (%s)", f, e)
            continue
    return out


# ── Pretty-printing ──────────────────────────────────────────────────────


def format_report(report: GateReport) -> str:
    """Render the gate verdict as human-readable text for the CI log."""
    lines: list[str] = []
    verdict = "BLOCKED" if report.blocked else "PASS"
    lines.append(f"Eval gate: {verdict} (threshold {report.threshold:.2f}pt)")
    lines.append("-" * 60)
    if not report.deltas:
        lines.append("No benchmark summaries provided. Gate passed by default.")
        return "\n".join(lines)
    for d in report.deltas:
        marker = "REGRESSED" if d.regressed else "ok"
        lines.append(
            f"{marker:>10}  {d.benchmark:<14}  "
            f"baseline={d.baseline:6.2f}  current={d.current:6.2f}  "
            f"Δ={d.delta:+6.2f}"
        )
        if d.note:
            lines.append(f"            note: {d.note}")
        for cat, base_v, cur_v, delta in d.per_category_regressions:
            lines.append(
                f"            └─ {cat}: {base_v:.2f} → {cur_v:.2f} ({delta:+.2f})"
            )
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="evals.gate")
    parser.add_argument(
        "--summaries-dir", required=True,
        help="Directory containing *_summary.json files.",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to the baseline JSON (e.g. docs/bench/v4-baseline.json).",
    )
    parser.add_argument(
        "--report-out", default=None,
        help="Optional path to write the gate report JSON.",
    )
    parser.add_argument(
        "--block-on-category-regression", action="store_true",
        help=(
            "Also block when a per-category metric regresses. By default, "
            "only the primary metric drives the gate."
        ),
    )
    args = parser.parse_args(argv)

    summaries = load_summaries(args.summaries_dir)
    report = compare_to_baseline(
        summaries, args.baseline,
        block_on_category_regression=args.block_on_category_regression,
    )

    print(format_report(report))

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2))

    return 1 if report.blocked else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
