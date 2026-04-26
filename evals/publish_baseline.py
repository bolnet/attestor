"""Publish a new regression baseline (Phase 9.6.1).

Workflow:
  1. Operator triggers a heavy benchmark run via workflow_dispatch.
  2. Each runner emits ``*_summary.json`` into ``bench-output/``.
  3. Operator inspects the summaries and the gate verdict.
  4. If the new numbers are correct, ``publish_baseline`` writes them
     into ``docs/bench/v4-baseline.json``, preserving the configured
     thresholds.

Why a separate tool:
  - The gate (``evals.gate``) READS the baseline; this tool WRITES it.
  - Keeping them separate prevents accidental "tests passed → baseline
    silently updated" — promoting a baseline is a deliberate human act.
  - ``--dry-run`` lets the operator see exactly what would change
    before committing the new file to git.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from evals.baseline import (
    DEFAULT_THRESHOLD, load_baseline, load_default_threshold,
    load_threshold_overrides, write_baseline,
)
from evals.gate import load_summaries
from evals.summary import BenchmarkSummary

logger = logging.getLogger("attestor.evals.publish")


# ── Diff helpers ──────────────────────────────────────────────────────────


def _format_diff(
    old: dict,    # {benchmark: BenchmarkSummary}
    new: List[BenchmarkSummary],
) -> str:
    """Render a human-readable promotion diff for the CLI."""
    lines: List[str] = []
    new_by_b = {s.benchmark: s for s in new}
    all_keys = sorted(set(old) | set(new_by_b))
    if not all_keys:
        return "(no benchmarks)"
    for k in all_keys:
        o = old.get(k)
        n = new_by_b.get(k)
        if o is None and n is not None:
            lines.append(f"  + {k:<14}  new={n.primary_metric:6.2f}")
        elif n is None and o is not None:
            lines.append(f"  - {k:<14}  removing (was {o.primary_metric:6.2f})")
        else:
            delta = n.primary_metric - o.primary_metric
            lines.append(
                f"  ~ {k:<14}  {o.primary_metric:6.2f} -> "
                f"{n.primary_metric:6.2f}  ({delta:+.2f})"
            )
    return "\n".join(lines)


def merge_summaries(
    existing: dict,    # {benchmark: BenchmarkSummary}
    incoming: Iterable[BenchmarkSummary],
    *,
    only: Optional[Iterable[str]] = None,
) -> List[BenchmarkSummary]:
    """Compute the new baseline list.

    For every benchmark in ``incoming``, the new entry replaces the old
    one. Benchmarks present in ``existing`` but NOT in ``incoming`` are
    preserved as-is (a partial run shouldn't silently delete the rest
    of the baseline).

    ``only`` restricts which incoming benchmarks are promoted — useful
    for "publish just LME, leave the rest alone".
    """
    only_set = set(only) if only is not None else None
    new_map = dict(existing)
    for s in incoming:
        if only_set is not None and s.benchmark not in only_set:
            continue
        new_map[s.benchmark] = s
    return [new_map[k] for k in sorted(new_map)]


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.publish_baseline")
    parser.add_argument(
        "--summaries-dir", required=True,
        help="Directory of *_summary.json from the heavy benchmark run",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Baseline JSON to update (e.g. docs/bench/v4-baseline.json)",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help=(
            "Promote only this benchmark name. May be repeated. "
            "Useful for partial promotions."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff but do NOT write the baseline file",
    )
    parser.add_argument(
        "--default-threshold", type=float, default=None,
        help=(
            "Override default_threshold in the new file. "
            "Without this, the existing default is preserved."
        ),
    )
    args = parser.parse_args(argv)

    incoming = load_summaries(args.summaries_dir)
    if not incoming:
        print(f"No *_summary.json files found in {args.summaries_dir!r}; "
              f"nothing to publish.")
        return 1

    baseline_path = Path(args.baseline)
    existing = load_baseline(baseline_path)
    overrides = load_threshold_overrides(baseline_path)
    default_threshold = (
        args.default_threshold
        if args.default_threshold is not None
        else load_default_threshold(baseline_path)
    )

    new_summaries = merge_summaries(
        existing, incoming, only=args.only,
    )

    print(f"Baseline: {baseline_path}")
    print(f"Default threshold: {default_threshold:.2f}pt")
    if overrides:
        print("Per-benchmark overrides preserved:")
        for k, v in sorted(overrides.items()):
            print(f"  {k}: {v:.2f}pt")
    print("Diff:")
    print(_format_diff(existing, new_summaries))

    if args.dry_run:
        print("\n(dry-run; no file written)")
        return 0

    write_baseline(
        baseline_path, new_summaries,
        default_threshold=default_threshold,
        thresholds=overrides,
    )
    print(f"\nBaseline written ({len(new_summaries)} benchmarks).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
