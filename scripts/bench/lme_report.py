"""Aggregate ``docs/bench/lme-*.summary.json`` into a markdown table.

Reads every ``lme-{variant}-{category}-{date}.summary.json`` in the
output dir from ``configs/bench.yaml``, picks the most recent file per
(variant, category) pair, and prints a markdown table suitable for
pasting into ``README.md`` or ``docs/index.html``.

Usage::

    .venv/bin/python scripts/bench/lme_report.py
    .venv/bin/python scripts/bench/lme_report.py --variant s
    .venv/bin/python scripts/bench/lme_report.py --markdown-out docs/bench/LME-S.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._bench_config import get_bench  # noqa: E402

# lme-{variant}-{category}-{YYYYMMDD}.summary.json
_FNAME_RE = re.compile(
    r"^lme-(?P<variant>[a-z]+)-(?P<category>[a-z0-9-]+)-(?P<date>\d{8})\.summary\.json$",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lme_report")
    p.add_argument(
        "--variant", default=None,
        help="Filter to one variant (s|m|oracle). Default: all.",
    )
    p.add_argument(
        "--markdown-out", default=None,
        help="Write the markdown table to this path in addition to stdout.",
    )
    p.add_argument(
        "--trend", action="store_true",
        help=(
            "Read docs/bench/trend.csv (path from bench.yaml's "
            "report.trend_csv) and emit a chronological per-(variant, "
            "category) progression table with deltas. Replaces the "
            "default summary view."
        ),
    )
    return p.parse_args()


def _scan_summaries(output_dir: Path) -> List[dict]:
    """Return the most-recent summary per (variant, category) tuple."""
    if not output_dir.exists():
        return []

    # group by (variant, category) → sorted-by-date list
    bucket: Dict[Tuple[str, str], List[Tuple[str, Path]]] = {}
    for f in output_dir.iterdir():
        if not f.is_file():
            continue
        m = _FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m.group("variant"), m.group("category"))
        bucket.setdefault(key, []).append((m.group("date"), f))

    rows: List[dict] = []
    for (variant, category), entries in bucket.items():
        entries.sort(key=lambda pair: pair[0], reverse=True)
        date, path = entries[0]
        try:
            payload = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"warning: skipping unreadable {path}: {e}\n")
            continue

        rows.append({
            "variant": variant,
            "category": category,
            "date": date,
            "score_pct": payload.get("primary_metric", 0.0),
            "total": payload.get("total", 0),
            "answer_model": payload.get("metadata", {}).get("answer_model", ""),
            "judges": payload.get("metadata", {}).get("judges", []),
            "path": str(path.relative_to(ROOT)),
        })
    return rows


def _format_markdown(rows: List[dict], variant_filter: Optional[str]) -> str:
    if variant_filter:
        rows = [r for r in rows if r["variant"] == variant_filter]
    if not rows:
        return (
            "_No bench summaries found. Run `scripts/bench/lme_run.sh` "
            "or `scripts/bench/lme_all.sh` to populate `docs/bench/`._\n"
        )

    rows.sort(key=lambda r: (r["variant"], r["category"]))

    lines = [
        "| Variant | Category | Score | N | Date | Answer | Judges |",
        "| ------- | -------- | -----:| -:| ---- | ------ | ------ |",
    ]
    for r in rows:
        judges = ", ".join(r["judges"]) if isinstance(r["judges"], list) else str(r["judges"])
        lines.append(
            f"| {r['variant']} | {r['category']} | "
            f"{r['score_pct']:.1f}% | {r['total']} | {r['date']} | "
            f"{r['answer_model']} | {judges} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args()
    cfg = get_bench()
    output_dir = ROOT / cfg.lme.output_dir

    if args.trend:
        from scripts.bench.trend import format_trend_markdown, read_trend
        trend_path = ROOT / cfg.report.trend_csv
        rows = read_trend(trend_path)
        if args.variant:
            rows = [r for r in rows if r.get("variant") == args.variant]
        md = format_trend_markdown(rows)
    else:
        rows = _scan_summaries(output_dir)
        md = _format_markdown(rows, variant_filter=args.variant)

    print(md)

    if args.markdown_out:
        out_path = Path(args.markdown_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"# wrote → {out_path.relative_to(ROOT)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
