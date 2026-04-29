"""Trend.csv accumulator + reader for LME bench runs.

One row appended per bench summary persisted by ``scripts/lme_smoke_local.py``.
The CSV lives at the path configured in ``configs/bench.yaml``'s
``report.trend_csv`` (default ``docs/bench/trend.csv``) and grows monotonically
across runs so the team can see the score curve over time.

Schema (CSV columns):
    timestamp        ISO 8601 UTC, sub-second resolution
    date             YYYYMMDD, for fast date filtering
    git_sha          short hash of HEAD at run time (or ``unknown``)
    variant          s | m | oracle
    category         single slice name or ``all``
    n                number of samples scored
    score_pct        primary metric (0-100 float)
    answer_model     models.answerer at run time
    judges           semicolon-separated judge models
    features         comma-separated feature flags that were enabled
                     (multi_query, hyde, temporal_prefilter,
                     self_consistency, critique_revise) — empty when
                     all flags were default (off)
    run_label        free-text marker for ad-hoc runs

The schema is intentionally append-only and forward-compatible: new
columns can be added at the right edge without breaking existing
readers (the report uses ``DictReader`` so missing columns just yield
None).
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


TREND_HEADERS = (
    "timestamp",
    "date",
    "git_sha",
    "variant",
    "category",
    "n",
    "score_pct",
    "answer_model",
    "judges",
    "features",
    "run_label",
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _git_sha_short(repo_root: Optional[Path] = None) -> str:
    """Return the short HEAD hash, or ``unknown`` when git is unavailable
    or the repo isn't initialized. Defensive — bench tracking must not
    crash because of git.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace").strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _features_from_stack(stack: Any) -> List[str]:
    """Inspect a StackConfig and list the retrieval/answerer features
    that are currently enabled. Used to record what was on for a given
    bench run.
    """
    enabled: List[str] = []
    retr = getattr(stack, "retrieval", None)
    if retr is not None:
        for name in ("multi_query", "hyde", "temporal_prefilter"):
            cfg = getattr(retr, name, None)
            if cfg is not None and getattr(cfg, "enabled", False):
                enabled.append(name)
    for name in ("self_consistency", "critique_revise"):
        cfg = getattr(stack, name, None)
        if cfg is not None and getattr(cfg, "enabled", False):
            enabled.append(name)
    return enabled


# ──────────────────────────────────────────────────────────────────────
# Append a row
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrendRow:
    """One bench run, one row. All fields are stringly-typed at write
    time because CSV is stringly-typed at read time anyway — keeps the
    appender simple and the schema robust to extension."""

    timestamp: str
    date: str
    git_sha: str
    variant: str
    category: str
    n: int
    score_pct: float
    answer_model: str
    judges: str
    features: str
    run_label: str

    def as_dict(self) -> Dict[str, Any]:
        return {h: getattr(self, h) for h in TREND_HEADERS}


def append_trend_row(
    csv_path: Path | str,
    *,
    summary: Dict[str, Any],
    variant: str,
    category: Optional[str],
    features: List[str],
    git_sha: Optional[str] = None,
    run_label: str = "",
) -> TrendRow:
    """Append one row to ``csv_path``. Creates the file with headers
    if missing, writes only the data row otherwise. Returns the row
    that was appended for logging.

    ``summary`` is the BenchmarkSummary dict (``primary_metric``,
    ``total``, ``metadata.{answer_model, judges}``).
    """
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    metadata = summary.get("metadata") or {}
    judges = metadata.get("judges") or []
    if isinstance(judges, list):
        judges_str = ";".join(str(j) for j in judges)
    else:
        judges_str = str(judges)

    row = TrendRow(
        timestamp=now.isoformat(),
        date=now.strftime("%Y%m%d"),
        git_sha=(git_sha if git_sha is not None else _git_sha_short()),
        variant=str(variant),
        category=str(category or "all"),
        n=int(summary.get("total", 0)),
        score_pct=float(summary.get("primary_metric", 0.0)),
        answer_model=str(metadata.get("answer_model", "")),
        judges=judges_str,
        features=",".join(features),
        run_label=str(run_label),
    )

    write_headers = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TREND_HEADERS))
        if write_headers:
            writer.writeheader()
        writer.writerow(row.as_dict())

    return row


# ──────────────────────────────────────────────────────────────────────
# Reader + markdown formatter
# ──────────────────────────────────────────────────────────────────────


def read_trend(csv_path: Path | str) -> List[Dict[str, Any]]:
    """Read every row from ``csv_path``. Empty / missing → empty list."""
    p = Path(csv_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def _coerce(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numeric fields to numeric types; leave others as strings."""
    out = dict(row)
    try:
        out["score_pct"] = float(row.get("score_pct", 0.0))
    except (TypeError, ValueError):
        out["score_pct"] = 0.0
    try:
        out["n"] = int(row.get("n", 0))
    except (TypeError, ValueError):
        out["n"] = 0
    return out


def format_trend_markdown(rows: List[Dict[str, Any]]) -> str:
    """Render the trend as a markdown table grouped by (variant, category).
    Within each group, rows are sorted by timestamp ascending and a
    ``Δ`` column shows the score delta from the previous run in the
    same group.
    """
    if not rows:
        return (
            "_No trend rows yet. Bench runs append to "
            "`docs/bench/trend.csv`; come back after the first run._\n"
        )

    rows = [_coerce(r) for r in rows]
    rows.sort(key=lambda r: (
        r.get("variant", ""), r.get("category", ""), r.get("timestamp", ""),
    ))

    lines = [
        "| Variant | Category | Date | N | Score | Δ | SHA | Features | Run |",
        "| ------- | -------- | ---- | -:| -----:| -:| --- | -------- | --- |",
    ]
    prev_score: Dict[tuple, float] = {}
    for r in rows:
        key = (r.get("variant", ""), r.get("category", ""))
        prev = prev_score.get(key)
        delta = "" if prev is None else f"{r['score_pct'] - prev:+.1f}"
        lines.append(
            f"| {r.get('variant','')} | {r.get('category','')} | "
            f"{r.get('date','')} | {r['n']} | {r['score_pct']:.1f}% | "
            f"{delta} | {r.get('git_sha','')} | {r.get('features','')} | "
            f"{r.get('run_label','')} |"
        )
        prev_score[key] = r["score_pct"]
    return "\n".join(lines) + "\n"
