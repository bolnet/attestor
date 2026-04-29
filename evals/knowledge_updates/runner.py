"""Synthetic supersession suite runner.

For each fixture:
  1. Ingest Session 1 turns (the older fact).
  2. Ingest Session 5 turns (the newer, contradicting fact).
  3. Call ``recall(question)``.
  4. Score: did the top-1 result reference the *new* fact, the *old*
     fact, or neither?

The metric is intentionally retrieval-only — we don't run an answerer.
A real answerer would synthesize over the full top-K, but the question
here is whether *retrieval* surfaces the newer fact above the older
one. That's the supersession-detection signal we want to measure.

Each fixture gets a fresh user_id (UUID per case) so the suite is
hermetic across DB state. Schema is NOT auto-applied — caller is
responsible (e.g. via ``scripts/lme_smoke_local.py``-style flow).

Outputs (under ``configs/bench.yaml``'s ``lme.output_dir``):
  - ``knowledge-updates-{date}.report.json`` — full per-case verdicts
  - ``knowledge-updates-{date}.summary.json`` — aggregate scores

Empty result (no top-1) counts as "miss" — it neither resolves nor
contradicts the supersession.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from attestor.bench_config import get_bench  # noqa: E402
from attestor.config import (  # noqa: E402
    StackConfig, build_backend_config, configure_embedder,
)

logger = logging.getLogger("attestor.evals.knowledge_updates")


# ──────────────────────────────────────────────────────────────────────
# Fixture loading + verdict dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseResult:
    """Per-case scoring output. Frozen so a partial run can't be edited
    after the fact when persisting."""

    case_id: str
    category: str
    question: str
    gold_answer: str
    stale_answer: str
    top1_content: Optional[str]
    top1_score: Optional[float]
    verdict: str   # "new_wins" | "stale_wins" | "miss" | "ambiguous"


@dataclass
class SuiteReport:
    """Aggregate report over the whole suite."""

    total: int = 0
    new_wins: int = 0
    stale_wins: int = 0
    miss: int = 0
    ambiguous: int = 0
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        return (self.new_wins / self.total * 100.0) if self.total else 0.0

    def add(self, result: CaseResult) -> None:
        self.cases.append(result)
        self.total += 1
        bucket = self.by_category.setdefault(
            result.category, {"new_wins": 0, "stale_wins": 0, "miss": 0, "ambiguous": 0},
        )
        if result.verdict == "new_wins":
            self.new_wins += 1
            bucket["new_wins"] += 1
        elif result.verdict == "stale_wins":
            self.stale_wins += 1
            bucket["stale_wins"] += 1
        elif result.verdict == "miss":
            self.miss += 1
            bucket["miss"] += 1
        else:
            self.ambiguous += 1
            bucket["ambiguous"] += 1


def load_fixtures(path: Path | str) -> List[dict]:
    """Read fixtures.json and return the case list. Strict on shape."""
    p = Path(path)
    raw = json.loads(p.read_text())
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(
            f"[knowledge_updates] {p} has no cases — expected non-empty 'cases' list"
        )
    return cases


# ──────────────────────────────────────────────────────────────────────
# Verdict logic — pure, easy to unit-test without DB
# ──────────────────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Lowercase + strip punctuation/whitespace. Keeps digits."""
    keep = []
    for ch in s.lower():
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
    return " ".join("".join(keep).split())


def classify_top1(
    top1_content: Optional[str],
    gold_answer: str,
    stale_answer: str,
) -> str:
    """Return one of: new_wins, stale_wins, miss, ambiguous.

    'miss' means top-1 mentions neither answer (retrieval found
    something else entirely). 'ambiguous' means top-1 mentions both
    (e.g. "Used to be Postgres, now SQLite") — these are correct on a
    full-context read but the supersession-signal is weak.
    """
    if not top1_content:
        return "miss"
    body = _normalize(top1_content)
    has_new = _normalize(gold_answer) in body
    has_stale = _normalize(stale_answer) in body
    if has_new and has_stale:
        return "ambiguous"
    if has_new:
        return "new_wins"
    if has_stale:
        return "stale_wins"
    return "miss"


# ──────────────────────────────────────────────────────────────────────
# Per-case ingest + recall
# ──────────────────────────────────────────────────────────────────────


def _ingest_session(mem: Any, user_id: str, session: dict) -> int:
    """Ingest every user turn from a session. Returns count of writes.

    Uses event_date from the session so the temporal layer can later
    reason about supersession. assistant turns are skipped — the
    contradiction is in the *user* statements.
    """
    n = 0
    sess_date = session.get("date")
    for turn in session.get("turns", []):
        if turn.get("role") != "user":
            continue
        mem.add(
            content=turn["content"],
            user_id=user_id,
            event_date=sess_date,
            metadata={
                "synthetic_session_id": session.get("session_id"),
                "synthetic_session_date": sess_date,
            },
        )
        n += 1
    return n


def run_one_case(
    mem_factory: Any, case: dict, *, pg_url: Optional[str] = None,
) -> CaseResult:
    """Hermetic single-case run.

    Each call gets a fresh AgentMemory and a freshly-provisioned user
    so prior cases can't leak into the recall surface. The v4 schema
    requires the user row to exist before any memory write — we
    bootstrap via ``UserRepo.create_or_get(external_id=...)`` and use
    the returned UUID for ``mem.add`` / ``mem.recall``.
    """
    import psycopg2

    from attestor.identity.users import UserRepo

    pg_url = pg_url or os.environ.get(
        "POSTGRES_URL",
        "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
    )
    external_id = f"ku-{uuid.uuid4().hex[:12]}"
    boot = psycopg2.connect(pg_url)
    boot.autocommit = True
    try:
        user = UserRepo(boot).create_or_get(
            external_id=external_id,
            display_name="knowledge_updates_eval",
            metadata={"source": "knowledge_updates_suite"},
        )
        user_id = user.id
    finally:
        boot.close()

    mem = mem_factory()
    sessions = sorted(case["sessions"], key=lambda s: s.get("date", ""))
    for sess in sessions:
        _ingest_session(mem, user_id, sess)

    results = mem.recall(query=case["question"], user_id=user_id)
    top1_content = results[0].content if results else None
    top1_score = results[0].score if results else None

    verdict = classify_top1(
        top1_content,
        gold_answer=case["gold_answer"],
        stale_answer=case["stale_answer"],
    )
    return CaseResult(
        case_id=case["case_id"],
        category=case["category"],
        question=case["question"],
        gold_answer=case["gold_answer"],
        stale_answer=case["stale_answer"],
        top1_content=top1_content,
        top1_score=top1_score,
        verdict=verdict,
    )


# ──────────────────────────────────────────────────────────────────────
# Suite driver
# ──────────────────────────────────────────────────────────────────────


def run_suite(
    *,
    fixtures_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    case_limit: Optional[int] = None,
) -> SuiteReport:
    """End-to-end driver.

    Reads fixtures, builds a stack-backed mem_factory via the canonical
    bench config, runs every case, and writes report+summary JSON.
    """
    cfg = get_bench()

    fixtures_path = Path(
        fixtures_path or (ROOT / cfg.knowledge_updates.fixtures_path)
    )
    output_dir = Path(output_dir or (ROOT / cfg.lme.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_fixtures(fixtures_path)
    if case_limit is not None and case_limit > 0:
        cases = cases[:case_limit]

    mem_factory = _build_mem_factory(cfg.stack)

    report = SuiteReport()
    for i, case in enumerate(cases, 1):
        try:
            result = run_one_case(mem_factory, case)
        except Exception as e:  # noqa: BLE001
            logger.error("case %s failed: %s", case.get("case_id"), e)
            result = CaseResult(
                case_id=case.get("case_id", "?"),
                category=case.get("category", "?"),
                question=case.get("question", ""),
                gold_answer=case.get("gold_answer", ""),
                stale_answer=case.get("stale_answer", ""),
                top1_content=None,
                top1_score=None,
                verdict="miss",
            )
        report.add(result)
        print(
            f"[knowledge_updates] {i:>2}/{len(cases)} {result.case_id:<24s} "
            f"[{result.category:<14s}] → {result.verdict}",
        )

    _persist_outputs(report, output_dir, cfg.knowledge_updates.target_score)
    return report


def _build_mem_factory(stack: StackConfig) -> Any:
    """Construct a zero-arg callable producing fresh AgentMemory.

    Same pattern as ``scripts/lme_smoke_local.py::_mem_factory`` — each
    case writes through a tempdir so prior cases can't influence
    retrieval.
    """
    import tempfile

    from attestor.core import AgentMemory

    configure_embedder(stack)
    backend_cfg = build_backend_config(stack)

    def _make() -> AgentMemory:
        path = tempfile.mkdtemp(prefix="ku_")
        return AgentMemory(path, config=backend_cfg)

    return _make


def _persist_outputs(report: SuiteReport, output_dir: Path, target_score: float) -> None:
    """Write report + summary JSON. Filename uses today's date for trend."""
    date = datetime.now().strftime("%Y%m%d")
    base = output_dir / f"knowledge-updates-{date}"

    full = {
        "total": report.total,
        "new_wins": report.new_wins,
        "stale_wins": report.stale_wins,
        "miss": report.miss,
        "ambiguous": report.ambiguous,
        "score_pct": report.score_pct,
        "target_score_pct": target_score * 100.0,
        "passed_target": report.score_pct >= target_score * 100.0,
        "by_category": report.by_category,
        "cases": [asdict(c) for c in report.cases],
    }

    summary = {
        "benchmark": "knowledge_updates",
        "primary_metric": report.score_pct,
        "primary_metric_name": "supersession_top1_pct",
        "total": report.total,
        "per_category": {
            cat: (
                vals["new_wins"] / max(
                    vals["new_wins"] + vals["stale_wins"]
                    + vals["miss"] + vals["ambiguous"], 1,
                ) * 100.0
            )
            for cat, vals in report.by_category.items()
        },
        "metadata": {
            "target_score_pct": target_score * 100.0,
            "passed_target": report.score_pct >= target_score * 100.0,
        },
    }

    base.with_suffix(".report.json").write_text(json.dumps(full, indent=2))
    base.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 72)
    print(
        f"[knowledge_updates] DONE — {report.new_wins}/{report.total} "
        f"new_wins ({report.score_pct:.1f}%); target = "
        f"{target_score*100:.0f}%; "
        f"{'PASS' if summary['metadata']['passed_target'] else 'FAIL'}",
    )
    print(f"[knowledge_updates] persisted → {base}.{{report,summary}}.json")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evals.knowledge_updates")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of cases (smoke runs)")
    p.add_argument("--fixtures",
                   help="Override fixtures.json path")
    p.add_argument("--output-dir",
                   help="Override output dir (default = bench.yaml lme.output_dir)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    run_suite(
        fixtures_path=args.fixtures,
        output_dir=args.output_dir,
        case_limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
