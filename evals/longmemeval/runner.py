"""LongMemEval runner — thin wrapper over attestor.longmemeval (Phase 9.2.1).

Two responsibilities:

  1. Provide a single ``run(...)`` entry point that loads samples,
     constructs a ``mem_factory``, calls ``attestor.longmemeval.run_async``,
     and writes both the raw ``LMERunReport`` and a normalized
     ``BenchmarkSummary`` to disk.

  2. ``summarize(report)`` extracts the BenchmarkSummary from any
     ``LMERunReport`` — pure function, no I/O — so unit tests can verify
     the mapping without running the actual benchmark.

The full pipeline needs OPENROUTER_API_KEY (answerer + judges) and
either Postgres+Ollama (real recall) or a stub mem_factory. For CI
without API keys, only ``summarize`` is exercised.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from evals.summary import BenchmarkSummary

logger = logging.getLogger("attestor.evals.longmemeval")


# ── summarize: LMERunReport → BenchmarkSummary ────────────────────────────


def _overall_accuracy(by_judge: dict) -> float:
    """Average accuracy across judges. Returns 0.0 on empty input.

    LMERunReport.by_judge: ``{judge_model: {correct, total, accuracy}}``
    where ``accuracy`` is already a percentage (0-100).
    """
    accs = [b.get("accuracy", 0.0) for b in by_judge.values() if b.get("total", 0) > 0]
    return float(sum(accs) / len(accs)) if accs else 0.0


def _category_accuracies(by_category: dict) -> dict:
    """Average per-category accuracy across judges.

    by_category shape: ``{cat: {judge: {correct, total, accuracy}}}``
    Returns ``{cat: avg_accuracy}``. Categories with zero samples for
    every judge are omitted (no signal).
    """
    out = {}
    for cat, per_judge in by_category.items():
        accs = [b.get("accuracy", 0.0) for b in per_judge.values()
                if b.get("total", 0) > 0]
        if accs:
            out[cat] = float(sum(accs) / len(accs))
    return out


def summarize(report: Any) -> BenchmarkSummary:
    """Map an ``LMERunReport`` (or its ``asdict`` form) to a BenchmarkSummary.

    Accepts both the dataclass and a dict because the report often
    arrives via JSON (read from disk in CI) rather than from memory.
    """
    if hasattr(report, "by_judge"):
        by_judge = report.by_judge
        by_category = report.by_category
        total = report.total
        answer_model = report.answer_model
        judges = list(report.judge_models)
    else:
        by_judge = report.get("by_judge", {})
        by_category = report.get("by_category", {})
        total = int(report.get("total", 0))
        answer_model = report.get("answer_model", "")
        judges = list(report.get("judge_models", []))

    return BenchmarkSummary(
        benchmark="longmemeval",
        primary_metric=_overall_accuracy(by_judge),
        primary_metric_name="accuracy_pct",
        per_category=_category_accuracies(by_category),
        total=total,
        metadata={
            "answer_model": answer_model,
            "judges": judges,
        },
    )


# ── run: full pipeline ────────────────────────────────────────────────────


VALID_CATEGORIES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

# Round-robin order for submission runs — matches user directive 2026-05-05.
# KU first (smallest, fastest signal), then SSU, then the two larger groups.
SUBMISSION_CATEGORIES: tuple[str, ...] = (
    "knowledge-update",
    "single-session-user",
    "temporal-reasoning",
    "multi-session",
)


def run(
    *,
    mem_factory: Callable[[], Any],
    variant: str = "s",
    cache_dir: Optional[Path | str] = None,
    answer_model: Optional[str] = None,
    judge_models: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    budget: int = 4000,
    parallel: int = 4,
    output_dir: Optional[Path | str] = None,
    sample_limit: Optional[int] = None,
    category: Optional[str] = None,
) -> BenchmarkSummary:
    """End-to-end LME run. Returns the BenchmarkSummary.

    Side effects (when ``output_dir`` is set):
      - ``output_dir/longmemeval_report.json`` — raw LMERunReport
      - ``output_dir/longmemeval_summary.json`` — BenchmarkSummary

    ``sample_limit`` truncates the dataset; useful for smoke runs.
    ``category`` filters samples by ``question_type`` — pass one of
    :data:`VALID_CATEGORIES` to score a single slice.
    """
    from attestor.longmemeval import (
        DEFAULT_MODEL, load_or_download, run_async,
    )

    samples = load_or_download(cache_dir=cache_dir, variant=variant)

    if category is not None:
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"unknown category {category!r}. "
                f"Choose one of: {sorted(VALID_CATEGORIES)}"
            )
        before = len(samples)
        samples = [s for s in samples if s.question_type == category]
        logger.info(
            "longmemeval: filtered to category=%s (%d → %d samples)",
            category, before, len(samples),
        )
        if not samples:
            raise SystemExit(
                f"longmemeval: no samples for category={category!r} in "
                f"variant={variant!r} — empty result, aborting.",
            )

    if sample_limit is not None and sample_limit > 0:
        samples = samples[:sample_limit]
        logger.info("longmemeval: truncated dataset to %d samples", len(samples))

    report = asyncio.run(run_async(
        samples,
        mem_factory=mem_factory,
        answer_model=answer_model or DEFAULT_MODEL,
        judge_models=judge_models,
        api_key=api_key,
        budget=budget,
        parallel=parallel,
    ))
    summary = summarize(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        (out / "longmemeval_report.json").write_text(
            json.dumps(asdict(report), default=str, indent=2),
        )
        (out / "longmemeval_summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2),
        )

    return summary


# ── Submission mode (round-robin, sequential, resumable) ─────────────────


def _round_robin(
    by_category: dict[str, list[Any]],
    categories: tuple[str, ...] = SUBMISSION_CATEGORIES,
) -> list[Any]:
    """Interleave samples across categories.

    Round 0 emits the first sample from each category that still has one,
    in ``categories`` order; round 1 emits the second; etc. Smaller
    categories drop out as their list runs dry. Deterministic — no shuffle.
    """
    schedule: list[Any] = []
    rounds = max((len(by_category.get(c, [])) for c in categories), default=0)
    for r in range(rounds):
        for cat in categories:
            bucket = by_category.get(cat, [])
            if r < len(bucket):
                schedule.append(bucket[r])
    return schedule


def _load_done(submission_dir: Path) -> dict[str, set[str]]:
    """Scan per-category jsonls and build {category: {question_ids_done}}.

    Source of truth is the jsonl files (one ``{"question_id", "hypothesis"}``
    per line). PROGRESS.md is a human-readable mirror, not authoritative —
    if the two disagree, the jsonl wins.
    """
    done: dict[str, set[str]] = {c: set() for c in SUBMISSION_CATEGORIES}
    for cat in SUBMISSION_CATEGORIES:
        path = submission_dir / f"lme_s_{cat}.jsonl"
        if not path.exists():
            continue
        with path.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s: %r", path, line[:60])
                    continue
                qid = obj.get("question_id")
                if qid:
                    done[cat].add(qid)
    return done


def _append_jsonl(path: Path, payload: dict) -> None:
    """Append one JSON object as a line, fsync to survive crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)
        fp.flush()
        import os
        os.fsync(fp.fileno())


def _write_progress_md(
    submission_dir: Path,
    done: dict[str, set[str]],
    targets: dict[str, int],
    started_at: str,
) -> None:
    """Render PROGRESS.md from the in-memory ``done`` map.

    Whole file is rewritten on each update — cheap (4 categories) and
    avoids partial-line corruption under crash.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# LME-S submission progress",
        "",
        f"- started: `{started_at}`",
        f"- updated: `{now}`",
        f"- output dir: `{submission_dir.resolve()}`",
        "",
        "| category | done | target | jsonl |",
        "| --- | ---: | ---: | --- |",
    ]
    for cat in SUBMISSION_CATEGORIES:
        if cat not in done:
            continue  # category was filtered out via --submission-categories
        n_done = len(done[cat])
        target = targets.get(cat, 0)
        lines.append(
            f"| {cat} | {n_done} | {target} | "
            f"`lme_s_{cat}.jsonl` |"
        )
    lines.append("")
    lines.append("## Resume")
    lines.append("")
    lines.append(
        "Re-run the same command — already-done `question_id`s are skipped "
        "(source of truth = the per-category jsonl files)."
    )
    (submission_dir / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def submission_run(
    *,
    mem_factory: Callable[[], Any],
    submission_dir: Path | str,
    variant: str = "s",
    cache_dir: Optional[Path | str] = None,
    answer_model: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: int = 4000,
    sample_limit: Optional[int] = None,
    fresh: bool = False,
    parallel: Optional[int] = None,
    categories: Optional[tuple[str, ...]] = None,
) -> dict[str, int]:
    """Round-robin, resumable LME-S submission run.

    - One question from each of 4 categories (KU → SSU → temporal → MS),
      then round-2, etc. Deterministic dataset order, no shuffle.
    - ``parallel`` defaults to ``stack.parallel`` from the YAML config
      (currently 2). Pass an explicit value to override.
    - skip_judge=True — emits ``{question_id, hypothesis}`` only.
    - Resume by reading existing per-category jsonls in ``submission_dir``.
    - PROGRESS.md updated after each completed sample (fsync per line).

    Returns: ``{category: count_completed_after_run}``.
    """
    from attestor.longmemeval import (
        DEFAULT_MODEL, load_or_download, run_async,
    )
    from datetime import datetime, timezone

    active_categories: tuple[str, ...] = (
        tuple(categories) if categories else SUBMISSION_CATEGORIES
    )
    for c in active_categories:
        if c not in SUBMISSION_CATEGORIES:
            raise ValueError(
                f"unknown category {c!r}; valid: {SUBMISSION_CATEGORIES}"
            )

    sub_dir = Path(submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        for cat in active_categories:
            p = sub_dir / f"lme_s_{cat}.jsonl"
            if p.exists():
                p.unlink()
        prog = sub_dir / "PROGRESS.md"
        if prog.exists():
            prog.unlink()

    samples = load_or_download(cache_dir=cache_dir, variant=variant)

    by_category: dict[str, list[Any]] = {c: [] for c in active_categories}
    for s in samples:
        if s.question_type in by_category:
            by_category[s.question_type].append(s)

    if sample_limit is not None and sample_limit > 0:
        for cat in by_category:
            by_category[cat] = by_category[cat][:sample_limit]

    targets = {cat: len(by_category[cat]) for cat in active_categories}
    full_done = _load_done(sub_dir)
    done = {cat: full_done.get(cat, set()) for cat in active_categories}

    schedule = _round_robin(by_category, categories=active_categories)
    pending = [
        s for s in schedule
        if s.question_id not in done[s.question_type]
    ]

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_progress_md(sub_dir, done, targets, started_at)

    logger.info(
        "submission_run: %d pending of %d scheduled (already done: %d) — "
        "writing to %s",
        len(pending), len(schedule),
        sum(len(d) for d in done.values()),
        sub_dir,
    )

    if not pending:
        logger.info("submission_run: nothing to do — all samples already in jsonls")
        return {cat: len(done[cat]) for cat in active_categories}

    def _on_complete(_done_n: int, _total: int, report: Any) -> None:
        cat = report.category
        path = sub_dir / f"lme_s_{cat}.jsonl"
        _append_jsonl(path, {
            "question_id": report.question_id,
            "hypothesis": report.answer,
        })
        done[cat].add(report.question_id)
        _write_progress_md(sub_dir, done, targets, started_at)

    if parallel is None:
        from attestor.config import get_stack
        parallel = max(1, int(getattr(get_stack(), "parallel", 1) or 1))

    asyncio.run(run_async(
        pending,
        mem_factory=mem_factory,
        answer_model=answer_model or DEFAULT_MODEL,
        judge_models=[],
        api_key=api_key,
        budget=budget,
        parallel=parallel,
        skip_judge=True,
        progress_callback=_on_complete,
        verbose=True,
    ))

    return {cat: len(done[cat]) for cat in active_categories}


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_mem_factory(*, warm_schema: bool = True) -> Callable[[], Any]:
    """Build a zero-arg factory that returns ``AgentMemory`` instances
    wired with the canonical YAML stack (PG + Pinecone + Neo4j).

    Pre-fix: the factory called ``AgentMemory(tempfile.mkdtemp(...))``
    with no config kwarg, so AgentMemory fell through to a passwordless
    Postgres default and the smoke run died with
    ``fe_sendauth: no password supplied``. The bug was a silent skip of
    ``configs/attestor.yaml`` for the ``python -m evals.longmemeval``
    entry point. ``evals/braintrust_longmemeval.py`` was already correct.

    ``warm_schema=True`` runs one schema-init pass with a throw-away
    AgentMemory before parallel workers spin up, so they don't race on
    CREATE TABLE catalog locks. Tests pass ``warm_schema=False`` to
    avoid the live DB call.
    """
    import tempfile
    from attestor.config import build_backend_config, get_stack
    from attestor.core import AgentMemory

    backend_config = build_backend_config(get_stack())
    if warm_schema:
        backend_config["backend_configs"]["postgres"]["skip_schema_init"] = False
        AgentMemory(tempfile.mkdtemp(prefix="lme_warmup_"), config=backend_config)
    backend_config["backend_configs"]["postgres"]["skip_schema_init"] = True

    def _factory() -> AgentMemory:
        return AgentMemory(
            tempfile.mkdtemp(prefix="lme_"), config=backend_config,
        )

    return _factory


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m evals.longmemeval` entry point.

    Reads a YAML config (path passed via ``--config``) describing the
    mem_factory and run knobs. The shipped example is in
    ``evals/longmemeval/example_config.yaml``. Returns 0 on success.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="evals.longmemeval")
    parser.add_argument("--variant", default="s", help="LME dataset variant (s|m|oracle)")
    parser.add_argument("--limit", type=int, default=None, help="cap samples")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="override stack.parallel from YAML; default = read from config",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--category", default=None,
        help=(
            "filter to one slice — "
            f"{'|'.join(VALID_CATEGORIES)}"
        ),
    )
    parser.add_argument(
        "--submission-dir", default=None,
        help=(
            "ROUND-ROBIN SUBMISSION MODE. When set, runs sequentially across "
            "the 4 LME-S categories (KU → SSU → temporal → MS), forces "
            "parallel=1, skips the judge, and writes per-category jsonl "
            "files with {question_id, hypothesis} plus a human-readable "
            "PROGRESS.md. Resumes from existing jsonl files automatically. "
            "--limit caps PER CATEGORY in this mode."
        ),
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="submission mode: wipe existing jsonls + PROGRESS.md before starting",
    )
    parser.add_argument(
        "--submission-categories", nargs="+", default=None,
        choices=list(SUBMISSION_CATEGORIES),
        help=(
            "submission mode: subset of categories to run "
            "(default: all 4). Order is preserved from SUBMISSION_CATEGORIES."
        ),
    )
    args = parser.parse_args(argv)

    _factory = _build_mem_factory()

    if args.submission_dir is not None:
        counts = submission_run(
            mem_factory=_factory,
            submission_dir=args.submission_dir,
            variant=args.variant,
            cache_dir=args.cache_dir,
            budget=args.budget,
            categories=tuple(args.submission_categories) if args.submission_categories else None,
            sample_limit=args.limit,
            fresh=args.fresh,
            parallel=args.parallel,
        )
        print(json.dumps({"submission_counts": counts}, indent=2))
        return 0

    if args.parallel is None:
        from attestor.config import get_stack
        resolved_parallel = max(1, int(getattr(get_stack(), "parallel", 4) or 4))
    else:
        resolved_parallel = args.parallel

    summary = run(
        mem_factory=_factory,
        variant=args.variant,
        cache_dir=args.cache_dir,
        budget=args.budget,
        parallel=resolved_parallel,
        output_dir=args.output_dir,
        sample_limit=args.limit,
        category=args.category,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
