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
) -> BenchmarkSummary:
    """End-to-end LME run. Returns the BenchmarkSummary.

    Side effects (when ``output_dir`` is set):
      - ``output_dir/longmemeval_report.json`` — raw LMERunReport
      - ``output_dir/longmemeval_summary.json`` — BenchmarkSummary

    ``sample_limit`` truncates the dataset; useful for smoke runs.
    """
    from attestor.longmemeval import (
        DEFAULT_MODEL, load_or_download, run_async,
    )

    samples = load_or_download(cache_dir=cache_dir, variant=variant)
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


# ── CLI ───────────────────────────────────────────────────────────────────


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
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    # The mem_factory builder is intentionally split out — operators wire
    # their own (Postgres URL, Neo4j URL, embedding model). Default uses
    # the SOLO config from the user's environment.
    from attestor.core import AgentMemory
    import tempfile

    def _factory() -> AgentMemory:
        return AgentMemory(tempfile.mkdtemp(prefix="lme_"))

    summary = run(
        mem_factory=_factory,
        variant=args.variant,
        cache_dir=args.cache_dir,
        budget=args.budget,
        parallel=args.parallel,
        output_dir=args.output_dir,
        sample_limit=args.limit,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
