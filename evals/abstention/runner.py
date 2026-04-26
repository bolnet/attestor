"""AbstentionBench runner (Phase 9.4.3).

Same shape as evals/beam/runner.py — pluggable loader / ingest / answer
protocols, per-sample error isolation, ``summarize()`` mapping the
run report to the standard BenchmarkSummary shape.

The primary metric exposed to the gate is F1 over the abstention
decision (positive class = "abstain"). Per-category surfacing in the
summary breaks down F1 by sample category (unknown_topic / false_premise /
underspecified / etc.) — that's the actionable signal when over- or
under-abstention regresses.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

from evals.abstention.detector import Detector
from evals.abstention.scorer import aggregate
from evals.abstention.types import (
    AbstentionPrediction, AbstentionRunReport, AbstentionSample,
)
from evals.summary import BenchmarkSummary

logger = logging.getLogger("attestor.evals.abstention")


# ── Protocols ─────────────────────────────────────────────────────────────


class DatasetLoader(Protocol):
    def __call__(self) -> List[AbstentionSample]: ...


class IngestFn(Protocol):
    def __call__(self, sample: AbstentionSample, mem: Any) -> None: ...


class AnswerFn(Protocol):
    """Produce the model's answer to a sample's query.

    The implementation MUST use the Chain-of-Note ABSTAIN-clause prompt
    (or an equivalent) — that's the entire point of this benchmark.
    """

    def __call__(self, sample: AbstentionSample, mem: Any) -> str: ...


# ── Default loader (fail loud) ───────────────────────────────────────────


class DefaultDatasetLoader:
    """Placeholder loader. Operators MUST wire their own."""

    def __call__(self) -> List[AbstentionSample]:
        raise NotImplementedError(
            "AbstentionBench dataset loader not configured. Pass a "
            "DatasetLoader to evals.abstention.runner.run() returning "
            "List[AbstentionSample]. The benchmark needs a balanced mix "
            "of answerable (with relevant context) and unanswerable "
            "(with empty/unrelated context) samples to be meaningful."
        )


# ── Run ───────────────────────────────────────────────────────────────────


def _run_one(
    sample: AbstentionSample,
    *,
    mem_factory: Callable[[], Any],
    ingest: IngestFn,
    answer: AnswerFn,
) -> AbstentionPrediction:
    mem = mem_factory()
    started = time.perf_counter()
    try:
        ingest(sample, mem)
        response = answer(sample, mem)
        elapsed = (time.perf_counter() - started) * 1000.0
        return AbstentionPrediction(
            sample_id=sample.sample_id,
            response=response,
            answerable=sample.answerable,
            category=sample.category,
            latency_ms=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("AbstentionBench sample %s failed", sample.sample_id)
        return AbstentionPrediction(
            sample_id=sample.sample_id,
            response="",
            answerable=sample.answerable,
            category=sample.category,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        close = getattr(mem, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning(
                    "mem.close() raised after sample %s; continuing",
                    sample.sample_id,
                )


def run(
    *,
    mem_factory: Callable[[], Any],
    ingest: IngestFn,
    answer: AnswerFn,
    loader: Optional[DatasetLoader] = None,
    detector: Optional[Detector] = None,
    sample_limit: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> BenchmarkSummary:
    """End-to-end AbstentionBench run.

    Returns the BenchmarkSummary so the gate can compare it.
    Side effects (when ``output_dir`` is set):
      - ``output_dir/abstention_report.json`` — raw AbstentionRunReport
      - ``output_dir/abstention_summary.json`` — BenchmarkSummary
    """
    loader = loader or DefaultDatasetLoader()
    samples = loader()
    if sample_limit is not None and sample_limit > 0:
        samples = samples[:sample_limit]
        logger.info(
            "AbstentionBench: truncated dataset to %d samples", len(samples),
        )

    predictions: List[AbstentionPrediction] = []
    for i, s in enumerate(samples, 1):
        predictions.append(_run_one(
            s, mem_factory=mem_factory, ingest=ingest, answer=answer,
        ))
        if i % 25 == 0 or i == len(samples):
            logger.info("AbstentionBench: %d/%d done", i, len(samples))

    report = aggregate(samples, predictions, detector=detector)
    summary = summarize(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "abstention_report.json").write_text(
            json.dumps(report.to_dict(), indent=2),
        )
        (out / "abstention_summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2),
        )
    return summary


# ── Summarize ─────────────────────────────────────────────────────────────


def summarize(report: AbstentionRunReport) -> BenchmarkSummary:
    """Map an AbstentionRunReport to the standard BenchmarkSummary.

    Primary metric: F1 over the abstention decision (0-100, scaled from
    the 0-1 ratio for consistency with the other runners).

    Per-category: F1 per sample category.
    """
    overall_f1_pct = float(report.overall.f1) * 100.0
    per_cat = {
        cat: float(m.f1) * 100.0
        for cat, m in report.by_category.items()
        if m.total > 0
    }
    return BenchmarkSummary(
        benchmark="abstention",
        primary_metric=overall_f1_pct,
        primary_metric_name="abstention_f1_pct",
        per_category=per_cat,
        total=report.overall.total,
        metadata={
            "precision": round(report.overall.precision, 4),
            "recall": round(report.overall.recall, 4),
            "over_abstention_rate": round(report.overall.over_abstention_rate, 4),
            "confabulation_rate": round(report.overall.confabulation_rate, 4),
            "answer_accuracy": round(report.answer_accuracy, 4),
        },
    )
