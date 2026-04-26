"""BEAM runner (Phase 9.3.3).

Glue that drives a long-context QA dataset through the memory stack:

    samples = loader()                    # operator-supplied
    for s in samples:
        ingest(s.context, into=mem)       # operator's IngestFn
        ans = answer(s.query, mem=mem)    # operator's AnswerFn
        record(BeamPrediction(...))
    report = aggregate(samples, predictions)
    summary = summarize(report)

Why so much pluggability:
  - "BEAM" in the v4 plan is a placeholder for whichever 1M-token
    benchmark the operator chooses (RULER, BABILong, ∞Bench, internal).
    Each has its own dataset shape, ingestion strategy, and answerer.
  - Locking the runner to one specific dataset would make this module
    obsolete the moment the operator picks a different one.
  - The interesting deterministic logic lives in scorer.py, which is
    fully unit-tested without any dataset.

Default behavior:
  - ``DefaultDatasetLoader`` raises NotImplementedError with a clear
    message so the failure mode is "operator hasn't wired the dataset",
    not "test silently passed against zero samples".
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

from evals.beam.scorer import aggregate
from evals.beam.types import (
    BeamPrediction, BeamRunReport, BeamSample,
    DEFAULT_BUCKETS, bucket_for,
)
from evals.summary import BenchmarkSummary

logger = logging.getLogger("attestor.evals.beam")


# ── Protocols ─────────────────────────────────────────────────────────────


class DatasetLoader(Protocol):
    """Loader contract: returns a list of BeamSamples ready to score."""

    def __call__(self) -> List[BeamSample]: ...


class IngestFn(Protocol):
    """Absorb a sample's context into the memory backend.

    Called once per sample. The implementation chooses the chunking
    strategy — typically ``mem.add(chunk, ...)`` for raw passages or
    ``mem.ingest_round(...)`` for dialogue.
    """

    def __call__(self, sample: BeamSample, mem: Any) -> None: ...


class AnswerFn(Protocol):
    """Produce the model's answer to a sample's query.

    The implementation usually does ``pack = mem.recall_as_pack(query)``
    then calls an LLM with the pack-rendered prompt. Returning the
    string only keeps the runner agnostic to the answerer.
    """

    def __call__(self, sample: BeamSample, mem: Any) -> str: ...


# ── Default loader (operator must override) ──────────────────────────────


class DefaultDatasetLoader:
    """Placeholder loader that fails loud if the operator forgot to wire one."""

    def __call__(self) -> List[BeamSample]:
        raise NotImplementedError(
            "BEAM dataset loader not configured. Pass a DatasetLoader to "
            "evals.beam.runner.run() that returns a list of BeamSample. "
            "See evals/beam/README.md for a worked example."
        )


# ── Run ───────────────────────────────────────────────────────────────────


def _run_one(
    sample: BeamSample,
    *,
    mem_factory: Callable[[], Any],
    ingest: IngestFn,
    answer: AnswerFn,
) -> BeamPrediction:
    """One sample → one prediction. Errors recorded, not raised."""
    mem = mem_factory()
    bucket = bucket_for(sample.token_count)
    started = time.perf_counter()
    try:
        ingest(sample, mem)
        predicted = answer(sample, mem)
        elapsed = (time.perf_counter() - started) * 1000.0
        return BeamPrediction(
            sample_id=sample.sample_id,
            predicted_answer=predicted,
            bucket=bucket,
            category=sample.category,
            latency_ms=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("BEAM sample %s failed", sample.sample_id)
        return BeamPrediction(
            sample_id=sample.sample_id,
            predicted_answer="",
            bucket=bucket,
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
    metric: str = "substring",
    sample_limit: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> BenchmarkSummary:
    """End-to-end BEAM run.

    The four mandatory wiring points (mem_factory, ingest, answer,
    loader) are intentionally explicit — this is not a benchmark you
    casually run, and silent defaults would hide bugs.

    Returns the BenchmarkSummary so the gate can compare it.
    Side effects (when ``output_dir`` is set):
      - ``output_dir/beam_report.json`` — raw BeamRunReport
      - ``output_dir/beam_summary.json`` — BenchmarkSummary
    """
    loader = loader or DefaultDatasetLoader()
    samples = loader()
    if sample_limit is not None and sample_limit > 0:
        samples = samples[:sample_limit]
        logger.info("BEAM: truncated dataset to %d samples", len(samples))

    predictions: List[BeamPrediction] = []
    for i, s in enumerate(samples, 1):
        predictions.append(_run_one(
            s, mem_factory=mem_factory, ingest=ingest, answer=answer,
        ))
        if i % 10 == 0 or i == len(samples):
            logger.info("BEAM: %d/%d done", i, len(samples))

    report = aggregate(samples, predictions, metric=metric)
    summary = summarize(report)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "beam_report.json").write_text(
            json.dumps(report.to_dict(), indent=2),
        )
        (out / "beam_summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2),
        )

    return summary


# ── Summarize: BeamRunReport → BenchmarkSummary ──────────────────────────


def summarize(report: BeamRunReport) -> BenchmarkSummary:
    """Map a BeamRunReport to the standard BenchmarkSummary shape.

    Per-category in the summary uses the BUCKET breakdown (not the
    sample category), because for a 1M benchmark the most actionable
    drift signal is "where on the context-length curve did we lose
    points?".
    """
    per_bucket = {
        b: float(stats.get("accuracy", 0.0))
        for b, stats in report.by_bucket.items()
        if stats.get("total", 0) > 0
    }
    return BenchmarkSummary(
        benchmark="beam",
        primary_metric=float(report.accuracy),
        primary_metric_name="accuracy_pct",
        per_category=per_bucket,
        total=report.total,
        metadata={
            "scoring": "substring",
            "buckets": [b for b, _, _ in DEFAULT_BUCKETS],
        },
    )
