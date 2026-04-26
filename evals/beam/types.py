"""BEAM data types (Phase 9.3.1).

Frozen dataclasses describing the shape of one long-context QA sample,
the prediction the system produces, and the aggregate report.

Design choice: keep these dataset-agnostic. ``BeamSample.context`` is
the haystack the memory layer must absorb (typically via ``ingest_round``
or bulk ``add``); ``query`` is the question; ``answer`` is the gold
string. The dataset loader is responsible for chunking the context into
ingestible turns — the types don't dictate that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Context-length buckets ────────────────────────────────────────────────
# Standard bucket boundaries used to slice scores. Operators may override
# via the runner config; these are the reasonable defaults for a 1M
# benchmark.
DEFAULT_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("1k",   0,           2_000),
    ("8k",   2_000,      16_000),
    ("32k",  16_000,     64_000),
    ("128k", 64_000,    256_000),
    ("512k", 256_000,   768_000),
    ("1M",   768_000, 2_000_000),
)


def bucket_for(token_count: int,
               buckets: Tuple[Tuple[str, int, int], ...] = DEFAULT_BUCKETS) -> str:
    """Map a token count to its bucket label. Returns 'overflow' if no fit."""
    for label, lo, hi in buckets:
        if lo <= token_count < hi:
            return label
    return "overflow"


@dataclass(frozen=True)
class BeamSample:
    """One long-context QA sample.

    Attributes:
        sample_id: Unique id used to key reports.
        context: The haystack — passages, dialogue, documents. May be
            very large (1M tokens). The runner / mem_factory is
            responsible for ingesting this efficiently (chunked add()
            beats one giant ingest_round).
        query: The question to ask after ingestion.
        answer: The gold answer string.
        token_count: Approximate token length of ``context``. Used to
            bucket scores by haystack size.
        category: Optional dataset-defined category (e.g., "needle",
            "multi_hop", "summary"). Mirrored into per_category in the
            BenchmarkSummary.
        metadata: Free-form extras preserved through to the report.
    """
    sample_id: str
    context: str
    query: str
    answer: str
    token_count: int
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeamPrediction:
    """The system's response to one BeamSample."""
    sample_id: str
    predicted_answer: str
    bucket: str = ""
    category: str = "general"
    correct: Optional[bool] = None  # filled by scorer
    latency_ms: float = 0.0
    error: Optional[str] = None     # set when ingest/recall raised


@dataclass(frozen=True)
class BeamRunReport:
    """Aggregated BEAM run output. JSON-serializable via to_dict()."""
    total: int
    correct: int
    accuracy: float                                   # 0-100, primary metric
    by_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictions: Tuple[BeamPrediction, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "by_bucket": {k: dict(v) for k, v in self.by_bucket.items()},
            "by_category": {k: dict(v) for k, v in self.by_category.items()},
            "predictions": [
                {
                    "sample_id": p.sample_id,
                    "predicted_answer": p.predicted_answer,
                    "bucket": p.bucket,
                    "category": p.category,
                    "correct": p.correct,
                    "latency_ms": p.latency_ms,
                    "error": p.error,
                }
                for p in self.predictions
            ],
        }
