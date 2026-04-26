"""BenchmarkSummary — standard cross-runner result shape (Phase 9.2).

Every eval runner (LongMemEval, BEAM, AbstentionBench, regression) emits
a ``BenchmarkSummary``. The CI gate compares summaries against the
baseline JSON; if any score regresses by more than the configured
threshold, the merge is blocked.

Why a separate type:
  - LMERunReport / BeamReport / etc. each have their own shape; the gate
    needs one comparable view across all of them.
  - The baseline file (``docs/bench/v4-baseline.json``) is just a list
    of these summaries — easy to read, easy to update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkSummary:
    """One row of the baseline file.

    Attributes:
        benchmark: Stable identifier — "longmemeval", "beam",
            "abstention", "regression". Used as the key in the baseline
            JSON.
        primary_metric: The single number the gate compares. For LME
            this is overall accuracy across judges. For regression this
            is pass_rate.
        primary_metric_name: Human-readable name for logs / reports.
        per_category: Optional per-category breakdown the gate may also
            inspect (e.g., LME's 6 question_type categories).
        total: Number of test units (samples / cases / questions).
        metadata: Free-form runner-specific extras (model, dataset
            variant, judge ids). Not used for comparison; preserved in
            the report for provenance.
    """
    benchmark: str
    primary_metric: float
    primary_metric_name: str = "accuracy"
    per_category: Dict[str, float] = field(default_factory=dict)
    total: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "primary_metric": self.primary_metric,
            "primary_metric_name": self.primary_metric_name,
            "per_category": dict(self.per_category),
            "total": self.total,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkSummary":
        return cls(
            benchmark=str(d["benchmark"]),
            primary_metric=float(d["primary_metric"]),
            primary_metric_name=str(d.get("primary_metric_name", "accuracy")),
            per_category={k: float(v) for k, v in (d.get("per_category") or {}).items()},
            total=int(d.get("total", 0)),
            metadata=dict(d.get("metadata") or {}),
        )
