"""AbstentionBench data types (Phase 9.4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class AbstentionSample:
    """One abstention test case.

    Attributes:
        sample_id: Unique id used to key reports.
        context: Material the memory layer ingests before the query.
            For pure abstention tests this is often empty or contains
            unrelated facts; for "answerable" cases it carries the
            relevant fact.
        query: The question to answer.
        answerable: Gold label.
            True  → memory IS sufficient; the model should answer.
            False → no relevant info; the model should abstain.
        expected_answer: When ``answerable=True``, the substring that
            must appear in the model's response. Ignored when
            ``answerable=False``.
        category: Optional dataset-defined slice (e.g.,
            "unknown_topic", "false_premise", "subjective",
            "underspecified"). Surfaced in per-category metrics.
        metadata: Free-form extras preserved into the report.
    """
    sample_id: str
    context: str
    query: str
    answerable: bool
    expected_answer: Optional[str] = None
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbstentionPrediction:
    """The system's response to one AbstentionSample."""
    sample_id: str
    response: str                    # raw model output
    abstained: Optional[bool] = None # filled by detector
    correct: Optional[bool] = None   # filled by scorer
    category: str = "general"
    answerable: bool = True          # mirrored from sample for slicing
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class AbstentionMetrics:
    """Confusion-matrix view over the abstention decision.

    Treating "abstain" as the positive class:
        true_positive  — should abstain, did
        false_positive — should answer, but abstained (over-abstention)
        true_negative  — should answer, did (correctly)
        false_negative — should abstain, but answered (confabulation)
    """
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return (self.true_positive + self.false_positive
                + self.true_negative + self.false_negative)

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def over_abstention_rate(self) -> float:
        """Of cases where the model SHOULD have answered, how often
        did it abstain instead?"""
        denom = self.true_negative + self.false_positive
        return self.false_positive / denom if denom else 0.0

    @property
    def confabulation_rate(self) -> float:
        """Of cases where the model SHOULD have abstained, how often
        did it answer (likely confabulating) instead?"""
        denom = self.true_positive + self.false_negative
        return self.false_negative / denom if denom else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "total": self.total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "over_abstention_rate": round(self.over_abstention_rate, 4),
            "confabulation_rate": round(self.confabulation_rate, 4),
        }


@dataclass(frozen=True)
class AbstentionRunReport:
    """Aggregated AbstentionBench output. JSON-serializable via to_dict()."""
    overall: AbstentionMetrics
    by_category: Dict[str, AbstentionMetrics] = field(default_factory=dict)
    answer_accuracy: float = 0.0  # of answerable cases, what % were also right?
    answer_total: int = 0
    answer_correct: int = 0
    predictions: Tuple[AbstentionPrediction, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "answer_accuracy": round(self.answer_accuracy, 4),
            "answer_correct": self.answer_correct,
            "answer_total": self.answer_total,
            "predictions": [
                {
                    "sample_id": p.sample_id,
                    "response": p.response,
                    "abstained": p.abstained,
                    "correct": p.correct,
                    "category": p.category,
                    "answerable": p.answerable,
                    "latency_ms": p.latency_ms,
                    "error": p.error,
                }
                for p in self.predictions
            ],
        }
