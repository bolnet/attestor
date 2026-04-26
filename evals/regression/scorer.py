"""Regression case scorer (Phase 9.1.2).

Pure function: given a RegressionCase and the ContextPack the system
recalled, produce a CaseResult that is True iff:

  • every must_contain substring appears in some pack entry's content
  • no must_not_contain substring appears in any pack entry's content
  • abstain_required → pack has zero memories
  • abstain_ok      → empty pack also passes (over and above the
                      must_contain check, which would normally fail)

Comparisons are case-insensitive on stripped content. The scorer never
hits an LLM — that's a deliberate design choice for a CI gate. If you
want LLM-graded scoring, layer it on top using the same pack input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from evals.regression.cases import RegressionCase


@dataclass(frozen=True)
class CaseResult:
    """Outcome of scoring one case against a recalled pack."""
    case_id: str
    passed: bool
    reasons: tuple[str, ...] = ()      # one entry per failed check
    matched: tuple[str, ...] = ()      # must_contain substrings that matched
    pack_size: int = 0
    abstained: bool = False            # pack was empty (used or required)


# ── Helpers ───────────────────────────────────────────────────────────────


def _texts(pack: object) -> List[str]:
    """Extract lowercased content strings from a ContextPack-like object.

    Accepts either an attestor.models.ContextPack instance or any object
    with a ``memories`` attribute that yields entries with ``content``.
    Duck-typed so the scorer is independent of the data layer.
    """
    memories: Iterable = getattr(pack, "memories", None) or []
    out: List[str] = []
    for m in memories:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if content:
            out.append(str(content).strip().lower())
    return out


def _any_contains(texts: Sequence[str], needle: str) -> bool:
    needle = needle.strip().lower()
    return any(needle in t for t in texts) if needle else False


# ── Public API ────────────────────────────────────────────────────────────


def score_case(case: RegressionCase, pack: object) -> CaseResult:
    """Score one case against the recalled pack.

    The pack is duck-typed (anything with a ``memories`` iterable of
    objects with ``.content``). Returns a CaseResult; never raises on
    failure — it records the reason instead so the runner can build a
    full report.
    """
    texts = _texts(pack)
    pack_size = len(texts)
    abstained = pack_size == 0
    reasons: List[str] = []
    matched: List[str] = []

    # Abstention rules first — they short-circuit must_contain logic
    if case.abstain_required:
        if not abstained:
            preview = ", ".join(t[:40] for t in texts[:3])
            reasons.append(
                f"expected abstention (empty pack), got {pack_size} memories: "
                f"[{preview}]"
            )
        return CaseResult(
            case_id=case.id,
            passed=not reasons,
            reasons=tuple(reasons),
            pack_size=pack_size,
            abstained=abstained,
        )

    if abstained and case.abstain_ok:
        # Acceptable empty result — skip must_contain check, return pass
        return CaseResult(
            case_id=case.id,
            passed=True,
            pack_size=0,
            abstained=True,
        )

    # must_contain — every needle must appear in at least one entry
    for needle in case.must_contain:
        if _any_contains(texts, needle):
            matched.append(needle)
        else:
            preview = " | ".join(t[:60] for t in texts[:5]) or "(empty)"
            reasons.append(
                f"missing must_contain {needle!r}; pack had: [{preview}]"
            )

    # must_not_contain — no needle may appear in any entry
    for needle in case.must_not_contain:
        if _any_contains(texts, needle):
            reasons.append(
                f"forbidden must_not_contain {needle!r} appeared in pack"
            )

    return CaseResult(
        case_id=case.id,
        passed=not reasons,
        reasons=tuple(reasons),
        matched=tuple(matched),
        pack_size=pack_size,
        abstained=abstained,
    )


# ── Aggregation ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegressionReport:
    """Outcome of running a full case catalog. Used by the CI gate."""
    results: tuple[CaseResult, ...] = ()

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "pack_size": r.pack_size,
                    "abstained": r.abstained,
                    "matched": list(r.matched),
                    "reasons": list(r.reasons),
                }
                for r in self.results
            ],
        }
