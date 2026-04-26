"""Phase 9.1.2 — scorer tests. Pure logic, no I/O, no DB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from evals.regression.cases import RegressionCase
from evals.regression.scorer import (
    CaseResult, RegressionReport, score_case,
)


# ── Lightweight fakes (no dependency on attestor.models in unit tests) ────


@dataclass
class FakeEntry:
    content: str


@dataclass
class FakePack:
    memories: List[FakeEntry]


def _pack(*contents: str) -> FakePack:
    return FakePack(memories=[FakeEntry(content=c) for c in contents])


def _case(**overrides) -> RegressionCase:
    base = dict(
        id="t", description="", category="general",
        ingest=(), query="q?",
    )
    base.update(overrides)
    return RegressionCase(**base)


# ── must_contain ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_must_contain_single_match() -> None:
    c = _case(must_contain=("blue",))
    p = _pack("the user loves blue")
    r = score_case(c, p)
    assert r.passed is True
    assert r.matched == ("blue",)
    assert r.pack_size == 1


@pytest.mark.unit
def test_must_contain_case_insensitive() -> None:
    c = _case(must_contain=("BLUE",))
    p = _pack("user prefers Blue cars")
    assert score_case(c, p).passed is True


@pytest.mark.unit
def test_must_contain_missing_fails() -> None:
    c = _case(must_contain=("blue",))
    p = _pack("user prefers red cars")
    r = score_case(c, p)
    assert r.passed is False
    assert any("missing must_contain 'blue'" in x for x in r.reasons)


@pytest.mark.unit
def test_must_contain_partial_match_fails() -> None:
    """All must_contain needles must appear; failing one fails the case."""
    c = _case(must_contain=("blue", "shoes"))
    p = _pack("user prefers blue cars")
    r = score_case(c, p)
    assert r.passed is False
    assert r.matched == ("blue",)
    assert any("missing must_contain 'shoes'" in x for x in r.reasons)


# ── must_not_contain ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_must_not_contain_blocks_forbidden_substring() -> None:
    """Used for supersession: after 'I prefer red', recall must NOT
    surface the old 'I prefer blue' fact."""
    c = _case(
        must_contain=("red",),
        must_not_contain=("blue",),
    )
    p = _pack("user prefers red cars", "user used to prefer blue")
    r = score_case(c, p)
    assert r.passed is False
    assert any("forbidden must_not_contain 'blue'" in x for x in r.reasons)


@pytest.mark.unit
def test_must_not_contain_clean_passes() -> None:
    c = _case(
        must_contain=("red",),
        must_not_contain=("blue",),
    )
    p = _pack("user prefers red cars")
    assert score_case(c, p).passed is True


# ── abstention ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_abstain_required_passes_on_empty_pack() -> None:
    c = _case(category="abstention", abstain_required=True)
    r = score_case(c, _pack())
    assert r.passed is True
    assert r.abstained is True


@pytest.mark.unit
def test_abstain_required_fails_when_pack_has_memories() -> None:
    c = _case(category="abstention", abstain_required=True)
    r = score_case(c, _pack("some unrelated fact"))
    assert r.passed is False
    assert any("expected abstention" in x for x in r.reasons)


@pytest.mark.unit
def test_abstain_ok_lets_empty_pack_pass() -> None:
    """abstain_ok means: ideally we'd see the fact, but an empty pack
    also passes (under-recall is acceptable for this case)."""
    c = _case(must_contain=("blue",), abstain_ok=True)
    r = score_case(c, _pack())
    assert r.passed is True
    assert r.abstained is True


@pytest.mark.unit
def test_abstain_ok_does_not_excuse_wrong_recall() -> None:
    """abstain_ok permits empty, NOT permits returning the wrong thing."""
    c = _case(
        must_contain=("blue",),
        must_not_contain=("red",),
        abstain_ok=True,
    )
    r = score_case(c, _pack("user prefers red"))
    assert r.passed is False  # red appeared → forbidden


# ── Reporting ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_aggregates_pass_fail_counts() -> None:
    rep = RegressionReport(results=(
        CaseResult(case_id="a", passed=True),
        CaseResult(case_id="b", passed=False, reasons=("x",)),
        CaseResult(case_id="c", passed=True),
    ))
    assert rep.total == 3
    assert rep.passed == 2
    assert rep.failed == 1
    assert rep.pass_rate == pytest.approx(2 / 3)
    assert [f.case_id for f in rep.failures()] == ["b"]


@pytest.mark.unit
def test_report_pass_rate_empty_is_one() -> None:
    """No cases = vacuously passing. Avoids ZeroDivisionError on bootstrap."""
    rep = RegressionReport(results=())
    assert rep.pass_rate == 1.0


@pytest.mark.unit
def test_report_to_dict_round_trip() -> None:
    rep = RegressionReport(results=(
        CaseResult(case_id="a", passed=True, matched=("foo",), pack_size=2),
        CaseResult(case_id="b", passed=False,
                   reasons=("missing X",), pack_size=0, abstained=True),
    ))
    d = rep.to_dict()
    assert d["total"] == 2 and d["passed"] == 1 and d["failed"] == 1
    assert d["results"][0]["matched"] == ["foo"]
    assert d["results"][1]["abstained"] is True
