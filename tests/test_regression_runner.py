"""Phase 9.1.3 — runner tests.

Pure unit tests using a fake memory adapter. No DB, no LLM. Verifies the
runner's iteration, error-isolation, recall-kwarg threading, and the
report-aggregation glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from evals.regression.cases import RegressionCase, Round
from evals.regression.runner import (
    _parse_iso, _recall_kwargs,
    run_case, run_regression,
)


# ── Fakes ─────────────────────────────────────────────────────────────────


@dataclass
class FakeEntry:
    content: str


@dataclass
class FakePack:
    memories: list[FakeEntry]


@dataclass
class FakeMem:
    """Records every ingest, returns a pre-canned pack on recall."""
    pack_to_return: FakePack = field(default_factory=lambda: FakePack([]))
    ingested: list[tuple] = field(default_factory=list)
    recall_calls: list[tuple] = field(default_factory=list)
    raise_on_recall: Exception | None = None
    raise_on_ingest: Exception | None = None

    def ingest_round(self, user_turn: Any, assistant_turn: Any,
                     **kwargs: Any) -> None:
        if self.raise_on_ingest is not None:
            raise self.raise_on_ingest
        self.ingested.append((user_turn, assistant_turn, kwargs))

    def recall_as_pack(self, query: str, **kwargs: Any) -> FakePack:
        if self.raise_on_recall is not None:
            raise self.raise_on_recall
        self.recall_calls.append((query, kwargs))
        return self.pack_to_return


def _case(**kw) -> RegressionCase:
    base = dict(
        id="t", description="", category="general",
        ingest=(Round(user="hi", assistant="hello"),),
        query="q?",
    )
    base.update(kw)
    return RegressionCase(**base)


# ── _parse_iso / _recall_kwargs ──────────────────────────────────────────


@pytest.mark.unit
def test_parse_iso_handles_z_suffix() -> None:
    dt = _parse_iso("2026-04-26T10:30:00Z")
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.hour == 10


@pytest.mark.unit
def test_parse_iso_none_returns_none() -> None:
    assert _parse_iso(None) is None


@pytest.mark.unit
def test_recall_kwargs_passes_as_of() -> None:
    c = _case(as_of="2026-04-26T10:00:00Z")
    kw = _recall_kwargs(c)
    assert "as_of" in kw and isinstance(kw["as_of"], datetime)


@pytest.mark.unit
def test_recall_kwargs_passes_time_window() -> None:
    c = _case(
        time_window_start="2026-01-01T00:00:00Z",
        time_window_end="2026-01-31T23:59:59Z",
    )
    kw = _recall_kwargs(c)
    assert "time_window" in kw
    start, end = kw["time_window"]
    assert isinstance(start, datetime) and isinstance(end, datetime)


@pytest.mark.unit
def test_recall_kwargs_omits_optional_fields() -> None:
    c = _case()
    assert _recall_kwargs(c) == {}


# ── run_case ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_run_case_happy_path_passes() -> None:
    mem = FakeMem(pack_to_return=FakePack([FakeEntry("user loves blue")]))
    c = _case(must_contain=("blue",))
    r = run_case(c, mem)
    assert r.passed is True
    assert r.matched == ("blue",)
    assert len(mem.ingested) == 1
    assert mem.recall_calls == [("q?", {})]


@pytest.mark.unit
def test_run_case_threads_recall_kwargs() -> None:
    mem = FakeMem()
    c = _case(as_of="2026-04-26T10:00:00Z")
    run_case(c, mem)
    assert "as_of" in mem.recall_calls[0][1]


@pytest.mark.unit
def test_run_case_handles_ingest_error_gracefully() -> None:
    """A broken ingest must NOT propagate — record it and move on."""
    mem = FakeMem(raise_on_ingest=RuntimeError("db on fire"))
    c = _case()
    r = run_case(c, mem)
    assert r.passed is False
    assert any("RuntimeError" in x for x in r.reasons)


@pytest.mark.unit
def test_run_case_handles_recall_error_gracefully() -> None:
    mem = FakeMem(raise_on_recall=RuntimeError("vector store down"))
    c = _case()
    r = run_case(c, mem)
    assert r.passed is False
    assert any("RuntimeError" in x for x in r.reasons)


@pytest.mark.unit
def test_run_case_ingest_passes_thread_id_uniquely() -> None:
    """Each case uses a unique thread_id to keep episodes separate."""
    mem1 = FakeMem()
    mem2 = FakeMem()
    run_case(_case(id="a"), mem1)
    run_case(_case(id="b"), mem2)
    a_thread = mem1.ingested[0][0].thread_id
    b_thread = mem2.ingested[0][0].thread_id
    assert a_thread != b_thread
    assert a_thread.startswith("reg-")


# ── run_regression ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_run_regression_aggregates_results() -> None:
    mem = FakeMem(pack_to_return=FakePack([FakeEntry("user loves blue")]))
    cases = [
        _case(id="pass1", must_contain=("blue",)),
        _case(id="fail1", must_contain=("red",)),
        _case(id="pass2", must_contain=("blue",)),
    ]
    rep = run_regression(cases, mem)
    assert rep.total == 3 and rep.passed == 2 and rep.failed == 1
    assert {f.case_id for f in rep.failures()} == {"fail1"}


@pytest.mark.unit
def test_run_regression_invokes_isolate_between_cases() -> None:
    mem = FakeMem(pack_to_return=FakePack([FakeEntry("foo")]))
    calls: list[int] = []

    def isolate() -> None:
        calls.append(len(mem.ingested))

    run_regression(
        [_case(id="a", must_contain=("foo",)),
         _case(id="b", must_contain=("foo",))],
        mem, isolate=isolate,
    )
    # isolate fires before each case, so before any ingest happens for
    # that case → the count snapshots are 0 then 1 (one ingest per case)
    assert calls == [0, 1]


@pytest.mark.unit
def test_run_regression_isolate_failure_does_not_kill_suite() -> None:
    mem = FakeMem(pack_to_return=FakePack([FakeEntry("foo")]))

    def isolate() -> None:
        raise RuntimeError("schema reset failed")

    rep = run_regression(
        [_case(id="a", must_contain=("foo",))],
        mem, isolate=isolate,
    )
    # The case still ran and scored
    assert rep.total == 1 and rep.passed == 1


@pytest.mark.unit
def test_run_yaml_loads_and_executes(tmp_path) -> None:
    """End-to-end: YAML on disk → loaded → run → report."""
    from evals.regression.runner import run_yaml

    yaml_path = tmp_path / "qa.yaml"
    yaml_path.write_text("""
- id: simple
  category: preference
  query: what color?
  ingest:
    - user: I love blue
      assistant: noted
  must_contain: ["blue"]
""")
    mem = FakeMem(pack_to_return=FakePack([FakeEntry("user loves blue")]))
    rep = run_yaml(str(yaml_path), mem)
    assert rep.total == 1 and rep.passed == 1
