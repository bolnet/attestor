"""Phase 3.5 — apply_decisions unit tests with stub mem.

Pure-Python verification that the apply layer routes each Decision to
the correct store call (insert/update/supersede/no-op), and surfaces
errors as ERROR outcomes without raising.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from attestor.conversation.apply import apply_decisions
from attestor.extraction.conflict_resolver import Decision
from attestor.extraction.round_extractor import ExtractedFact
from attestor.models import Memory


def _fact(text: str, category: str = "preference") -> ExtractedFact:
    return ExtractedFact(
        text=text, category=category, entity="user",
        confidence=0.9, source_span=[0, len(text)], speaker="user",
    )


# ── Stub stores ──────────────────────────────────────────────────────────


class StubDocStore:
    def __init__(self) -> None:
        self.rows: dict[str, Memory] = {}
        self.insert_calls: int = 0
        self.update_calls: int = 0

    def insert(self, m: Memory) -> Memory:
        # Echo back with a synthetic id when missing
        if not m.id or len(m.id) < 12:
            m = replace(m, id=f"mem-{len(self.rows) + 1}")
        self.rows[m.id] = m
        self.insert_calls += 1
        return m

    def get(self, mid: str) -> Memory | None:
        return self.rows.get(mid)

    def update(self, m: Memory) -> Memory:
        self.rows[m.id] = m
        self.update_calls += 1
        return m


class StubVectorStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.adds: list[tuple] = []
        self._fail = fail

    def add(self, mid: str, content: str) -> None:
        if self._fail:
            raise RuntimeError("vector store down")
        self.adds.append((mid, content))


class StubTemporal:
    def __init__(self, store: StubDocStore) -> None:
        self.store = store
        self.supersede_calls: list[tuple] = []

    def supersede(self, old: Memory, new_id: str) -> Memory:
        old = replace(old, status="superseded", superseded_by=new_id)
        self.store.update(old)
        self.supersede_calls.append((old.id, new_id))
        return old


class StubMem:
    def __init__(self, *, vector: bool = True, vector_fail: bool = False) -> None:
        self._store = StubDocStore()
        self._vector_store = StubVectorStore(fail=vector_fail) if vector else None
        self._temporal = StubTemporal(self._store)


# ── ADD ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_apply_add_inserts_and_indexes() -> None:
    mem = StubMem()
    decisions = [Decision(
        operation="ADD", new_fact=_fact("user prefers pizza"),
        existing_id=None, rationale="new", evidence_episode_id="ep-1",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id="p1",
        session_id="s1", scope="user",
        extraction_model="test-model",
    )
    assert len(out) == 1
    assert out[0].operation == "ADD"
    assert mem._store.insert_calls == 1
    assert len(mem._vector_store.adds) == 1
    # Persisted memory carries provenance
    persisted = list(mem._store.rows.values())[0]
    assert persisted.user_id == "u1"
    assert persisted.source_episode_id == "ep-1"
    assert persisted.extraction_model == "test-model"
    assert persisted.source_span == [0, len("user prefers pizza")]


@pytest.mark.unit
def test_apply_add_works_without_vector_store() -> None:
    mem = StubMem(vector=False)
    decisions = [Decision(
        operation="ADD", new_fact=_fact("x"), existing_id=None,
        rationale="r", evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "ADD"
    assert mem._store.insert_calls == 1


@pytest.mark.unit
def test_apply_add_vector_failure_does_not_break_insert() -> None:
    """Vector store down → insert still succeeds."""
    mem = StubMem(vector_fail=True)
    decisions = [Decision(
        operation="ADD", new_fact=_fact("x"), existing_id=None,
        rationale="r", evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "ADD"


# ── UPDATE ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_apply_update_refreshes_content() -> None:
    mem = StubMem()
    existing = Memory(
        id="m-1", content="works at Google", category="career",
        confidence=0.7,
    )
    mem._store.rows["m-1"] = existing

    decisions = [Decision(
        operation="UPDATE",
        new_fact=ExtractedFact(
            text="Senior Engineer at Google", category="career",
            entity="Google", confidence=0.95,
            source_span=[0, 25], speaker="user",
        ),
        existing_id="m-1", rationale="refined", evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "UPDATE"
    assert out[0].memory_id == "m-1"
    updated = mem._store.rows["m-1"]
    assert updated.content == "Senior Engineer at Google"
    # Higher confidence wins, never downgrades
    assert updated.confidence == 0.95
    assert updated.entity == "Google"


@pytest.mark.unit
def test_apply_update_missing_target_returns_error() -> None:
    mem = StubMem()
    decisions = [Decision(
        operation="UPDATE", new_fact=_fact("x"),
        existing_id="missing-id", rationale="r", evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "ERROR"


@pytest.mark.unit
def test_apply_update_does_not_downgrade_confidence() -> None:
    mem = StubMem()
    mem._store.rows["m-1"] = Memory(id="m-1", content="x", confidence=0.95)
    decisions = [Decision(
        operation="UPDATE",
        new_fact=ExtractedFact(
            text="x refined", category="general", entity=None,
            confidence=0.4,  # lower
            source_span=[0, 9], speaker="user",
        ),
        existing_id="m-1", rationale="r", evidence_episode_id="ep",
    )]
    apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert mem._store.rows["m-1"].confidence == 0.95


# ── INVALIDATE ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_apply_invalidate_supersedes_and_inserts_new() -> None:
    mem = StubMem()
    old = Memory(id="m-1", content="lives in NYC")
    mem._store.rows["m-1"] = old

    decisions = [Decision(
        operation="INVALIDATE",
        new_fact=ExtractedFact(
            text="lives in SF", category="location", entity="user",
            confidence=0.9, source_span=[0, 11], speaker="user",
        ),
        existing_id="m-1", rationale="contradicts",
        evidence_episode_id="ep-99",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "INVALIDATE"
    assert out[0].memory_id == "m-1"           # the superseded one
    assert out[0].new_memory_id is not None    # the replacement
    # Old row marked superseded (NOT deleted — timeline must replay)
    assert mem._store.rows["m-1"].status == "superseded"
    assert mem._store.rows["m-1"].superseded_by == out[0].new_memory_id
    # supersede was called via temporal manager
    assert len(mem._temporal.supersede_calls) == 1


@pytest.mark.unit
def test_apply_invalidate_when_target_vanished_still_inserts_new() -> None:
    mem = StubMem()
    decisions = [Decision(
        operation="INVALIDATE", new_fact=_fact("new fact"),
        existing_id="vanished", rationale="r", evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "INVALIDATE"
    assert out[0].memory_id is None            # nothing to supersede
    assert out[0].new_memory_id is not None    # but new still inserted


# ── NOOP ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_apply_noop_does_nothing() -> None:
    mem = StubMem()
    decisions = [Decision(
        operation="NOOP", new_fact=_fact("dup"),
        existing_id="m-1", rationale="duplicate",
        evidence_episode_id="ep",
    )]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert out[0].operation == "NOOP"
    assert mem._store.insert_calls == 0
    assert mem._store.update_calls == 0


# ── Mixed batch — order preserved, ERROR isolated ───────────────────────


@pytest.mark.unit
def test_apply_mixed_batch_preserves_order_and_isolates_errors() -> None:
    mem = StubMem()
    # Pre-existing target for UPDATE; vanished target for one ADD's? No,
    # only UPDATE/INVALIDATE need targets.
    mem._store.rows["m-x"] = Memory(id="m-x", content="old")
    decisions = [
        Decision(
            operation="ADD", new_fact=_fact("a"), existing_id=None,
            rationale="r", evidence_episode_id="ep",
        ),
        Decision(
            operation="UPDATE", new_fact=_fact("a refined"),
            existing_id="m-x", rationale="r", evidence_episode_id="ep",
        ),
        Decision(
            operation="UPDATE", new_fact=_fact("c"),
            existing_id="vanished", rationale="r", evidence_episode_id="ep",
        ),
        Decision(
            operation="NOOP", new_fact=_fact("d"), existing_id="m-x",
            rationale="r", evidence_episode_id="ep",
        ),
    ]
    out = apply_decisions(
        decisions, mem=mem, user_id="u1", project_id=None,
        session_id=None, scope="user", extraction_model="m",
    )
    assert [a.operation for a in out] == ["ADD", "UPDATE", "ERROR", "NOOP"]
