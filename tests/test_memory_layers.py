"""Hierarchical memory layers — episodic / semantic / procedural / working.

Background — 2026 best-practice (Letta, LangGraph memory, Mem0):
    Attestor stored every fact in a flat namespace, distinguished only by
    `category`. The hierarchical-layer pattern adds a `layer` field that
    classifies the memory's intent:

      - episodic   (default) — what happened in conversation
      - semantic   — what is true (distilled facts; reflection pipeline)
      - procedural — how to do things (skills, workflows, recipes)
      - working    — session-scoped, ephemeral

Recall semantics:
    Default recall returns episodic + semantic — the natural answer to
    "what do I know about X". Procedural memories surface only when the
    caller asks for them (`layers=('procedural',)`); working memories
    are session-scoped and auto-purged on session end (the cron path is
    out of scope for this PR; the layer column + plumbing is in).

Most checks here are pure-unit (no Postgres). Three live-PG integration
tests at the bottom are env-gated through the standard ``mem`` fixture
in conftest.py.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import pytest

from attestor.models import LAYER_DEFAULT, VALID_LAYERS, Memory


# ── Memory.layer field ───────────────────────────────────────────────────


@pytest.mark.unit
def test_default_layer_is_episodic():
    """A bare Memory() with no layer arg defaults to 'episodic'."""
    m = Memory(content="hello")
    assert m.layer == "episodic"
    assert LAYER_DEFAULT == "episodic"


@pytest.mark.unit
def test_explicit_layer_persists():
    """Caller-provided layer round-trips through the dataclass."""
    m = Memory(content="run pytest -q", layer="procedural")
    assert m.layer == "procedural"


@pytest.mark.unit
def test_invalid_layer_raises():
    """Layer is a closed vocabulary. 'garbage' must raise ValueError."""
    with pytest.raises(ValueError) as exc:
        Memory(content="x", layer="garbage")
    assert "layer" in str(exc.value).lower()
    msg = str(exc.value)
    for ok_layer in ("episodic", "semantic", "procedural", "working"):
        assert ok_layer in msg


@pytest.mark.unit
def test_valid_layers_constant_is_closed():
    """The vocabulary is exactly 4 layers — no more, no less."""
    assert VALID_LAYERS == frozenset(
        {"episodic", "semantic", "procedural", "working"}
    )


@pytest.mark.unit
def test_to_dict_includes_layer():
    """Memory.to_dict round-trips the layer field for JSON export."""
    m = Memory(content="how to deploy", layer="procedural")
    d = m.to_dict()
    assert d["layer"] == "procedural"


# ── Memory.from_row layer round-trip ─────────────────────────────────────


@pytest.mark.unit
def test_from_row_with_layer_column():
    """Row with explicit layer column → Memory.layer matches."""
    row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "content": "distilled fact",
        "user_id": "00000000-0000-0000-0000-0000000000aa",
        "tags": [],
        "layer": "semantic",
    }
    mem = Memory.from_row(row)
    assert mem.layer == "semantic"


@pytest.mark.unit
def test_from_row_without_layer_falls_back_to_episodic():
    """Pre-existing rows that pre-date the column → 'episodic'."""
    row = {
        "id": "v3oldrow0001",
        "content": "old conversational fact",
        "tags": [],
        # NB: no `layer` key — pre-migration row
    }
    mem = Memory.from_row(row)
    assert mem.layer == "episodic"


@pytest.mark.unit
def test_from_row_with_invalid_layer_falls_back_to_episodic():
    """Defensive: a corrupted DB value should not crash from_row.
    Read-side coerces unknown values to 'episodic' (the safe default)."""
    row = {
        "id": "row-0002",
        "content": "x",
        "tags": [],
        "layer": "totally-bogus",
    }
    mem = Memory.from_row(row)
    assert mem.layer == "episodic"


# ── Reflection / consolidation pipeline integration ─────────────────────


@pytest.mark.unit
def test_consolidation_layer_default_is_semantic():
    """Distilled memories from the consolidation/apply path land in the
    semantic layer by default — that's the core "what is true" signal
    consolidation produces."""
    from attestor.conversation.apply import _fact_to_memory

    class _StubFact:
        text = "alice prefers python"
        category = "preference"
        entity = "alice"
        confidence = 0.9
        source_span = (0, 12)
        speaker = "user"

    mem = _fact_to_memory(
        _StubFact(),
        user_id="00000000-0000-0000-0000-0000000000aa",
        project_id=None,
        session_id=None,
        scope="user",
        source_episode_id="ep-1",
        extraction_model="consolidation:test",
    )
    assert mem.layer == "semantic"


# ── Schema migration (DDL only — no live PG) ─────────────────────────────


@pytest.mark.unit
def test_v3_schema_includes_layer_column():
    """The v3 inline DDL must declare the layer column + ALTER fallback +
    index."""
    import inspect

    from attestor.store.postgres_backend import PostgresBackend

    src = inspect.getsource(PostgresBackend._init_schema)
    assert "layer" in src
    assert "ADD COLUMN IF NOT EXISTS layer" in src
    assert "idx_memories_layer" in src


@pytest.mark.unit
def test_v4_schema_includes_layer_column():
    """``attestor/store/schema.sql`` must include the layer column + ALTER
    fallback + index."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parent.parent
        / "attestor"
        / "store"
        / "schema.sql"
    ).read_text()

    assert "layer" in sql
    assert "ADD COLUMN IF NOT EXISTS layer" in sql
    assert "idx_memories_layer" in sql


# ── Insert path: layer in INSERT params ──────────────────────────────────


@pytest.mark.unit
def test_postgres_insert_params_include_layer():
    """``_PostgresDocumentMixin._memory_to_params`` must surface the
    layer field so both v3 and v4 INSERT branches can write it."""
    from attestor.store._postgres_document import _PostgresDocumentMixin

    class _Stub(_PostgresDocumentMixin):
        _v4 = False

    stub = _Stub()
    m = Memory(content="x", layer="procedural")
    p = stub._memory_to_params(m)
    assert "layer" in p
    assert p["layer"] == "procedural"


@pytest.mark.unit
def test_v4_insert_sql_writes_layer_column():
    """The v4 INSERT statement must include ``layer`` in the column list
    and a matching ``%(layer)s`` placeholder."""
    import inspect

    from attestor.store._postgres_document import _PostgresDocumentMixin

    src = inspect.getsource(_PostgresDocumentMixin.insert)
    assert "layer" in src
    assert "%(layer)s" in src


@pytest.mark.unit
def test_v3_insert_sql_writes_layer_column():
    """The v3 INSERT statement must include the layer column too."""
    import inspect

    from attestor.store._postgres_document import _PostgresDocumentMixin

    src = inspect.getsource(_PostgresDocumentMixin.insert)
    # v3 + v4 INSERTs both reference the column → at least 2 occurrences.
    assert src.count("layer") >= 2


# ── list_memories layer filter ───────────────────────────────────────────


def _build_v4_backend_stub(captured_sql: list, captured_params: list):
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = True
    backend._execute = lambda sql, params=None: (
        captured_sql.append(sql)
        or captured_params.append(params)
        or []
    )
    return backend


def _build_v3_backend_stub(captured_sql: list, captured_params: list):
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = False
    backend._execute = lambda sql, params=None: (
        captured_sql.append(sql)
        or captured_params.append(params)
        or []
    )
    return backend


@pytest.mark.unit
def test_list_memories_filters_on_layer_v4():
    """``list_memories(layer='procedural')`` must add a layer WHERE
    clause so callers can list a single layer without pulling the
    whole user's memories."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.list_memories(layer="procedural")

    assert sqls
    assert "layer = %(layer)s" in sqls[0], sqls[0]
    assert params_log[0]["layer"] == "procedural"


@pytest.mark.unit
def test_list_memories_filters_on_layer_v3():
    sqls: list = []
    params_log: list = []
    backend = _build_v3_backend_stub(sqls, params_log)

    backend.list_memories(layer="semantic")

    assert sqls
    assert "layer = %(layer)s" in sqls[0], sqls[0]
    assert params_log[0]["layer"] == "semantic"


@pytest.mark.unit
def test_list_memories_filter_supports_layer_set():
    """list_memories(layer=['episodic','semantic']) → IN clause so the
    orchestrator's default layers filter (episodic + semantic) can push
    down to the document store."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.list_memories(layer=["episodic", "semantic"])

    assert sqls
    sql = sqls[0]
    # Must use ANY(...)/IN(...) — not equality — when filter is a list.
    assert "layer = ANY(%(layer)s)" in sql or "layer IN " in sql, sql
    assert sorted(params_log[0]["layer"]) == ["episodic", "semantic"]


# ── AgentMemory.add() / recall() argument validation ────────────────────


@pytest.mark.unit
def test_add_layer_argument_present():
    """``AgentMemory.add`` must expose a ``layer`` keyword argument."""
    import inspect

    from attestor.core.agent_memory import AgentMemory

    sig = inspect.signature(AgentMemory.add)
    assert "layer" in sig.parameters, sig.parameters
    assert sig.parameters["layer"].default == "episodic"


@pytest.mark.unit
def test_recall_layers_argument_present():
    """``AgentMemory.recall`` must expose a ``layers`` keyword argument
    with the documented (episodic, semantic) default."""
    import inspect

    from attestor.core.agent_memory import AgentMemory

    sig = inspect.signature(AgentMemory.recall)
    assert "layers" in sig.parameters, sig.parameters
    default = sig.parameters["layers"].default
    # Default = ('episodic', 'semantic') — natural answer to "what do
    # I know about X".
    assert tuple(default) == ("episodic", "semantic")


@pytest.mark.unit
def test_add_skill_helper_signature():
    """``AgentMemory.add_skill(name, content, ...)`` must exist."""
    import inspect

    from attestor.core.agent_memory import AgentMemory

    assert hasattr(AgentMemory, "add_skill")
    sig = inspect.signature(AgentMemory.add_skill)
    params = sig.parameters
    assert "name" in params
    assert "content" in params


# ── Layer-aware scoring (postprocess) ────────────────────────────────────


@pytest.mark.unit
def test_layer_score_boost_semantic_over_episodic():
    """Semantic memories carry a small score boost over episodic when the
    same content/score appears on both layers — codifies the recall
    preference defined in the PR description.

    The boost is small (~0.05) and configurable; this test just pins
    the inequality so a future tuning change doesn't accidentally
    invert it."""
    from attestor.models import Memory, RetrievalResult
    from attestor.retrieval.scorer import layer_boost

    ep = RetrievalResult(
        memory=Memory(content="x", layer="episodic"),
        score=0.5,
        match_source="vector",
    )
    se = RetrievalResult(
        memory=Memory(content="x", layer="semantic"),
        score=0.5,
        match_source="vector",
    )
    boosted = layer_boost([ep, se])
    by_layer = {r.memory.layer: r.score for r in boosted}
    assert by_layer["semantic"] >= by_layer["episodic"]


@pytest.mark.unit
def test_layer_score_boost_disabled_returns_unchanged():
    """layer_boost with weights={'episodic':0,'semantic':0} → no-op."""
    from attestor.models import Memory, RetrievalResult
    from attestor.retrieval.scorer import layer_boost

    items = [
        RetrievalResult(
            memory=Memory(content="x", layer="episodic"),
            score=0.5,
            match_source="vector",
        ),
        RetrievalResult(
            memory=Memory(content="y", layer="semantic"),
            score=0.5,
            match_source="vector",
        ),
    ]
    out = layer_boost(items, weights={"episodic": 0.0, "semantic": 0.0})
    assert all(o.score == 0.5 for o in out)


# ── Live integration tests (env-gated through `mem` fixture) ────────────
# Marked with the standard `live` mark; conftest.py skips them when
# POSTGRES_URL is not set.


@pytest.mark.live
def test_live_layer_column_round_trips(mem):
    """Write a procedural memory, fetch it back, layer field round-trips."""
    out = mem.add("how to deploy", layer="procedural", entity="deploy")
    fetched = mem.get(out.id)
    assert fetched is not None
    assert fetched.layer == "procedural"


@pytest.mark.live
def test_live_default_recall_excludes_procedural(mem):
    """Default recall (layers=(episodic, semantic)) must NOT return a
    procedural memory."""
    mem.add("alice prefers python", layer="semantic", entity="alice")
    mem.add("how to greet alice", layer="procedural", entity="alice")

    results = mem.recall("alice python")
    layers = {r.memory.layer for r in results}
    assert "procedural" not in layers, (
        f"default recall returned procedural memory; layers={layers}"
    )


@pytest.mark.live
def test_live_layers_filter_procedural_only(mem):
    mem.add("alice prefers python", layer="semantic", entity="alice")
    mem.add("how to greet alice", layer="procedural", entity="alice")

    results = mem.recall("alice", layers=("procedural",))
    if results:
        assert all(r.memory.layer == "procedural" for r in results)


@pytest.mark.live
def test_live_v4_schema_alter_idempotent(mem):
    """Re-running the schema apply (CREATE TABLE IF NOT EXISTS + ALTER ADD
    COLUMN IF NOT EXISTS) must not break — verifies the migration is
    safe to run twice."""
    backend = mem._store
    if not getattr(backend, "_v4", False):
        pytest.skip("v3 not under test in this fixture")
    # Re-apply the schema. This MUST succeed on an already-initialized DB.
    backend._init_schema_v4()


@pytest.mark.live
def test_live_v3_schema_alter_idempotent(mem):
    """Re-applying the v3 inline DDL (idempotent) must not fail."""
    backend = mem._store
    if getattr(backend, "_v4", False):
        pytest.skip("v4 path covered by sibling test")
    backend._init_schema()
