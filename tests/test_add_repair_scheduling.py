"""AgentMemory.add() — durable repair is scheduled ONLY on derived-write
failure and ONLY when ``durable.enabled`` is true.

Also covers the re-read seam the DeriveMemory activities call
(``derive_vector`` / ``derive_graph``): content comes from the document
store by id, never from the Temporal payload.

Fake stores; no Postgres / Pinecone / Neo4j / Temporal.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest

from attestor.config import DurableCfg
from attestor.core import derive_service
from attestor.core.derive_service import (
    DerivedStoreUnavailableError,
    MemoryNotFoundError,
    embed_payload_for,
)


class _FakeDocumentStore:
    ROLES = {"document"}

    def __init__(self) -> None:
        self.memories: dict[str, Any] = {}
        self.by_hash: dict[str, Any] = {}

    def insert(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory
        if memory.content_hash:
            self.by_hash[memory.content_hash] = memory
        return memory

    def get(self, memory_id, requester_agent_id=None):  # noqa: ANN001
        return self.memories.get(memory_id)

    def get_by_hash(self, content_hash, namespace="default"):  # noqa: ANN001
        return self.by_hash.get(content_hash)

    def update(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory
        return memory

    def list_memories(self, **kwargs):
        return list(self.memories.values())

    def stats(self):
        return {"total_memories": len(self.memories)}

    def close(self):
        pass

    def execute(self, sql, params=None):
        return []

    def increment_access(self, ids):
        pass


class _FakeVectorStore:
    ROLES = {"vector"}

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.adds: list[tuple[str, str, str]] = []

    def add(self, memory_id, content, namespace="default"):  # noqa: ANN001
        if self.fail:
            raise self.fail
        self.adds.append((memory_id, content, namespace))

    def search(self, query, limit=20, namespace=None):  # noqa: ANN001
        return []

    def count(self):
        return len(self.adds)

    def close(self):
        pass


class _FakeGraphStore:
    ROLES = {"graph"}

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.entities: list[tuple[str, str]] = []
        self.relations: list[tuple[str, str, str]] = []

    def add_entity(self, name, entity_type="general", attributes=None, namespace="default"):  # noqa: ANN001
        if self.fail:
            raise self.fail
        self.entities.append((name, namespace))

    def add_relation(  # noqa: ANN001
        self, src, dst, relation_type="related_to", metadata=None, namespace="default",
    ):
        self.relations.append((src, dst, namespace))

    def get_related(self, entity, depth=2):  # noqa: ANN001
        return []

    def get_subgraph(self, *a, **k):
        return {}

    def get_entities(self, *a, **k):
        return []

    def get_edges(self, entity):  # noqa: ANN001
        return []

    def graph_stats(self):
        return {}

    def save(self):
        pass

    def close(self):
        pass


def _build_mem(
    *,
    vector_fail: Exception | None = None,
    graph_fail: Exception | None = None,
    store: Any = None,
    vec: Any = None,
    graph: Any = None,
):
    from attestor.core.agent_memory import AgentMemory

    store = store if store is not None else _FakeDocumentStore()
    vec = vec if vec is not None else _FakeVectorStore(vector_fail)
    graph = graph if graph is not None else _FakeGraphStore(graph_fail)
    tmp = tempfile.mkdtemp()
    with patch("attestor.core.agent_memory._registry") as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {
            "document": "postgres", "vector": "vector", "graph": "graph",
        }
        _reg.return_value.instantiate_backend.side_effect = (
            lambda name, path, bcfg: {"postgres": store, "vector": vec, "graph": graph}[name]
        )
        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)
    return mem, store, vec, graph


@pytest.fixture
def repair_spy(monkeypatch):
    """Capture repair dispatches + trace events; pin the durable config."""
    pytest.importorskip("temporalio")
    from attestor import trace as _tr
    from attestor.durable import dispatch

    calls: list[tuple[Any, Any]] = []
    events: list[tuple[str, dict]] = []
    holder: dict = {"cfg": DurableCfg(enabled=False), "raise": None}

    def fake_schedule(cfg, request, *, wait_seconds=None):
        calls.append((cfg, request))
        if holder["raise"]:
            raise holder["raise"]
        return f"derive-{request.memory.memory_id}"

    monkeypatch.setattr(dispatch, "schedule_derive", fake_schedule)
    monkeypatch.setattr(derive_service, "resolve_durable_cfg", lambda: holder["cfg"])
    monkeypatch.setattr(_tr, "is_enabled", lambda: True)
    monkeypatch.setattr(_tr, "event", lambda name, **f: events.append((name, f)))
    return {"calls": calls, "events": events, "holder": holder}


def _repair_events(spy) -> list[dict]:
    return [f for name, f in spy["events"] if name == "ingest.write.repair_scheduled"]


# ── add(): repair only on failure, only when enabled ────────────────────

@pytest.mark.unit
def test_add_schedules_repair_when_vector_fails_and_durable_enabled(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True, task_queue="tq")
    mem, store, vec, graph = _build_mem(vector_fail=RuntimeError("pinecone down"))
    try:
        m = mem.add("User prefers Python.", tags=["pref"], namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories  # document path never breaks
    assert len(repair_spy["calls"]) == 1
    cfg, request = repair_spy["calls"][0]
    assert cfg.task_queue == "tq"
    assert request.memory.memory_id == m.id
    assert request.memory.namespace == "ns"
    # ids-only payload
    assert "Python" not in repr(request)
    events = _repair_events(repair_spy)
    assert len(events) == 1
    assert events[0]["memory_id"] == m.id
    assert events[0]["workflow_id"] == f"derive-{m.id}"
    assert events[0]["lanes"] == ["vector"]


@pytest.mark.unit
def test_add_schedules_repair_when_graph_fails_and_durable_enabled(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, _store, vec, _graph = _build_mem(graph_fail=RuntimeError("neo4j down"))
    try:
        m = mem.add("Alice uses Postgres.", entity="Alice", namespace="ns")
    finally:
        mem.close()
    assert vec.adds
    assert vec.adds[0][0] == m.id
    assert len(repair_spy["calls"]) == 1
    assert _repair_events(repair_spy)[0]["lanes"] == ["graph"]


@pytest.mark.unit
def test_add_schedules_one_repair_when_both_lanes_fail(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, _s, _v, _g = _build_mem(
        vector_fail=RuntimeError("v"), graph_fail=RuntimeError("g"),
    )
    try:
        mem.add("Both lanes down.", entity="X", namespace="ns")
    finally:
        mem.close()
    assert len(repair_spy["calls"]) == 1
    assert _repair_events(repair_spy)[0]["lanes"] == ["vector", "graph"]


@pytest.mark.unit
def test_add_does_not_schedule_repair_when_writes_succeed(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, _s, vec, graph = _build_mem()
    try:
        m = mem.add("Alice uses Postgres.", entity="Alice", namespace="ns")
    finally:
        mem.close()
    assert vec.adds[0][0] == m.id
    assert graph.entities
    assert repair_spy["calls"] == []
    assert _repair_events(repair_spy) == []


@pytest.mark.unit
def test_add_does_not_schedule_repair_when_durable_disabled(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=False)
    mem, store, _v, _g = _build_mem(vector_fail=RuntimeError("pinecone down"))
    try:
        m = mem.add("User prefers Python.", namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories
    assert repair_spy["calls"] == []
    assert _repair_events(repair_spy) == []


@pytest.mark.unit
def test_add_survives_when_dispatch_itself_fails(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    repair_spy["holder"]["raise"] = RuntimeError("temporal unreachable")
    mem, store, _v, _g = _build_mem(vector_fail=RuntimeError("pinecone down"))
    try:
        m = mem.add("User prefers Python.", namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories
    assert len(repair_spy["calls"]) == 1
    assert _repair_events(repair_spy) == []  # nothing was scheduled


@pytest.mark.unit
def test_repair_request_carries_owner_and_agent_ids_only(repair_spy):
    """The worker scopes RLS to ``user_id`` and applies the private-visibility
    guard as ``agent_id`` — both must ride along on the ids-only payload."""
    from attestor.durable.models import DeriveRef
    from attestor.models import Memory

    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, _s, _v, _g = _build_mem()
    owned = Memory(id="m-own", content="private text", namespace="ns",
                   user_id="u-1", agent_id="planner", visibility="private")
    try:
        wid = mem._schedule_derive_repair(owned, lanes=("vector",))
    finally:
        mem.close()
    assert wid == "derive-m-own"
    _cfg, request = repair_spy["calls"][0]
    assert request.memory == DeriveRef(
        memory_id="m-own", namespace="ns", user_id="u-1", agent_id="planner",
    )
    assert "private text" not in repr(request)


@pytest.mark.unit
def test_add_with_config_unavailable_stays_in_process(monkeypatch):
    """No YAML / no durable block → identical to today: log + continue."""
    monkeypatch.setattr(derive_service, "resolve_durable_cfg", lambda: None)
    mem, store, _v, _g = _build_mem(vector_fail=RuntimeError("pinecone down"))
    try:
        m = mem.add("hello", namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories


# ── derive_vector / derive_graph re-read seam ───────────────────────────

@pytest.mark.unit
def test_derive_vector_re_reads_content_and_writes_same_payload_as_add():
    mem, store, vec, _g = _build_mem()
    try:
        m = mem.add("User prefers Python.", namespace="ns")
        first = vec.adds[-1]
        out = mem.derive_vector(m.id)
    finally:
        mem.close()
    assert out.memory_id == m.id
    assert out.namespace == "ns"
    assert vec.adds[-1] == first  # idempotent re-derive writes the identical payload


@pytest.mark.unit
def test_derive_vector_honours_stored_context_prefix():
    mem, store, vec, _g = _build_mem()
    try:
        m = mem.add("User prefers Python.", namespace="ns")
        store.memories[m.id] = replace(m, metadata={"_context_prefix": "About langs."})
        mem.derive_vector(m.id)
    finally:
        mem.close()
    assert vec.adds[-1][1] == "[CTX] About langs.\n\nUser prefers Python."
    assert embed_payload_for(store.memories[m.id]) == vec.adds[-1][1]


@pytest.mark.unit
def test_derive_graph_re_reads_and_extracts():
    mem, _s, _v, graph = _build_mem()
    try:
        m = mem.add("Alice uses Postgres.", entity="Alice", namespace="ns")
        before = list(graph.entities)
        out = mem.derive_graph(m.id)
    finally:
        mem.close()
    assert out.memory_id == m.id
    assert out.namespace == "ns"
    assert out.entity_count >= 1
    assert graph.entities[len(before):] == before  # same nodes, same namespace (idempotent)


@pytest.mark.unit
def test_derive_raises_when_memory_missing():
    mem, _s, _v, _g = _build_mem()
    try:
        with pytest.raises(MemoryNotFoundError):
            mem.derive_vector("nope")
        with pytest.raises(MemoryNotFoundError):
            mem.derive_graph("nope")
    finally:
        mem.close()


@pytest.mark.unit
def test_derive_raises_retryable_when_store_unavailable(monkeypatch):
    mem, _s, _v, _g = _build_mem()
    try:
        m = mem.add("x", namespace="ns")
        mem._vector_store = None
        mem._graph = None
        monkeypatch.setattr(mem, "_try_recover_stores", lambda: {})
        with pytest.raises(DerivedStoreUnavailableError):
            mem.derive_vector(m.id)
        with pytest.raises(DerivedStoreUnavailableError):
            mem.derive_graph(m.id)
    finally:
        mem.close()


# ── cold-start outage: vector store failed to initialise ────────────────
#
# Found in live e2e (2026-09-03): with Pinecone down at AgentMemory
# construction the vector backend never initialises, ``_vector_sink()`` is
# None, the write is skipped without raising, and NO repair was scheduled —
# the only outage shape that left derived state permanently missing.

def _build_mem_vector_init_fails():
    from attestor.core.agent_memory import AgentMemory
    store = _FakeDocumentStore()
    graph = _FakeGraphStore()
    tmp = tempfile.mkdtemp()

    def instantiate(name, path, bcfg):  # noqa: ANN001
        if name == "vector":
            raise RuntimeError("pinecone control plane unreachable")
        return {"postgres": store, "graph": graph}[name]

    with patch("attestor.core.agent_memory._registry") as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {
            "document": "postgres", "vector": "vector", "graph": "graph",
        }
        _reg.return_value.instantiate_backend.side_effect = instantiate
        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)
    return mem, store


@pytest.mark.unit
def test_add_schedules_vector_repair_when_vector_store_never_initialised(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, store = _build_mem_vector_init_fails()
    assert mem._vector_store is None
    try:
        m = mem.add("Cold-start outage.", namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories
    assert len(repair_spy["calls"]) == 1
    assert _repair_events(repair_spy)[0]["lanes"] == ["vector"]


@pytest.mark.unit
def test_add_does_not_schedule_when_no_vector_role_configured(repair_spy):
    """No vector backend in the role map at all → nothing to repair."""
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    from attestor.core.agent_memory import AgentMemory
    store = _FakeDocumentStore()
    tmp = tempfile.mkdtemp()
    with patch("attestor.core.agent_memory._registry") as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {"document": "postgres"}
        _reg.return_value.instantiate_backend.side_effect = (
            lambda name, path, bcfg: {"postgres": store}[name]
        )
        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)
    try:
        mem.add("No vector role.", namespace="ns")
    finally:
        mem.close()
    assert repair_spy["calls"] == []


def _build_mem_graph_init_fails():
    from attestor.core.agent_memory import AgentMemory
    store = _FakeDocumentStore()
    vec = _FakeVectorStore()
    tmp = tempfile.mkdtemp()

    def instantiate(name, path, bcfg):  # noqa: ANN001
        if name == "graph":
            raise RuntimeError("neo4j unreachable")
        return {"postgres": store, "vector": vec}[name]

    with patch("attestor.core.agent_memory._registry") as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {
            "document": "postgres", "vector": "vector", "graph": "graph",
        }
        _reg.return_value.instantiate_backend.side_effect = instantiate
        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)
    return mem, store


@pytest.mark.unit
def test_add_schedules_graph_repair_when_graph_store_never_initialised(repair_spy):
    repair_spy["holder"]["cfg"] = DurableCfg(enabled=True)
    mem, store = _build_mem_graph_init_fails()
    assert mem._graph is None and mem._graph_init_failed is True
    try:
        m = mem.add("Alice uses Postgres.", entity="Alice", namespace="ns")
    finally:
        mem.close()
    assert m.id in store.memories
    assert _repair_events(repair_spy)[0]["lanes"] == ["graph"]
