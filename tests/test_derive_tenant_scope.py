"""Derive (repair / rebuild) always runs as the memory's OWNER.

Phase-2 review finding: the durable path re-read and re-wrote derived
state without ever scoping the Postgres RLS session user. These tests
pin the invariant ``consolidation/consolidator.py`` already follows for
episodes — set ``attestor.current_user_id`` to the row's owner before
every re-read / write, apply the ``visibility='private'`` requester
guard, and never page ids across tenants without an explicit scope.

Fake stores only; no Postgres / Pinecone / Neo4j / Temporal.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from attestor.core.derive_service import (
    DerivedIdPage,
    MemoryNotFoundError,
    RebuildScopeError,
    RebuildUnsupportedError,
)
from attestor.models import Memory
from tests._lockprobe import held_by_caller
from tests.test_add_repair_scheduling import (
    _build_mem,
    _FakeDocumentStore,
    _FakeGraphStore,
    _FakeVectorStore,
)

OWNER = "u-1"
OTHER = "u-2"
AGENT = "planner"


class _RlsStore(_FakeDocumentStore):
    """v4-shaped document store: RLS setter + owner-filtered id listing."""

    _v4 = True

    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.log = log
        self.list_calls: list[dict[str, Any]] = []

    def _set_rls_user(self, user_id: str | None) -> None:
        self.log.append(("rls", user_id))

    def get(self, memory_id, requester_agent_id=None):  # noqa: ANN001
        self.log.append(("get", memory_id, requester_agent_id))
        return self.memories.get(memory_id)

    def list_memory_ids(self, *, since, namespace, user_id, after_id, limit):  # noqa: ANN001
        self.list_calls.append(dict(
            since=since, namespace=namespace, user_id=user_id, after_id=after_id, limit=limit,
        ))
        ids = sorted(
            m.id for m in self.memories.values() if user_id is None or m.user_id == user_id
        )
        return ids, None


class _LoggingVector(_FakeVectorStore):
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.log = log

    def add(self, memory_id, content, namespace="default"):  # noqa: ANN001
        self.log.append(("vector", memory_id))
        super().add(memory_id, content, namespace)


class _LoggingGraph(_FakeGraphStore):
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.log = log

    def add_entity(  # noqa: ANN001
        self, name, entity_type="general", attributes=None, namespace="default",
    ):
        self.log.append(("graph", name))
        super().add_entity(name, entity_type, attributes, namespace)


def _owned(
    memory_id: str = "m-1", user_id: str | None = OWNER, agent_id: str | None = AGENT,
) -> Memory:
    return Memory(
        id=memory_id, content="Alice uses Postgres.", entity="Alice", namespace="ns",
        user_id=user_id, agent_id=agent_id, visibility="private",
    )


@pytest.fixture
def rls_mem():
    log: list[tuple[Any, ...]] = []
    mem, store, _vec, _graph = _build_mem(
        store=_RlsStore(log), vec=_LoggingVector(log), graph=_LoggingGraph(log),
    )
    mem._default_user = None  # multi-tenant: no implicit SOLO scope
    try:
        yield mem, store, log
    finally:
        mem.close()


# ── derive_vector / derive_graph run as the row owner ───────────────────

@pytest.mark.unit
def test_derive_vector_sets_rls_to_owner_before_read_and_before_write(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned())
    log.clear()
    out = mem.derive_vector("m-1", user_id=OWNER, agent_id=AGENT)
    assert out.memory_id == "m-1"
    # scope → guarded re-read → scope reset to the ROW's owner → derived write
    assert log == [("rls", OWNER), ("get", "m-1", AGENT), ("rls", OWNER), ("vector", "m-1")]


@pytest.mark.unit
def test_derive_graph_sets_rls_to_owner_before_write(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned())
    log.clear()
    mem.derive_graph("m-1", user_id=OWNER, agent_id=AGENT)
    first_graph = next(i for i, e in enumerate(log) if e[0] == "graph")
    rls_before = [e for e in log[:first_graph] if e[0] == "rls"]
    assert rls_before
    assert rls_before[-1] == ("rls", OWNER)
    assert ("get", "m-1", AGENT) in log[:first_graph]


@pytest.mark.unit
def test_derive_refuses_row_owned_by_another_user_even_when_readable(rls_mem):
    """A BYPASSRLS worker role can read any row; the application check
    still refuses to re-derive across tenants — and never switches the
    session to the other owner."""
    mem, store, log = rls_mem
    store.insert(_owned(user_id=OTHER))
    with pytest.raises(MemoryNotFoundError):
        mem.derive_vector("m-1", user_id=OWNER)
    assert ("rls", OTHER) not in log
    assert not [e for e in log if e[0] == "vector"]


@pytest.mark.unit
def test_derive_without_scope_on_multi_tenant_store_is_refused(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned())
    with pytest.raises(RebuildScopeError):
        mem.derive_vector("m-1")
    assert not [e for e in log if e[0] in ("get", "vector")]


@pytest.mark.unit
def test_derive_without_scope_uses_solo_default_user(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned(user_id="solo-1", agent_id=None))
    mem._default_user = SimpleNamespace(id="solo-1")
    log.clear()
    mem.derive_vector("m-1")
    assert log[0] == ("rls", "solo-1")
    assert log[1] == ("get", "m-1", None)  # no agent → no requester guard


@pytest.mark.unit
def test_derive_on_v3_store_stays_unscoped():
    """No user_id column, no RLS policies → nothing to scope (today's path)."""
    calls: list[str | None] = []
    store = _FakeDocumentStore()
    store._set_rls_user = lambda uid: calls.append(uid)  # type: ignore[attr-defined]
    mem, store, vec, _g = _build_mem(store=store)
    try:
        m = mem.add("User prefers Python.", namespace="ns")
        mem.derive_vector(m.id)
    finally:
        mem.close()
    assert calls == []
    assert vec.adds[-1][0] == m.id


# ── list_derivable_ids: explicit tenant scope for RebuildDerived ────────

@pytest.mark.unit
def test_list_derivable_ids_scopes_rls_and_filters_by_owner(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned("m-1", user_id=OWNER))
    store.insert(_owned("m-2", user_id=OTHER))
    log.clear()
    page = mem.list_derivable_ids(
        user_id=OWNER, since=None, namespace="ns", after_id=None, limit=10,
    )
    assert isinstance(page, DerivedIdPage)
    assert page == DerivedIdPage(ids=("m-1",), next_after_id=None, user_id=OWNER)
    assert log[0] == ("rls", OWNER)
    assert store.list_calls[0]["user_id"] == OWNER
    assert store.list_calls[0]["namespace"] == "ns"


@pytest.mark.unit
def test_list_derivable_ids_multi_tenant_without_user_is_refused(rls_mem):
    mem, store, _log = rls_mem
    store.insert(_owned())
    with pytest.raises(RebuildScopeError, match="--user"):
        mem.list_derivable_ids(user_id=None, since=None, namespace=None, after_id=None, limit=10)
    assert store.list_calls == []


@pytest.mark.unit
def test_list_derivable_ids_solo_resolves_default_user(rls_mem):
    mem, store, log = rls_mem
    store.insert(_owned(user_id="solo-1"))
    mem._default_user = SimpleNamespace(id="solo-1")
    page = mem.list_derivable_ids(user_id=None, since=None, namespace=None, after_id=None, limit=10)
    assert page.user_id == "solo-1"
    assert page.ids == ("m-1",)
    assert ("rls", "solo-1") in log


@pytest.mark.unit
def test_list_derivable_ids_v3_store_is_unscoped():
    store = _FakeDocumentStore()
    seen: dict[str, Any] = {}

    def lister(*, since, namespace, user_id, after_id, limit):  # noqa: ANN001
        seen.update(user_id=user_id, limit=limit)
        return ["a", "b"], "b"

    store.list_memory_ids = lister  # type: ignore[attr-defined]
    mem, _s, _v, _g = _build_mem(store=store)
    try:
        page = mem.list_derivable_ids(
            user_id=None, since=None, namespace=None, after_id=None, limit=2,
        )
    finally:
        mem.close()
    assert page == DerivedIdPage(ids=("a", "b"), next_after_id="b", user_id=None)
    assert seen == {"user_id": None, "limit": 2}


@pytest.mark.unit
def test_list_derivable_ids_without_store_support_is_unsupported():
    mem, _s, _v, _g = _build_mem()
    try:
        with pytest.raises(RebuildUnsupportedError):
            mem.list_derivable_ids(user_id=None, since=None, namespace=None, after_id=None, limit=1)
    finally:
        mem.close()


# ── the RLS scope-set → read → ownership check → write is ONE locked unit ──
#
# Review finding: ``set_config(..., false)`` is SESSION state on a
# connection the worker shares across activity threads. Another tenant's
# activity must not be able to flip the session user between the scope
# set and the derived write, so the store's connection lock is held for
# the whole sequence (re-entrant: the inner ``get`` locks again).

def _locked_fakes(lock: threading.RLock, held: list[tuple[str, bool]]):
    log: list[tuple[Any, ...]] = []

    class _LockedStore(_RlsStore):
        def _get_conn_lock(self):  # noqa: ANN202
            return lock

        def _set_rls_user(self, user_id):  # noqa: ANN001
            held.append(("rls", held_by_caller(lock)))
            super()._set_rls_user(user_id)

        def get(self, memory_id, requester_agent_id=None):  # noqa: ANN001
            held.append(("get", held_by_caller(lock)))
            with lock:  # a real backend's get() re-locks via _execute
                return super().get(memory_id, requester_agent_id)

        def list_memory_ids(self, **kw):  # noqa: ANN003
            held.append(("list", held_by_caller(lock)))
            return super().list_memory_ids(**kw)

    class _LockedVector(_LoggingVector):
        def add(self, memory_id, content, namespace="default"):  # noqa: ANN001
            held.append(("vector", held_by_caller(lock)))
            super().add(memory_id, content, namespace)

    class _LockedGraph(_LoggingGraph):
        def add_entity(  # noqa: ANN001
            self, name, entity_type="general", attributes=None, namespace="default",
        ):
            held.append(("graph", held_by_caller(lock)))
            super().add_entity(name, entity_type, attributes, namespace)

    return _build_mem(store=_LockedStore(log), vec=_LockedVector(log), graph=_LockedGraph(log))


@pytest.mark.unit
def test_derive_holds_the_connection_lock_from_scope_set_through_write():
    lock = threading.RLock()
    held: list[tuple[str, bool]] = []
    mem, store, _v, _g = _locked_fakes(lock, held)
    mem._default_user = None
    store.insert(_owned())
    try:
        mem.derive_vector("m-1", user_id=OWNER, agent_id=AGENT)
        mem.derive_graph("m-1", user_id=OWNER, agent_id=AGENT)
    finally:
        mem.close()
    assert {step for step, _ in held} >= {"rls", "get", "vector", "graph"}
    assert all(ok for _, ok in held), f"steps outside the lock: {held}"
    assert not held_by_caller(lock), "lock leaked after derive"


@pytest.mark.unit
def test_derive_releases_the_lock_when_the_row_is_foreign():
    lock = threading.RLock()
    held: list[tuple[str, bool]] = []
    mem, store, _v, _g = _locked_fakes(lock, held)
    mem._default_user = None
    store.insert(_owned(user_id=OTHER))
    try:
        with pytest.raises(MemoryNotFoundError):
            mem.derive_vector("m-1", user_id=OWNER, agent_id=AGENT)
    finally:
        mem.close()
    assert not held_by_caller(lock)


@pytest.mark.unit
def test_list_derivable_ids_holds_the_lock_across_scope_and_listing():
    lock = threading.RLock()
    held: list[tuple[str, bool]] = []
    mem, store, _v, _g = _locked_fakes(lock, held)
    mem._default_user = None
    store.insert(_owned())
    try:
        mem.list_derivable_ids(
            user_id=OWNER, since=None, namespace=None, after_id=None, limit=10,
        )
    finally:
        mem.close()
    assert [step for step, _ in held] == ["rls", "list"]
    assert all(ok for _, ok in held), held
    assert not held_by_caller(lock)
