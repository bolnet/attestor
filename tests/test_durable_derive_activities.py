"""DeriveActivities — embed_and_upsert / graph_extract / list_memory_ids.

Pure unit tests: a fake AgentMemory stands in for the I/O side. The
activities are plain callables outside a Temporal context.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.core.derive_service import (  # noqa: E402
    DerivedGraph,
    DerivedIdPage,
    DerivedStoreUnavailableError,
    DerivedVector,
    MemoryNotFoundError,
    RebuildScopeError,
    RebuildUnsupportedError,
)
from attestor.durable.activities.derive import (  # noqa: E402
    ERROR_TYPE_MISSING,
    ERROR_TYPE_PERMANENT,
    ERROR_TYPE_TRANSIENT,
    DeriveActivities,
    is_permanent_derive_error,
)
from attestor.durable.models import (  # noqa: E402
    DERIVE_STEP_GRAPH,
    DERIVE_STEP_VECTOR,
    EMBED_AND_UPSERT_ACTIVITY,
    GRAPH_EXTRACT_ACTIVITY,
    LIST_MEMORY_IDS_ACTIVITY,
    DeriveRef,
    ListMemoryIdsRequest,
    MemoryIdPage,
)


class _FakeMemory:
    """Stands in for ``AgentMemory``'s derive seam (``attestor.core.derive_service``)."""

    def __init__(self, *, vector_exc: Exception | None = None,
                 graph_exc: Exception | None = None,
                 pages: dict[str | None, DerivedIdPage] | None = None,
                 list_exc: Exception | None = None) -> None:
        self.vector_exc = vector_exc
        self.graph_exc = graph_exc
        self.list_exc = list_exc
        self.pages = pages or {}
        self.vector_calls: list[str] = []
        self.graph_calls: list[str] = []
        self.scopes: list[tuple[str | None, str | None]] = []
        self.list_calls: list[dict] = []

    def derive_vector(self, memory_id: str, *, user_id=None, agent_id=None) -> DerivedVector:  # noqa: ANN001
        self.vector_calls.append(memory_id)
        self.scopes.append((user_id, agent_id))
        if self.vector_exc:
            raise self.vector_exc
        return DerivedVector(memory_id=memory_id, namespace="ns", payload_len=12)

    def derive_graph(self, memory_id: str, *, user_id=None, agent_id=None) -> DerivedGraph:  # noqa: ANN001
        self.graph_calls.append(memory_id)
        self.scopes.append((user_id, agent_id))
        if self.graph_exc:
            raise self.graph_exc
        return DerivedGraph(memory_id=memory_id, namespace="ns", entity_count=2, relation_count=1)

    def list_derivable_ids(self, *, user_id, since, namespace, after_id, limit) -> DerivedIdPage:  # noqa: ANN001
        self.list_calls.append(dict(
            user_id=user_id, since=since, namespace=namespace, after_id=after_id, limit=limit,
        ))
        if self.list_exc:
            raise self.list_exc
        return self.pages.get(after_id, DerivedIdPage(ids=(), next_after_id=None, user_id=user_id))


def _acts(mem: _FakeMemory) -> DeriveActivities:
    built: list[int] = []

    def provider():
        built.append(1)
        return mem

    acts = DeriveActivities(provider)
    assert built == []  # lazy — constructing the activity set opens nothing
    return acts


@pytest.mark.unit
def test_activity_names_match_models_constants():
    from temporalio.activity import _Definition

    acts = _acts(_FakeMemory())
    names = {
        _Definition.from_callable(fn).name
        for fn in (acts.embed_and_upsert, acts.graph_extract, acts.list_memory_ids)
    }
    assert names == {EMBED_AND_UPSERT_ACTIVITY, GRAPH_EXTRACT_ACTIVITY, LIST_MEMORY_IDS_ACTIVITY}


@pytest.mark.unit
def test_embed_and_upsert_re_reads_by_id_and_reports_step():
    mem = _FakeMemory()
    out = _acts(mem).embed_and_upsert(DeriveRef(memory_id="m-1", namespace="ns"))
    assert mem.vector_calls == ["m-1"]
    assert out.ok
    assert out.step == DERIVE_STEP_VECTOR
    assert out.memory_id == "m-1"


@pytest.mark.unit
def test_activities_pass_owner_and_agent_scope_to_the_derive_seam():
    mem = _FakeMemory()
    ref = DeriveRef(memory_id="m-1", namespace="ns", user_id="u-1", agent_id="planner")
    _acts(mem).embed_and_upsert(ref)
    _acts(mem).graph_extract(ref)
    assert mem.scopes == [("u-1", "planner"), ("u-1", "planner")]


@pytest.mark.unit
def test_missing_tenant_scope_is_non_retryable():
    mem = _FakeMemory(vector_exc=RebuildScopeError("needs --user"))
    with pytest.raises(ApplicationError) as info:
        _acts(mem).embed_and_upsert(DeriveRef(memory_id="m-1"))
    assert info.value.non_retryable
    assert info.value.type == ERROR_TYPE_PERMANENT


@pytest.mark.unit
def test_graph_extract_reports_counts():
    mem = _FakeMemory()
    out = _acts(mem).graph_extract(DeriveRef(memory_id="m-2"))
    assert mem.graph_calls == ["m-2"]
    assert out.ok
    assert out.step == DERIVE_STEP_GRAPH
    assert (out.entity_count, out.relation_count) == (2, 1)


@pytest.mark.unit
def test_missing_memory_is_non_retryable():
    mem = _FakeMemory(vector_exc=MemoryNotFoundError("memory m-9 not found"))
    with pytest.raises(ApplicationError) as exc:
        _acts(mem).embed_and_upsert(DeriveRef(memory_id="m-9"))
    assert exc.value.non_retryable
    assert exc.value.type == ERROR_TYPE_MISSING


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Vector dimension 1536 does not match the dimension of the index 1024"),
        RuntimeError("HTTP 401 Unauthorized: invalid api key"),
        RuntimeError("AuthError: The client is unauthorized due to authentication failure."),
        PermissionError("Forbidden (403)"),
        # psycopg2 InvalidTextRepresentation: a malformed --user against the uuid column
        RuntimeError('invalid input syntax for type uuid: "not-a-user"'),
    ],
)
def test_dimension_and_auth_errors_are_non_retryable(exc: Exception):
    assert is_permanent_derive_error(exc)
    mem = _FakeMemory(vector_exc=exc)
    with pytest.raises(ApplicationError) as info:
        _acts(mem).embed_and_upsert(DeriveRef(memory_id="m-1"))
    assert info.value.non_retryable
    assert info.value.type == ERROR_TYPE_PERMANENT


@pytest.mark.unit
def test_embedder_dim_mismatch_error_type_is_non_retryable():
    from attestor.store.embedder_dim_check import EmbedderDimMismatchError

    assert is_permanent_derive_error(EmbedderDimMismatchError("schema vector(1024) vs 1536"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection refused"),
        RuntimeError("503 Service Unavailable"),
        DerivedStoreUnavailableError("vector store not initialised"),
        TimeoutError("read timed out"),
        # A uuid containing '401' must not be mistaken for an HTTP status.
        RuntimeError("upsert failed for 6f4013aa-0000-4000-8000-000000000000: reset by peer"),
    ],
)
def test_transient_errors_are_retryable(exc: Exception):
    assert not is_permanent_derive_error(exc)
    mem = _FakeMemory(graph_exc=exc)
    with pytest.raises(ApplicationError) as info:
        _acts(mem).graph_extract(DeriveRef(memory_id="m-1"))
    assert not info.value.non_retryable
    assert info.value.type == ERROR_TYPE_TRANSIENT


@pytest.mark.unit
def test_list_memory_ids_pages_through_scoped_seam():
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mem = _FakeMemory(pages={
        None: DerivedIdPage(ids=("a", "b"), next_after_id="b", user_id="u-1"),
        "b": DerivedIdPage(ids=("c",), next_after_id=None, user_id="u-1"),
    })
    acts = _acts(mem)
    first = acts.list_memory_ids(
        ListMemoryIdsRequest(since=since, namespace="ns", user_id="u-1", limit=2),
    )
    assert first == MemoryIdPage(ids=("a", "b"), next_after_id="b", user_id="u-1")
    second = acts.list_memory_ids(
        ListMemoryIdsRequest(since=since, namespace="ns", user_id="u-1", after_id="b", limit=2),
    )
    assert second == MemoryIdPage(ids=("c",), next_after_id=None, user_id="u-1")
    assert mem.list_calls[0] == dict(
        user_id="u-1", since=since, namespace="ns", after_id=None, limit=2,
    )
    assert mem.list_calls[1]["after_id"] == "b"


@pytest.mark.unit
def test_list_memory_ids_reports_resolved_solo_scope():
    """No explicit user → the seam resolves the SOLO default user and the
    page carries it, so every child DeriveMemory gets an explicit owner."""
    mem = _FakeMemory(pages={None: DerivedIdPage(ids=("a",), next_after_id=None, user_id="solo-1")})
    page = _acts(mem).list_memory_ids(ListMemoryIdsRequest())
    assert page.user_id == "solo-1"
    assert mem.list_calls[0]["user_id"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        RebuildUnsupportedError("store cannot list ids"),
        RebuildScopeError("needs --user"),
        RuntimeError('invalid input syntax for type uuid: "not-a-user"'),
    ],
)
def test_list_memory_ids_permanent_errors_are_non_retryable(exc: Exception):
    mem = _FakeMemory(list_exc=exc)
    with pytest.raises(ApplicationError) as info:
        _acts(mem).list_memory_ids(ListMemoryIdsRequest())
    assert info.value.non_retryable
    assert info.value.type == ERROR_TYPE_PERMANENT


@pytest.mark.unit
def test_list_memory_ids_transient_errors_are_retryable():
    mem = _FakeMemory(list_exc=ConnectionError("postgres connection reset"))
    with pytest.raises(ApplicationError) as info:
        _acts(mem).list_memory_ids(ListMemoryIdsRequest())
    assert not info.value.non_retryable
    assert info.value.type == ERROR_TYPE_TRANSIENT
