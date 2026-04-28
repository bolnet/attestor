"""Unit tests for Neo4j namespace tenancy.

The Neo4j Python driver and a live database are NOT required — every test
patches ``neo4j.GraphDatabase.driver`` with a recording stub that captures
the Cypher and parameters passed to ``session.run`` so we can assert:

* upserts MERGE on ``(key, namespace)`` so the same entity name in two
  namespaces produces two distinct nodes
* edges carry ``namespace`` on the relationship as well as on both endpoints
* every read in the recall hot path adds a
  ``coalesce(<n>.namespace, 'default') = $namespace`` clause and binds
  ``namespace`` as a parameter
* the orchestrator forwards ``namespace`` from a tenant-scoped recall
  through the graph step

These checks close the cross-tenant graph affinity leak documented in
``attestor/store/neo4j_backend.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from attestor.retrieval.orchestrator import RetrievalOrchestrator


# ── Driver / session stub ────────────────────────────────────────────────


class _PermissiveRow:
    """Row stand-in that returns a benign default for any column lookup.

    The backend's read methods reach into rows with mixed schemas across
    queries (``key``, ``name``, ``subject``, ``etype``, …); rather than
    hand-shape per-query rows in the stub, we return ``""`` for unknown
    columns and ``"placeholder"`` for the well-known identity columns.
    The assertions only inspect captured Cypher / parameters, never the
    return shape, so this is safe.
    """

    def __getitem__(self, key: str) -> str:
        if key in {"key", "subject", "object", "name"}:
            return "placeholder"
        return ""


class _RecordingResult:
    """Iterable / .single() shim — most calls in the backend ignore the value.

    We feed back a single synthetic row by default so the backend's
    ``get_subgraph`` doesn't bail out on the first empty MATCH and still
    issues its second (edge-fetching) Cypher call.
    """

    def __init__(self, rows: Optional[List[Any]] = None) -> None:
        self._rows = rows if rows is not None else [_PermissiveRow()]

    def __iter__(self):
        return iter(self._rows)

    def single(self) -> Any:
        return self._rows[0] if self._rows else _PermissiveRow()


class _RecordingSession:
    """Captures every ``run(cypher, **params)`` call into a shared list."""

    def __init__(self, sink: List[Tuple[str, Dict[str, Any]]]) -> None:
        self._sink = sink

    def __enter__(self) -> "_RecordingSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> _RecordingResult:
        self._sink.append((cypher, dict(params)))
        return _RecordingResult()


class _RecordingDriver:
    """Stand-in for ``neo4j.Driver`` — opens recording sessions on demand."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def session(self, database: Optional[str] = None) -> _RecordingSession:
        return _RecordingSession(self.calls)

    def verify_connectivity(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def neo4j_backend(monkeypatch):
    """Build a Neo4jBackend bound to a recording driver — no live DB needed."""
    from attestor.store import neo4j_backend as backend_mod

    driver = _RecordingDriver()
    monkeypatch.setattr(
        backend_mod.GraphDatabase, "driver",
        lambda *args, **kwargs: driver,
    )

    config = {
        "neo4j_uri": "bolt://stub:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "stub",
    }
    instance = backend_mod.Neo4jBackend(config)
    # Reset call log so schema-init noise doesn't pollute assertions
    driver.calls.clear()
    return instance, driver


def _last_call(driver: _RecordingDriver) -> Tuple[str, Dict[str, Any]]:
    assert driver.calls, "expected at least one Cypher call"
    return driver.calls[-1]


# ── add_entity ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_add_entity_merges_on_name_and_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.add_entity("Caroline", entity_type="person", namespace="acme")

    cypher, params = _last_call(driver)
    # MERGE must scope by BOTH key AND namespace — that is the whole
    # contract of this change.
    assert "MERGE (e:Entity {key: $key, namespace: $namespace})" in cypher
    assert params["namespace"] == "acme"
    # ``key`` is the normalized form of the display name
    assert params["key"] == "caroline"


@pytest.mark.unit
def test_add_entity_defaults_to_default_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.add_entity("Caroline", entity_type="person")  # no namespace kwarg

    _, params = _last_call(driver)
    # Backend-level fallback mirrors the read-side coalesce so legacy
    # callers that haven't been plumbed through with namespace still land
    # in a single, well-known tenant.
    assert params["namespace"] == "default"


# ── add_relation ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_add_relation_tags_endpoints_and_edge_with_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.add_relation(
        "Caroline", "Acme",
        relation_type="works_at",
        metadata={"event_date": "2026-04-28"},
        namespace="acme",
    )

    cypher, params = _last_call(driver)
    # Both endpoint MERGEs scope by (key, namespace) and the edge itself
    # carries namespace so a future query can gate on r.namespace without
    # traversing back to a node.
    assert "MERGE (a:Entity {key: $from_key, namespace: $namespace})" in cypher
    assert "MERGE (b:Entity {key: $to_key, namespace: $namespace})" in cypher
    assert "namespace: $namespace" in cypher  # edge-level
    assert params["namespace"] == "acme"
    # Metadata is also stamped — useful for graph_stats / debugging queries
    assert params["meta"]["namespace"] == "acme"


# ── Read filters ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_related_filters_by_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.get_related("Caroline", depth=2, namespace="acme")

    cypher, params = _last_call(driver)
    # The coalesce clause is the back-compat bridge — pre-namespace nodes
    # still recall under namespace="default" without a migration script.
    assert "coalesce(start.namespace, 'default') = $namespace" in cypher
    assert "coalesce(other.namespace, 'default') = $namespace" in cypher
    assert params["namespace"] == "acme"


@pytest.mark.unit
def test_get_subgraph_filters_by_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.get_subgraph("Caroline", depth=2, namespace="acme")

    # Two queries are issued (nodes + edges); both must filter.
    nodes_cypher, nodes_params = driver.calls[0]
    edges_cypher, edges_params = driver.calls[1]
    assert "coalesce(start.namespace, 'default') = $namespace" in nodes_cypher
    assert "coalesce(a.namespace, 'default') = $namespace" in edges_cypher
    assert "coalesce(b.namespace, 'default') = $namespace" in edges_cypher
    assert nodes_params["namespace"] == "acme"
    assert edges_params["namespace"] == "acme"


@pytest.mark.unit
def test_get_edges_filters_by_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.get_edges("Caroline", namespace="acme")

    cypher, params = _last_call(driver)
    assert "coalesce(a.namespace, 'default') = $namespace" in cypher
    assert "coalesce(b.namespace, 'default') = $namespace" in cypher
    assert params["namespace"] == "acme"


@pytest.mark.unit
def test_read_defaults_to_default_namespace(neo4j_backend):
    backend, driver = neo4j_backend
    backend.get_related("Caroline", depth=1)

    _, params = _last_call(driver)
    # Symmetric to writes — None at the API surface means "default" tenant.
    assert params["namespace"] == "default"


# ── Schema init ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_init_schema_uses_composite_constraint(monkeypatch):
    """Fresh backends drop the legacy key-only constraint and create the composite one."""
    from attestor.store import neo4j_backend as backend_mod

    driver = _RecordingDriver()
    monkeypatch.setattr(
        backend_mod.GraphDatabase, "driver",
        lambda *args, **kwargs: driver,
    )
    backend_mod.Neo4jBackend({
        "neo4j_uri": "bolt://stub:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "stub",
    })

    cyphers = [c for c, _ in driver.calls]
    assert any("DROP CONSTRAINT entity_key_unique" in c for c in cyphers), (
        "must drop the legacy single-property constraint to make room for "
        "the composite (key, namespace) one"
    )
    assert any(
        "CREATE CONSTRAINT entity_key_namespace_unique" in c
        and "REQUIRE (e.key, e.namespace) IS UNIQUE" in c
        for c in cyphers
    )


# ── Orchestrator integration ─────────────────────────────────────────────


class _FakeGraph:
    """Captures namespace forwarded by the orchestrator's graph step."""

    def __init__(self) -> None:
        self.related_calls: List[Dict[str, Any]] = []
        self.edges_calls: List[Dict[str, Any]] = []

    def get_related(self, entity: str, depth: int = 2,
                    namespace: Optional[str] = None) -> List[str]:
        self.related_calls.append(
            {"entity": entity, "depth": depth, "namespace": namespace}
        )
        return []

    def get_edges(self, entity: str,
                  namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        self.edges_calls.append({"entity": entity, "namespace": namespace})
        return []


class _NoopStore:
    """Just enough of DocumentStore to let recall() short-circuit cleanly."""

    def get(self, memory_id: str):
        return None


@pytest.mark.unit
def test_orchestrator_forwards_namespace_to_graph_step():
    graph = _FakeGraph()
    orch = RetrievalOrchestrator(
        store=_NoopStore(),
        vector_store=None,  # bypass the vector lane entirely
        graph=graph,
    )
    # Capitalised token so _question_entities picks it up; the orchestrator
    # then calls graph.get_related / get_edges in the narrow + triple-injection
    # steps.
    orch.recall("Where does Caroline work?", namespace="acme")

    assert graph.related_calls, "graph affinity step did not fire"
    for call in graph.related_calls:
        assert call["namespace"] == "acme", (
            "orchestrator must forward the active tenant's namespace "
            "into the BFS — otherwise tenant A pulls bonuses from tenant B"
        )
    for call in graph.edges_calls:
        assert call["namespace"] == "acme"
