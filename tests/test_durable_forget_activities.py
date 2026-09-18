"""ForgetActivities — one activity per backend, lanes from ``compliance.forget_lanes``.

Fake AgentMemory; no Postgres / Pinecone / Neo4j / Temporal server. Verifies
the outcome mapping and the retryable-vs-permanent error contract.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.compliance import forget_lanes as lanes  # noqa: E402
from attestor.durable.activities.forget import (  # noqa: E402
    ERROR_TYPE_PERMANENT,
    ERROR_TYPE_TRANSIENT,
    ForgetActivities,
)
from attestor.durable.models import (  # noqa: E402
    FORGET_LANE_AUDIT,
    FORGET_LANE_DOC,
    FORGET_LANE_GRAPH,
    FORGET_LANE_STATE,
    FORGET_LANE_VECTOR,
    LANE_STATUS_OK,
    ForgetRequest,
)

pytestmark = pytest.mark.unit

REQ = ForgetRequest(user_id="u-1", audit_id="a-1", initiated_by="ops")


@pytest.fixture
def acts(monkeypatch):
    calls: list[tuple[str, Any]] = []
    behaviour: dict[str, Any] = {}

    def fake(name: str, result: Any):
        def fn(mem, *args, **kwargs):
            calls.append((name, args, kwargs))
            if isinstance(behaviour.get(name), Exception):
                raise behaviour[name]
            return behaviour.get(name, result)
        return fn

    monkeypatch.setattr(lanes, "count_user_rows", fake("count", lanes.UserRowCounts(3, 2)))
    monkeypatch.setattr(lanes, "write_forget_audit", fake("audit", "a-1"))
    monkeypatch.setattr(lanes, "forget_doc", fake("doc", 3))
    monkeypatch.setattr(lanes, "forget_state", fake("state", lanes.LaneDelete(deleted=2)))
    monkeypatch.setattr(lanes, "forget_vector", fake("vector", lanes.LaneDelete(deleted=5)))
    monkeypatch.setattr(lanes, "forget_graph", fake("graph", lanes.GraphDelete(nodes=4, edges=7)))
    built: list[int] = []

    def provider():
        built.append(1)
        return object()

    return {"acts": ForgetActivities(provider), "calls": calls, "behaviour": behaviour,
            "built": built}


def test_activities_are_lazy(acts):
    assert acts["built"] == []


def test_write_forget_audit_counts_first_then_writes_with_request_ids(acts):
    out = acts["acts"].write_forget_audit(REQ)
    assert out.lane == FORGET_LANE_AUDIT
    assert out.status == LANE_STATUS_OK
    names = [c[0] for c in acts["calls"]]
    assert names == ["count", "audit"]
    _, _args, kwargs = acts["calls"][1]
    assert kwargs == {"audit_id": "a-1", "user_id": "u-1", "doc_rows": 3, "state_rows": 2,
                      "initiated_by": "ops"}
    assert acts["built"] == [1]  # built lazily, on the first activity call
    acts["acts"].forget_doc(REQ)
    assert acts["built"] == [1, 1]  # provider asked again; it owns caching


def test_each_lane_maps_counts(acts):
    a = acts["acts"]
    assert a.forget_doc(REQ).deleted == 3
    assert a.forget_doc(REQ).lane == FORGET_LANE_DOC
    vec = a.forget_vector(REQ)
    assert (vec.lane, vec.deleted, vec.skipped) == (FORGET_LANE_VECTOR, 5, False)
    graph = a.forget_graph(REQ)
    assert (graph.lane, graph.deleted, graph.edges_deleted) == (FORGET_LANE_GRAPH, 4, 7)
    state = a.forget_state(REQ)
    assert (state.lane, state.deleted) == (FORGET_LANE_STATE, 2)
    assert all(o.ok for o in (vec, graph, state))


def test_skipped_backend_is_ok_but_flagged(acts):
    acts["behaviour"]["vector"] = lanes.LaneDelete(deleted=0, skipped=True)
    out = acts["acts"].forget_vector(REQ)
    assert out.ok
    assert out.skipped
    assert out.deleted == 0


def test_transient_error_is_retryable(acts):
    acts["behaviour"]["graph"] = RuntimeError("neo4j: connection refused")
    with pytest.raises(ApplicationError) as exc:
        acts["acts"].forget_graph(REQ)
    assert exc.value.type == ERROR_TYPE_TRANSIENT
    assert not exc.value.non_retryable


def test_permission_error_is_permanent(acts):
    acts["behaviour"]["doc"] = RuntimeError("permission denied for table memories")
    with pytest.raises(ApplicationError) as exc:
        acts["acts"].forget_doc(REQ)
    assert exc.value.type == ERROR_TYPE_PERMANENT
    assert exc.value.non_retryable


def test_audit_failure_is_raised_not_swallowed(acts):
    acts["behaviour"]["audit"] = RuntimeError("disk full")
    with pytest.raises(ApplicationError) as exc:
        acts["acts"].write_forget_audit(REQ)
    assert not exc.value.non_retryable


def test_activities_ask_the_provider_on_every_call(acts):
    """The provider owns caching (and rebuilds a memory whose backends failed
    to initialise — live e2e 2026-09-03); an activity must never pin the
    first instance it saw."""
    acts["acts"].forget_vector(REQ)
    acts["acts"].forget_vector(REQ)
    assert len(acts["built"]) == 2
