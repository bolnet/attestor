"""DeriveMemory workflow on the time-skipping test server, mocked activities.

Both lanes are retried per the plan policy; a permanent (non-retryable)
failure in one lane does not stop the other, and the workflow ends
FAILED with the per-lane outcome in the error details.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    DERIVE_STEP_GRAPH,
    DERIVE_STEP_VECTOR,
    EMBED_AND_UPSERT_ACTIVITY,
    GRAPH_EXTRACT_ACTIVITY,
    ActivityPolicy,
    DeriveOutcome,
    DeriveRef,
    DeriveRequest,
    DeriveStepOutcome,
    derive_workflow_id,
)

pytestmark = pytest.mark.integration

FAST = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)


def _cfg() -> DurableCfg:
    return DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")


class _StubDerive:
    def __init__(self, *, vector_fail_times: int = 0, graph_fail_times: int = 0,
                 vector_permanent: bool = False) -> None:
        self.vector_fail_times = vector_fail_times
        self.graph_fail_times = graph_fail_times
        self.vector_permanent = vector_permanent
        self.vector_seen: list[DeriveRef] = []
        self.graph_seen: list[DeriveRef] = []

    @activity.defn(name=EMBED_AND_UPSERT_ACTIVITY)
    async def embed_and_upsert(self, ref: DeriveRef) -> DeriveStepOutcome:
        self.vector_seen.append(ref)
        if self.vector_permanent:
            raise ApplicationError("dimension mismatch", type="DerivePermanent", non_retryable=True)
        if len(self.vector_seen) <= self.vector_fail_times:
            raise ApplicationError("pinecone 503", type="DeriveFailed")
        return DeriveStepOutcome(memory_id=ref.memory_id, step=DERIVE_STEP_VECTOR, ok=True)

    @activity.defn(name=GRAPH_EXTRACT_ACTIVITY)
    async def graph_extract(self, ref: DeriveRef) -> DeriveStepOutcome:
        self.graph_seen.append(ref)
        if len(self.graph_seen) <= self.graph_fail_times:
            raise ApplicationError("neo4j unavailable", type="DeriveFailed")
        return DeriveStepOutcome(
            memory_id=ref.memory_id, step=DERIVE_STEP_GRAPH, ok=True,
            entity_count=2, relation_count=1,
        )


async def _run(env, stub: _StubDerive, request: DeriveRequest, cfg: DurableCfg) -> DeriveOutcome:
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.derive import DeriveMemory

    acts = [stub.embed_and_upsert, stub.graph_extract]
    async with build_worker(env.client, cfg, activities=acts):
        return await env.client.execute_workflow(
            DeriveMemory.run, request,
            id=derive_workflow_id(request.memory.memory_id),
            task_queue=cfg.task_queue,
        )


async def test_derive_runs_vector_then_graph_and_returns_outcome(temporal_env):
    stub = _StubDerive()
    ref = DeriveRef(memory_id="m-ok", namespace="ns")
    out = await _run(temporal_env, stub, DeriveRequest(memory=ref, policy=FAST), _cfg())
    assert out.ok
    assert out.memory_id == "m-ok"
    assert out.vector.step == DERIVE_STEP_VECTOR
    assert out.graph.step == DERIVE_STEP_GRAPH
    assert (out.graph.entity_count, out.graph.relation_count) == (2, 1)
    assert stub.vector_seen == [ref]
    assert stub.graph_seen == [ref]


async def test_derive_retries_each_lane_independently(temporal_env):
    stub = _StubDerive(vector_fail_times=2, graph_fail_times=1)
    ref = DeriveRef(memory_id="m-retry")
    out = await _run(temporal_env, stub, DeriveRequest(memory=ref, policy=FAST), _cfg())
    assert out.ok
    assert len(stub.vector_seen) == 3
    assert len(stub.graph_seen) == 2


async def test_derive_permanent_vector_error_still_runs_graph_then_fails(temporal_env):
    stub = _StubDerive(vector_permanent=True)
    ref = DeriveRef(memory_id="m-perm")
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stub, DeriveRequest(memory=ref, policy=FAST), _cfg())
    assert len(stub.vector_seen) == 1  # non-retryable → no retry
    assert len(stub.graph_seen) == 1   # graph lane still attempted
    cause = exc.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    assert "dimension mismatch" in str(cause)
    outcome = cause.details[0]
    assert outcome["vector"]["ok"] is False
    assert outcome["graph"]["ok"] is True


async def test_derive_workflow_id_dedups(temporal_env):
    from attestor.durable.client import derive_result, start_derive
    from attestor.durable.worker import build_worker

    stub = _StubDerive()
    cfg = _cfg()
    req = DeriveRequest(memory=DeriveRef(memory_id="m-dup"), policy=FAST)
    async with build_worker(
        temporal_env.client, cfg, activities=[stub.embed_and_upsert, stub.graph_extract],
    ):
        wid1 = await start_derive(temporal_env.client, cfg, req)
        wid2 = await start_derive(temporal_env.client, cfg, req)
        assert wid1 == wid2 == "derive-m-dup"
        out = await derive_result(temporal_env.client, wid1)
    assert out.ok
    assert len(stub.vector_seen) == 1
