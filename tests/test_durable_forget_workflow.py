"""ForgetUser saga on the time-skipping test server, stub activities.

Proves the plan's acceptance criteria: the audit row is written BEFORE any
backend delete; a graph outage keeps retrying while doc + vector (+ state)
complete; a permanent audit failure touches no backend; terminal failures
are reported per backend.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    FORGET_DOC_ACTIVITY,
    FORGET_GRAPH_ACTIVITY,
    FORGET_LANE_AUDIT,
    FORGET_LANE_DOC,
    FORGET_LANE_GRAPH,
    FORGET_LANE_STATE,
    FORGET_LANE_VECTOR,
    FORGET_STATE_ACTIVITY,
    FORGET_VECTOR_ACTIVITY,
    LANE_STATUS_OK,
    WRITE_FORGET_AUDIT_ACTIVITY,
    ActivityPolicy,
    ForgetLaneOutcome,
    ForgetRequest,
    ForgetUserOutcome,
    forget_workflow_id,
)

pytestmark = pytest.mark.integration

FAST = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)


def _cfg() -> DurableCfg:
    return DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")


class _Stubs:
    """Scripted backends: ``fail[lane] = n`` transient failures; ``permanent`` lanes."""

    def __init__(self, *, fail: dict[str, int] | None = None,
                 permanent: set[str] | None = None) -> None:
        self.fail = dict(fail or {})
        self.permanent = permanent or set()
        self.order: list[str] = []
        self.attempts: dict[str, int] = {}

    async def _lane(self, lane: str, req: ForgetRequest, **counts: int) -> ForgetLaneOutcome:
        self.attempts[lane] = self.attempts.get(lane, 0) + 1
        if lane in self.permanent:
            raise ApplicationError(f"{lane} permission denied", type="ForgetPermanent",
                                   non_retryable=True)
        if self.fail.get(lane, 0) > 0:
            self.fail[lane] -= 1
            raise ApplicationError(f"{lane} unavailable", type="ForgetFailed")
        await asyncio.sleep(0.01)
        self.order.append(lane)
        return ForgetLaneOutcome(lane=lane, status=LANE_STATUS_OK, **counts)

    @activity.defn(name=WRITE_FORGET_AUDIT_ACTIVITY)
    async def write_forget_audit(self, req: ForgetRequest) -> ForgetLaneOutcome:
        return await self._lane(FORGET_LANE_AUDIT, req)

    @activity.defn(name=FORGET_DOC_ACTIVITY)
    async def forget_doc(self, req: ForgetRequest) -> ForgetLaneOutcome:
        return await self._lane(FORGET_LANE_DOC, req, deleted=3)

    @activity.defn(name=FORGET_VECTOR_ACTIVITY)
    async def forget_vector(self, req: ForgetRequest) -> ForgetLaneOutcome:
        return await self._lane(FORGET_LANE_VECTOR, req, deleted=5)

    @activity.defn(name=FORGET_GRAPH_ACTIVITY)
    async def forget_graph(self, req: ForgetRequest) -> ForgetLaneOutcome:
        return await self._lane(FORGET_LANE_GRAPH, req, deleted=4, edges_deleted=7)

    @activity.defn(name=FORGET_STATE_ACTIVITY)
    async def forget_state(self, req: ForgetRequest) -> ForgetLaneOutcome:
        return await self._lane(FORGET_LANE_STATE, req, deleted=2)

    def activities(self) -> list:
        return [self.write_forget_audit, self.forget_doc, self.forget_vector,
                self.forget_graph, self.forget_state]


def _req(uid: str) -> ForgetRequest:
    return ForgetRequest(user_id=uid, audit_id=uuid.uuid4().hex[:8], initiated_by="ops",
                         policy=FAST)


async def _run(env, stubs: _Stubs, request: ForgetRequest, cfg: DurableCfg) -> ForgetUserOutcome:
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.forget import ForgetUser

    async with build_worker(env.client, cfg, activities=stubs.activities()):
        return await env.client.execute_workflow(
            ForgetUser.run, request,
            id=forget_workflow_id(request.user_id, request.audit_id),
            task_queue=cfg.task_queue,
        )


async def test_audit_row_precedes_every_delete(temporal_env):
    stubs = _Stubs()
    req = _req("u-order")
    out = await _run(temporal_env, stubs, req, _cfg())
    assert out.ok
    assert out.user_id == "u-order"
    assert out.audit_id == req.audit_id
    assert stubs.order[0] == FORGET_LANE_AUDIT
    assert stubs.order[1] == FORGET_LANE_DOC
    assert set(stubs.order[2:]) == {FORGET_LANE_VECTOR, FORGET_LANE_GRAPH, FORGET_LANE_STATE}
    assert (out.document.deleted, out.vector.deleted, out.graph.deleted,
            out.graph.edges_deleted, out.state.deleted) == (3, 5, 4, 7, 2)


async def test_graph_outage_keeps_retrying_while_doc_and_vector_complete(temporal_env):
    stubs = _Stubs(fail={FORGET_LANE_GRAPH: 4})
    out = await _run(temporal_env, stubs, _req("u-graph"), _cfg())
    assert out.ok
    assert stubs.attempts[FORGET_LANE_GRAPH] == 5  # four failures + success
    assert stubs.attempts[FORGET_LANE_DOC] == 1
    assert stubs.attempts[FORGET_LANE_VECTOR] == 1
    # doc + vector landed before the graph lane finally succeeded
    assert stubs.order.index(FORGET_LANE_DOC) < stubs.order.index(FORGET_LANE_GRAPH)
    assert stubs.order.index(FORGET_LANE_VECTOR) < stubs.order.index(FORGET_LANE_GRAPH)


async def test_permanent_audit_failure_touches_no_backend(temporal_env):
    stubs = _Stubs(permanent={FORGET_LANE_AUDIT})
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stubs, _req("u-audit"), _cfg())
    assert stubs.order == []
    assert set(stubs.attempts) == {FORGET_LANE_AUDIT}
    cause = exc.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    outcome = cause.details[0]
    assert outcome["audit"]["status"] == "failed"
    assert outcome["document"]["status"] == "pending"


async def test_terminal_graph_failure_is_reported_per_backend(temporal_env):
    stubs = _Stubs(permanent={FORGET_LANE_GRAPH})
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stubs, _req("u-term"), _cfg())
    assert stubs.attempts[FORGET_LANE_GRAPH] == 1
    assert FORGET_LANE_DOC in stubs.order
    assert FORGET_LANE_VECTOR in stubs.order
    assert FORGET_LANE_STATE in stubs.order
    cause = exc.value.cause
    assert isinstance(cause, ApplicationError)
    outcome = cause.details[0]
    assert outcome["document"]["status"] == "ok"
    assert outcome["document"]["deleted"] == 3
    assert outcome["vector"]["status"] == "ok"
    assert outcome["graph"]["status"] == "failed"
    assert "permission denied" in outcome["graph"]["error"]


async def test_start_forget_is_idempotent_by_id(temporal_env):
    from attestor.durable.client import forget_result, start_forget
    from attestor.durable.worker import build_worker

    stubs = _Stubs()
    cfg = _cfg()
    req = _req("u-dup")
    async with build_worker(temporal_env.client, cfg, activities=stubs.activities()):
        wid1 = await start_forget(temporal_env.client, cfg, req)
        wid2 = await start_forget(temporal_env.client, cfg, req)
        assert wid1 == wid2 == f"forget-u-dup-{req.audit_id}"
        out = await forget_result(temporal_env.client, wid1)
    assert out.ok
    assert stubs.attempts[FORGET_LANE_AUDIT] == 1
