"""RebuildDerived — sliding-window batch of child DeriveMemory workflows.

The list activity is a stub paging over fake ids; the derive activities
are stubs. Verifies: one child per id with the deterministic
``derive-{id}`` id, window bound respected, failed children counted not
fatal, and continue-as-new carries the cursor + counts.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    DERIVE_STEP_GRAPH,
    DERIVE_STEP_VECTOR,
    EMBED_AND_UPSERT_ACTIVITY,
    GRAPH_EXTRACT_ACTIVITY,
    LIST_MEMORY_IDS_ACTIVITY,
    ActivityPolicy,
    DeriveRef,
    DeriveStepOutcome,
    ListMemoryIdsRequest,
    MemoryIdPage,
    RebuildOutcome,
    RebuildRequest,
)

pytestmark = pytest.mark.integration

FAST = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)


def _cfg() -> DurableCfg:
    return DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")


SOLO_USER = "solo-1"


class _Stubs:
    def __init__(self, ids: list[str], *, permanent_fail: set[str] | None = None) -> None:
        self.ids = ids
        self.permanent_fail = permanent_fail or set()
        self.list_calls: list[ListMemoryIdsRequest] = []
        self.vector_seen: list[str] = []
        self.graph_seen: list[str] = []
        self.refs_seen: list[DeriveRef] = []
        self.inflight = 0
        self.max_inflight = 0

    @activity.defn(name=LIST_MEMORY_IDS_ACTIVITY)
    async def list_memory_ids(self, req: ListMemoryIdsRequest) -> MemoryIdPage:
        self.list_calls.append(req)
        start = self.ids.index(req.after_id) + 1 if req.after_id else 0
        chunk = self.ids[start:start + req.limit]
        nxt = chunk[-1] if chunk and start + req.limit < len(self.ids) else None
        # Like the real seam: an explicit scope is echoed, none resolves SOLO.
        return MemoryIdPage(ids=tuple(chunk), next_after_id=nxt, user_id=req.user_id or SOLO_USER)

    @activity.defn(name=EMBED_AND_UPSERT_ACTIVITY)
    async def embed_and_upsert(self, ref: DeriveRef) -> DeriveStepOutcome:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.refs_seen.append(ref)
        try:
            await asyncio.sleep(0.01)
            self.vector_seen.append(ref.memory_id)
            if ref.memory_id in self.permanent_fail:
                raise ApplicationError(
                    "dimension mismatch", type="DerivePermanent", non_retryable=True,
                )
            return DeriveStepOutcome(memory_id=ref.memory_id, step=DERIVE_STEP_VECTOR, ok=True)
        finally:
            self.inflight -= 1

    @activity.defn(name=GRAPH_EXTRACT_ACTIVITY)
    async def graph_extract(self, ref: DeriveRef) -> DeriveStepOutcome:
        self.graph_seen.append(ref.memory_id)
        return DeriveStepOutcome(memory_id=ref.memory_id, step=DERIVE_STEP_GRAPH, ok=True)

    def activities(self) -> list:
        return [self.list_memory_ids, self.embed_and_upsert, self.graph_extract]


async def _run(env, stubs: _Stubs, request: RebuildRequest, cfg: DurableCfg,
               workflow_id: str = "rebuild-derived-test") -> RebuildOutcome:
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.rebuild import RebuildDerived

    async with build_worker(env.client, cfg, activities=stubs.activities()):
        return await env.client.execute_workflow(
            RebuildDerived.run, request, id=workflow_id, task_queue=cfg.task_queue,
        )


async def test_rebuild_spawns_one_child_per_id_with_deterministic_ids(temporal_env):
    ids = [f"m-{i}" for i in range(7)]
    stubs = _Stubs(ids)
    cfg = _cfg()
    req = RebuildRequest(namespace="ns", page_size=3, window_size=2, policy=FAST)
    out = await _run(temporal_env, stubs, req, cfg)

    assert out.done
    assert out.progress.listed == 7
    assert out.progress.ok == 7
    assert out.progress.failed == 0
    assert out.progress.pages == 3
    assert sorted(stubs.vector_seen) == ids
    assert sorted(stubs.graph_seen) == ids
    assert stubs.max_inflight <= 2  # sliding window honoured
    # Children carry the deterministic derive-{id} workflow id.
    handle = temporal_env.client.get_workflow_handle("derive-m-3")
    desc = await handle.describe()
    assert desc.status is not None
    assert desc.status.name == "COMPLETED"
    # The cursor + filters reach the list activity untouched.
    assert stubs.list_calls[0].namespace == "ns"
    assert stubs.list_calls[0].after_id is None
    assert stubs.list_calls[1].after_id == "m-2"


async def test_rebuild_children_carry_explicit_tenant_scope(temporal_env):
    ids = [f"m-{i}" for i in range(3)]
    stubs = _Stubs(ids)
    req = RebuildRequest(user_id="u-1", page_size=2, window_size=2, policy=FAST)
    out = await _run(temporal_env, stubs, req, _cfg(), workflow_id="rebuild-derived-scoped")
    assert out.done
    assert out.progress.ok == 3
    assert all(r.user_id == "u-1" for r in stubs.refs_seen)
    assert all(c.user_id == "u-1" for c in stubs.list_calls)


async def test_rebuild_children_inherit_scope_resolved_by_list_activity(temporal_env):
    stubs = _Stubs(["m-0", "m-1"])
    req = RebuildRequest(page_size=10, window_size=2, policy=FAST)
    out = await _run(temporal_env, stubs, req, _cfg(), workflow_id="rebuild-derived-solo")
    assert out.done
    assert out.progress.ok == 2
    assert stubs.list_calls[0].user_id is None
    assert {r.user_id for r in stubs.refs_seen} == {SOLO_USER}


async def test_rebuild_counts_permanent_child_failures_without_aborting(temporal_env):
    ids = [f"m-{i}" for i in range(4)]
    stubs = _Stubs(ids, permanent_fail={"m-1", "m-2"})
    req = RebuildRequest(page_size=10, window_size=4, policy=FAST)
    out = await _run(temporal_env, stubs, req, _cfg(), workflow_id="rebuild-derived-fail")
    assert out.done
    assert out.progress.ok == 2
    assert out.progress.failed == 2
    assert sorted(out.failed_ids) == ["m-1", "m-2"]
    assert sorted(stubs.vector_seen) == ids


async def test_rebuild_continues_as_new_across_page_budget(temporal_env):
    ids = [f"m-{i}" for i in range(5)]
    stubs = _Stubs(ids)
    req = RebuildRequest(page_size=2, window_size=2, max_pages_per_run=1, policy=FAST)
    cfg = _cfg()
    out = await _run(temporal_env, stubs, req, cfg, workflow_id="rebuild-derived-can")
    assert out.done
    assert out.progress.listed == 5
    assert out.progress.ok == 5
    assert out.progress.pages == 3
    # One page per run → three runs chained by continue-as-new.
    assert out.progress.runs == 3
    # The final run is the one that completed under the same workflow id.
    desc = await temporal_env.client.get_workflow_handle("rebuild-derived-can").describe()
    assert desc.status is not None
    assert desc.status.name == "COMPLETED"


async def test_rebuild_empty_store_completes_immediately(temporal_env):
    stubs = _Stubs([])
    out = await _run(temporal_env, stubs, RebuildRequest(policy=FAST), _cfg(), "rebuild-derived-0")
    assert out.done
    assert out.progress.listed == 0
    assert out.progress.pages == 1
