"""ConsolidateEpisode workflow on the time-skipping test server, mocked activities.

Uses ``build_worker`` from ``attestor.durable.worker`` so the sandbox
pass-through configuration is exercised (the workflow file imports
``attestor.durable.models``). Activities are stubs — no Postgres/LLM.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ActivityError, ApplicationError  # noqa: E402

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    CONSOLIDATE_EPISODE_ACTIVITY,
    ActivityPolicy,
    ConsolidateRequest,
    ConsolidationOutcome,
    EpisodeRef,
    consolidate_workflow_id,
)

pytestmark = pytest.mark.integration


def _ref(eid: str) -> EpisodeRef:
    ts = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    return EpisodeRef(
        episode_id=eid, user_id="u-1", session_id="s-1", thread_id="t-1",
        user_ts=ts, assistant_ts=ts, project_id="p-1", agent_id="planner",
    )


def _cfg() -> DurableCfg:
    return DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")


class _StubActivities:
    """Scripted stand-in for ConsolidationActivities."""

    def __init__(self, fail_times: int = 0, *, permanent: bool = False) -> None:
        self.fail_times = fail_times
        self.permanent = permanent
        self.seen: list[EpisodeRef] = []

    @activity.defn(name=CONSOLIDATE_EPISODE_ACTIVITY)
    async def consolidate_episode(self, ref: EpisodeRef) -> ConsolidationOutcome:
        self.seen.append(ref)
        if self.permanent:
            raise ApplicationError("rls: denied", type="ConsolidationPermanent", non_retryable=True)
        if len(self.seen) <= self.fail_times:
            raise ApplicationError("transient llm 503", type="ConsolidationFailed")
        return ConsolidationOutcome(
            episode_id=ref.episode_id, ok=True, written_memory_ids=("m-1", "m-2"),
            user_fact_count=1, agent_fact_count=1,
        )


async def _run(env, stub, request: ConsolidateRequest, cfg: DurableCfg):
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.consolidate import ConsolidateEpisode

    async with build_worker(env.client, cfg, activities=[stub.consolidate_episode]):
        return await env.client.execute_workflow(
            ConsolidateEpisode.run, request,
            id=consolidate_workflow_id(request.episode.episode_id),
            task_queue=cfg.task_queue,
        )


async def test_workflow_round_trips_models_and_returns_outcome(temporal_env):
    stub = _StubActivities()
    ref = _ref("ep-ok")
    out = await _run(temporal_env, stub, ConsolidateRequest(episode=ref), _cfg())
    assert out == ConsolidationOutcome(
        episode_id="ep-ok", ok=True, written_memory_ids=("m-1", "m-2"),
        user_fact_count=1, agent_fact_count=1,
    )
    assert stub.seen == [ref]  # datetime + optional fields survive the boundary


async def test_workflow_retries_transient_activity_failures(temporal_env):
    stub = _StubActivities(fail_times=2)
    policy = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)
    out = await _run(
        temporal_env, stub, ConsolidateRequest(episode=_ref("ep-retry"), policy=policy), _cfg(),
    )
    assert out.ok
    assert len(stub.seen) == 3  # two failures + one success


async def test_workflow_fails_fast_on_non_retryable_error(temporal_env):
    stub = _StubActivities(permanent=True)
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stub, ConsolidateRequest(episode=_ref("ep-perm")), _cfg())
    assert len(stub.seen) == 1
    cause = exc.value.cause
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable


async def test_workflow_honours_maximum_attempts(temporal_env):
    stub = _StubActivities(fail_times=99)
    policy = ActivityPolicy(
        initial_interval_seconds=0.01, maximum_interval_seconds=0.02, maximum_attempts=3,
    )
    with pytest.raises(WorkflowFailureError):
        await _run(
            temporal_env, stub, ConsolidateRequest(episode=_ref("ep-cap"), policy=policy), _cfg(),
        )
    assert len(stub.seen) == 3


async def test_workflow_id_dedups_duplicate_starts(temporal_env):
    """Same episode dispatched twice → one execution (id-based idempotency)."""
    from attestor.durable.client import consolidation_result, start_consolidation
    from attestor.durable.worker import build_worker

    stub = _StubActivities()
    cfg = _cfg()
    req = ConsolidateRequest(episode=_ref("ep-dup"))
    async with build_worker(temporal_env.client, cfg, activities=[stub.consolidate_episode]):
        wid1 = await start_consolidation(temporal_env.client, cfg, req)
        wid2 = await start_consolidation(temporal_env.client, cfg, req)
        assert wid1 == wid2 == "consolidate-ep-dup"
        out = await consolidation_result(temporal_env.client, wid1)
    assert out.ok
    assert len(stub.seen) == 1
