"""attestor.durable.worker — worker construction + loud failure modes."""

from __future__ import annotations

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from attestor.config import DurableCfg  # noqa: E402


@pytest.mark.integration
async def test_build_worker_registers_workflows_and_passthrough(temporal_env):
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.consolidate import ConsolidateEpisode

    worker = build_worker(temporal_env.client, DurableCfg(enabled=True), activities=[])
    assert worker.task_queue == "attestor-governance"
    from attestor.durable.workflows.derive import DeriveMemory
    from attestor.durable.workflows.forget import ForgetUser
    from attestor.durable.workflows.rebuild import RebuildDerived
    from attestor.durable.workflows.retention import RetentionSweep
    from attestor.durable.workflows.sessions import SessionSweep

    registered = worker.config()["workflows"]
    assert ConsolidateEpisode in registered
    assert DeriveMemory in registered
    assert RebuildDerived in registered
    assert ForgetUser in registered
    assert RetentionSweep in registered
    assert SessionSweep in registered
    runner = worker.config()["workflow_runner"]
    assert isinstance(runner, SandboxedWorkflowRunner)
    passthrough = runner.restrictions.passthrough_modules
    assert "attestor" in passthrough


@pytest.mark.unit
async def test_run_worker_fails_loudly_when_disabled():
    from attestor.durable.config import DurableDisabledError
    from attestor.durable.worker import run_worker

    with pytest.raises(DurableDisabledError):
        await run_worker(DurableCfg(enabled=False), store_path="/nonexistent")


@pytest.mark.unit
async def test_run_worker_fails_loudly_when_server_unreachable(monkeypatch):
    from attestor.durable import client as durable_client
    from attestor.durable.config import DurableUnavailableError
    from attestor.durable.worker import run_worker

    async def fail_connect(cfg):
        raise DurableUnavailableError(f"unreachable {cfg.address}")

    monkeypatch.setattr(durable_client, "connect", fail_connect)
    with pytest.raises(DurableUnavailableError, match="unreachable"):
        await run_worker(DurableCfg(enabled=True, address="x:1"), store_path="/nonexistent")


@pytest.mark.unit
def test_default_activities_are_lazy(tmp_path):
    """Building the activity list must not open Postgres / construct AgentMemory."""
    from attestor.durable.activities.consolidation import ConsolidationActivities
    from attestor.durable.activities.derive import DeriveActivities
    from attestor.durable.activities.forget import ForgetActivities
    from attestor.durable.activities.retention import RetentionActivities
    from attestor.durable.activities.sessions import SessionActivities
    from attestor.durable.models import (
        APPLY_POLICY_ACTIVITY,
        CONSOLIDATE_EPISODE_ACTIVITY,
        EMBED_AND_UPSERT_ACTIVITY,
        FORGET_DOC_ACTIVITY,
        FORGET_GRAPH_ACTIVITY,
        FORGET_STATE_ACTIVITY,
        FORGET_VECTOR_ACTIVITY,
        GRAPH_EXTRACT_ACTIVITY,
        LIST_DUE_POLICIES_ACTIVITY,
        LIST_MEMORY_IDS_ACTIVITY,
        SWEEP_ARCHIVED_ACTIVITY,
        SWEEP_ENDED_ACTIVITY,
        SWEEP_IDLE_ACTIVITY,
        WRITE_FORGET_AUDIT_ACTIVITY,
    )
    from attestor.durable.worker import default_activities

    acts = default_activities(str(tmp_path / "store"))
    owners = {type(fn.__self__) for fn in acts}
    assert owners == {
        ConsolidationActivities, DeriveActivities, ForgetActivities,
        RetentionActivities, SessionActivities,
    }
    from temporalio.activity import _Definition

    names = {_Definition.from_callable(fn).name for fn in acts}
    assert names == {
        CONSOLIDATE_EPISODE_ACTIVITY, EMBED_AND_UPSERT_ACTIVITY,
        GRAPH_EXTRACT_ACTIVITY, LIST_MEMORY_IDS_ACTIVITY,
        WRITE_FORGET_AUDIT_ACTIVITY, FORGET_DOC_ACTIVITY, FORGET_VECTOR_ACTIVITY,
        FORGET_GRAPH_ACTIVITY, FORGET_STATE_ACTIVITY,
        LIST_DUE_POLICIES_ACTIVITY, APPLY_POLICY_ACTIVITY,
        SWEEP_IDLE_ACTIVITY, SWEEP_ENDED_ACTIVITY, SWEEP_ARCHIVED_ACTIVITY,
    }


# ── the worker's memory is rebuilt while a backend failed to initialise ──

def test_memory_provider_rebuilds_until_backends_initialise(monkeypatch):
    """A memory built during an outage must not be cached forever."""
    from attestor.durable import worker as worker_mod

    built: list[object] = []

    class _FakeMemory:
        def __init__(self, path: str) -> None:
            n = len(built)
            built.append(self)
            # first construction: graph down; second: vector down; third: healthy
            self._graph_init_failed = n == 0
            self._vector_init_failed = n == 1
            self.closed = False

        def close(self) -> None:
            self.closed = True

    import attestor.core as core_pkg
    monkeypatch.setattr(core_pkg, "AgentMemory", _FakeMemory)
    make = worker_mod._memory_provider("/tmp/store")

    first = make()
    second = make()
    third = make()
    fourth = make()

    assert first is not second and second is not third
    assert third is fourth  # healthy instance is cached
    assert first.closed and second.closed
    assert len(built) == 3


def test_memory_provider_builds_once_under_concurrent_first_calls(monkeypatch):
    """Activity threads race on the first call; two parallel AgentMemory
    constructions run schema DDL concurrently and deadlock Postgres
    (live e2e 2026-09-03). The provider must serialise construction."""
    import threading
    import time

    from attestor.durable import worker as worker_mod

    built: list[object] = []

    class _SlowHealthyMemory:
        def __init__(self, path: str) -> None:
            time.sleep(0.05)  # widen the race window
            built.append(self)
            self._graph_init_failed = False
            self._vector_init_failed = False

        def close(self) -> None:
            pass

    import attestor.core as core_pkg
    monkeypatch.setattr(core_pkg, "AgentMemory", _SlowHealthyMemory)
    make = worker_mod._memory_provider("/tmp/store")

    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(make())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1
    assert all(r is built[0] for r in results)
