"""attestor.durable.schedules — declarative specs + idempotent apply.

Pure tests use a fake client; the live tests run against the local dev
server (``WorkflowEnvironment.start_local``) because the time-skipping test
server does not implement the Schedule service. "Retention fires at 03:00"
is proven by backfilling one day: exactly one action fires, stamped
``retention-sweep-<day>T03:00:00Z``, and the stub worker receives the
declared ``RetentionSweepRequest``.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import ScheduleAlreadyRunningError  # noqa: E402

from attestor.config import DurableCfg, DurableSchedulesCfg  # noqa: E402
from attestor.durable import schedules  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    APPLY_POLICY_ACTIVITY,
    LIST_DUE_POLICIES_ACTIVITY,
    RETENTION_SCHEDULE_ID,
    RETENTION_WORKFLOW,
    RETENTION_WORKFLOW_ID_PREFIX,
    SESSION_SCHEDULE_ID,
    SESSION_WORKFLOW,
    SESSION_WORKFLOW_ID_PREFIX,
    DuePolicies,
    RetentionSweepRequest,
    SessionSweepRequest,
)


def _cfg(**over: Any) -> DurableCfg:
    base = {"enabled": True, "task_queue": f"tq-{uuid.uuid4().hex[:6]}",
            "schedules": DurableSchedulesCfg(retention_sweep="0 3 * * *",
                                             session_sweep="*/10 * * * *")}
    base.update(over)
    return DurableCfg(**base)


# ── pure ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_declared_schedules_read_cron_from_config():
    cfg = _cfg(schedules=DurableSchedulesCfg(retention_sweep="15 4 * * *",
                                             session_sweep="*/5 * * * *"))
    decls = schedules.declared_schedules(cfg)
    by_id = {d.schedule_id: d for d in decls}
    assert set(by_id) == {RETENTION_SCHEDULE_ID, SESSION_SCHEDULE_ID}
    ret = by_id[RETENTION_SCHEDULE_ID]
    assert ret.cron == "15 4 * * *"
    assert ret.workflow == RETENTION_WORKFLOW
    assert ret.workflow_id_prefix == RETENTION_WORKFLOW_ID_PREFIX
    assert ret.arg == RetentionSweepRequest()
    assert ret.task_queue == cfg.task_queue
    ses = by_id[SESSION_SCHEDULE_ID]
    assert ses.cron == "*/5 * * * *"
    assert ses.workflow == SESSION_WORKFLOW
    assert ses.workflow_id_prefix == SESSION_WORKFLOW_ID_PREFIX
    assert ses.arg == SessionSweepRequest()


@pytest.mark.unit
def test_apply_refuses_when_disabled():
    from attestor.durable.config import DurableDisabledError

    with pytest.raises(DurableDisabledError):
        asyncio.run(schedules.apply(object(), DurableCfg(enabled=False)))


class _FakeHandle:
    def __init__(self, client: _FakeClient, sid: str) -> None:
        self._client, self.id = client, sid
        self.updates = 0

    async def update(self, updater):
        self.updates += 1
        result = updater(type("Inp", (), {"description": self._client.existing[self.id]})())
        if result is not None:
            self._client.existing[self.id] = type("Desc", (), {"schedule": result.schedule})()


class _FakeClient:
    def __init__(self, existing: set[str] = frozenset()) -> None:
        self.existing: dict[str, Any] = {
            sid: type("Desc", (), {"schedule": None})() for sid in existing
        }
        self.created: list[tuple[str, Any]] = []
        self.handles: dict[str, _FakeHandle] = {}

    async def create_schedule(self, sid: str, schedule: Any, **kw: Any) -> _FakeHandle:
        if sid in self.existing:
            raise ScheduleAlreadyRunningError()
        self.created.append((sid, schedule))
        self.existing[sid] = type("Desc", (), {"schedule": schedule})()
        return self.get_schedule_handle(sid)

    def get_schedule_handle(self, sid: str) -> _FakeHandle:
        return self.handles.setdefault(sid, _FakeHandle(self, sid))


@pytest.mark.unit
async def test_apply_creates_missing_and_updates_existing():
    cfg = _cfg()
    client = _FakeClient(existing={SESSION_SCHEDULE_ID})
    results = schedules.summarize(await schedules.apply(client, cfg))
    assert results == {
        RETENTION_SCHEDULE_ID: schedules.APPLY_CREATED,
        SESSION_SCHEDULE_ID: schedules.APPLY_UPDATED,
    }
    assert [sid for sid, _ in client.created] == [RETENTION_SCHEDULE_ID]
    created = client.created[0][1]
    assert created.spec.cron_expressions == ["0 3 * * *"]
    assert created.action.task_queue == cfg.task_queue
    assert created.action.id == RETENTION_WORKFLOW_ID_PREFIX
    assert client.handles[SESSION_SCHEDULE_ID].updates == 1
    updated = client.existing[SESSION_SCHEDULE_ID].schedule
    assert updated.spec.cron_expressions == ["*/10 * * * *"]
    # second apply converges: nothing new created, both updated in place
    again = schedules.summarize(await schedules.apply(client, cfg))
    assert set(again.values()) == {schedules.APPLY_UPDATED}
    assert len(client.created) == 1


# ── live (dev server; schedules unsupported on the time-skipping server) ──

class _Stubs:
    def __init__(self) -> None:
        self.seen: list[RetentionSweepRequest] = []

    @activity.defn(name=LIST_DUE_POLICIES_ACTIVITY)
    async def list_due_policies(self, req: RetentionSweepRequest) -> DuePolicies:
        self.seen.append(req)
        return DuePolicies()

    @activity.defn(name=APPLY_POLICY_ACTIVITY)
    async def apply_policy(self, req):  # pragma: no cover - no policies are due
        raise AssertionError("no policies")


async def _wait_for_actions(handle, n: int, *, timeout: float = 20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        info = (await handle.describe()).info
        if info.num_actions >= n:
            return info
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"schedule fired {info.num_actions} < {n} actions")
        await asyncio.sleep(0.25)


@pytest.mark.integration
async def test_apply_is_idempotent_and_retention_fires_at_0300(temporal_dev_env):
    from temporalio.client import ScheduleBackfill, ScheduleOverlapPolicy

    from attestor.durable.worker import build_worker

    env = temporal_dev_env
    cfg = _cfg(namespace="default")
    stubs = _Stubs()
    try:
        first = schedules.summarize(await schedules.apply(env.client, cfg))
        assert set(first.values()) == {schedules.APPLY_CREATED}
        second = schedules.summarize(await schedules.apply(env.client, cfg))
        assert set(second.values()) == {schedules.APPLY_UPDATED}

        listed = {v.schedule_id: v for v in await schedules.list_declared(env.client, cfg)}
        assert listed[RETENTION_SCHEDULE_ID].workflow == RETENTION_WORKFLOW
        assert listed[SESSION_SCHEDULE_ID].workflow == SESSION_WORKFLOW
        assert listed[RETENTION_SCHEDULE_ID].present
        assert listed[SESSION_SCHEDULE_ID].present

        async with build_worker(env.client, cfg, activities=[stubs.list_due_policies,
                                                             stubs.apply_policy]):
            handle = env.client.get_schedule_handle(RETENTION_SCHEDULE_ID)
            day = datetime(2026, 1, 1, tzinfo=timezone.utc)
            await handle.backfill(ScheduleBackfill(
                start_at=day, end_at=day + timedelta(days=1),
                overlap=ScheduleOverlapPolicy.ALLOW_ALL,
            ))
            info = await _wait_for_actions(handle, 1)
            assert info.num_actions == 1  # one tick per day: 03:00
            fired = info.recent_actions[0]
            assert fired.scheduled_at == day.replace(hour=3)
            wid = fired.action.workflow_id
            assert wid == f"{RETENTION_WORKFLOW_ID_PREFIX}-2026-01-01T03:00:00Z"
            out = await env.client.get_workflow_handle(wid).result()
        assert out["policies_evaluated"] == 0
        assert stubs.seen == [RetentionSweepRequest()]
    finally:
        for decl in schedules.declared_schedules(cfg):
            with contextlib.suppress(Exception):  # best-effort cleanup
                await env.client.get_schedule_handle(decl.schedule_id).delete()
