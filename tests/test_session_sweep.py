"""SessionSweep — time-driven idle → ended → archived transitions.

``SessionRepo.sweep_*`` are pure SQL (fake connection asserts shape + params);
``SessionActivities`` wrap them; the workflow runs the three steps in order
on the time-skipping server with stub activities.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

import pytest

from attestor.identity.sessions import SessionRepo
from attestor.store.conn_lock import lock_for_connection
from tests._lockprobe import held_by_caller

# ── SessionRepo sweep queries (pure SQL) ────────────────────────────────


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._rows: list[Any] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append((" ".join(sql.split()), tuple(params or ())))
        self._conn.lock_held.append(held_by_caller(lock_for_connection(self._conn)))
        if self._conn.fail:
            raise self._conn.fail
        self._rows = [{"id": i} for i in self._conn.ids]

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class _Conn:
    def __init__(self, ids: list[str] | None = None, fail: Exception | None = None) -> None:
        self.ids = ids or []
        self.fail = fail
        self.executed: list[tuple[str, tuple]] = []
        self.lock_held: list[bool] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **_: Any) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.mark.unit
def test_sweep_idle_moves_stale_active_sessions():
    conn = _Conn(ids=["s1", "s2"])
    out = SessionRepo(conn).sweep_idle(timedelta(minutes=30), limit=100)
    assert out == ("s1", "s2")
    sql, params = conn.executed[-1]
    low = sql.lower()
    assert low.startswith("update sessions set status = 'idle'")
    assert "status = 'active'" in low
    assert "last_active_at < now() - %s" in low
    assert "limit %s" in low
    assert "returning id" in low
    assert params == (timedelta(minutes=30), 100)
    assert conn.commits == 1


@pytest.mark.unit
def test_sweeps_run_inside_the_connection_lock():
    """The worker pool shares ONE connection; every sweep must hold its lock."""
    conn = _Conn(ids=["s1"])
    repo = SessionRepo(conn)
    repo.sweep_idle(timedelta(minutes=30), limit=5)
    repo.sweep_ended(timedelta(hours=1), limit=5)
    repo.sweep_archived(timedelta(days=1), limit=5)
    assert conn.lock_held == [True, True, True]
    assert not held_by_caller(lock_for_connection(conn))


@pytest.mark.unit
def test_sweep_releases_the_lock_after_a_failure():
    conn = _Conn(fail=RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        SessionRepo(conn).sweep_idle(timedelta(minutes=1), limit=1)
    assert conn.rollbacks == 1
    assert not held_by_caller(lock_for_connection(conn))


@pytest.mark.unit
def test_sweep_ended_stamps_ended_at_from_idle():
    conn = _Conn(ids=["s3"])
    out = SessionRepo(conn).sweep_ended(timedelta(hours=24), limit=10)
    assert out == ("s3",)
    sql, params = conn.executed[-1]
    low = sql.lower()
    assert "set status = 'ended', ended_at = now()" in low
    assert "status = 'idle'" in low
    assert params == (timedelta(hours=24), 10)


@pytest.mark.unit
def test_sweep_archived_from_ended():
    conn = _Conn(ids=[])
    out = SessionRepo(conn).sweep_archived(timedelta(days=30), limit=10)
    assert out == ()
    sql, params = conn.executed[-1]
    low = sql.lower()
    assert "set status = 'archived'" in low
    assert "status = 'ended'" in low
    assert "coalesce(ended_at, last_active_at) < now() - %s" in low
    assert params == (timedelta(days=30), 10)


@pytest.mark.unit
def test_sweeps_validate_inputs_and_roll_back_on_error():
    repo = SessionRepo(_Conn())
    with pytest.raises(ValueError, match="interval must be positive"):
        repo.sweep_idle(timedelta(0), limit=10)
    with pytest.raises(ValueError, match="limit must be > 0"):
        repo.sweep_idle(timedelta(minutes=1), limit=0)
    conn = _Conn(fail=RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        SessionRepo(conn).sweep_ended(timedelta(minutes=1), limit=1)
    assert conn.rollbacks == 1


@pytest.mark.unit
def test_end_accepts_idle_sessions():
    """Docstring contract: active/idle → ended (the sweeper produces idle rows)."""
    import inspect

    src = inspect.getsource(SessionRepo.end)
    assert "'idle'" in src


# ── activities + workflow ──────────────────────────────────────────────

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    SWEEP_ARCHIVED_ACTIVITY,
    SWEEP_ENDED_ACTIVITY,
    SWEEP_IDLE_ACTIVITY,
    ActivityPolicy,
    SessionSweepOutcome,
    SessionSweepRequest,
    SessionSweepStep,
)

FAST = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)


class _Mem:
    def __init__(self, conn: _Conn) -> None:
        self._store = type("S", (), {"_conn": conn})()


@pytest.mark.unit
def test_session_activities_translate_minutes_to_intervals():
    from attestor.durable.activities.sessions import SessionActivities

    conn = _Conn(ids=["a", "b", "c"])
    acts = SessionActivities(lambda: _Mem(conn))
    req = SessionSweepRequest(idle_after_minutes=15, ended_after_minutes=60,
                              archived_after_minutes=1440, batch_limit=50)
    idle = acts.sweep_idle(req)
    assert idle == SessionSweepStep(transition="idle", count=3)
    assert conn.executed[-1][1] == (timedelta(minutes=15), 50)
    ended = acts.sweep_ended(req)
    assert ended.transition == "ended"
    assert conn.executed[-1][1][0] == timedelta(minutes=60)
    archived = acts.sweep_archived(req)
    assert archived.transition == "archived"
    assert conn.executed[-1][1][0] == timedelta(minutes=1440)


@pytest.mark.unit
def test_session_activity_errors_follow_retry_contract():
    from attestor.durable.activities.sessions import SessionActivities

    acts = SessionActivities(lambda: _Mem(_Conn(fail=RuntimeError("connection refused"))))
    with pytest.raises(ApplicationError) as exc:
        acts.sweep_idle(SessionSweepRequest())
    assert not exc.value.non_retryable


class _Stubs:
    def __init__(self, *, fail: dict[str, int] | None = None,
                 permanent: set[str] | None = None) -> None:
        self.fail = dict(fail or {})
        self.permanent = permanent or set()
        self.order: list[str] = []
        self.seen: list[SessionSweepRequest] = []

    async def _step(self, transition: str, req: SessionSweepRequest) -> SessionSweepStep:
        self.seen.append(req)
        if transition in self.permanent:
            raise ApplicationError("permission denied", type="SessionSweepPermanent",
                                   non_retryable=True)
        if self.fail.get(transition, 0) > 0:
            self.fail[transition] -= 1
            raise ApplicationError("pg unavailable", type="SessionSweepFailed")
        self.order.append(transition)
        return SessionSweepStep(transition=transition, count=len(self.order))

    @activity.defn(name=SWEEP_IDLE_ACTIVITY)
    async def sweep_idle(self, req: SessionSweepRequest) -> SessionSweepStep:
        return await self._step("idle", req)

    @activity.defn(name=SWEEP_ENDED_ACTIVITY)
    async def sweep_ended(self, req: SessionSweepRequest) -> SessionSweepStep:
        return await self._step("ended", req)

    @activity.defn(name=SWEEP_ARCHIVED_ACTIVITY)
    async def sweep_archived(self, req: SessionSweepRequest) -> SessionSweepStep:
        return await self._step("archived", req)

    def activities(self) -> list:
        return [self.sweep_idle, self.sweep_ended, self.sweep_archived]


async def _run(env, stubs: _Stubs, request: SessionSweepRequest,
               workflow_id: str) -> SessionSweepOutcome:
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.sessions import SessionSweep

    cfg = DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")
    async with build_worker(env.client, cfg, activities=stubs.activities()):
        return await env.client.execute_workflow(
            SessionSweep.run, request, id=workflow_id, task_queue=cfg.task_queue,
        )


@pytest.mark.integration
async def test_session_sweep_runs_transitions_in_order_and_retries(temporal_env):
    stubs = _Stubs(fail={"ended": 2})
    req = SessionSweepRequest(idle_after_minutes=5, policy=FAST)
    out = await _run(temporal_env, stubs, req, "session-sweep-test-ok")
    assert out.ok
    assert stubs.order == ["idle", "ended", "archived"]
    assert (out.idle.count, out.ended.count, out.archived.count) == (1, 2, 3)
    assert out.total == 6
    assert all(s.idle_after_minutes == 5 for s in stubs.seen)


@pytest.mark.integration
async def test_session_sweep_reports_permanent_step_failure(temporal_env):
    stubs = _Stubs(permanent={"ended"})
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stubs, SessionSweepRequest(policy=FAST),
                   "session-sweep-test-fail")
    assert stubs.order == ["idle", "archived"]  # later steps still run
    cause = exc.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    outcome = cause.details[0]
    assert outcome["ended"]["ok"] is False
    assert outcome["idle"]["ok"] is True
