"""RetentionSweep — activities (fake mem) + workflow (time-skipping, stub activities).

Also pins the ``apply_retention(policy_ids=...)`` filter the per-policy
activity relies on.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.compliance import retention  # noqa: E402
from attestor.compliance.retention import RetentionApplyResult, RetentionPolicy  # noqa: E402
from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import (  # noqa: E402
    APPLY_POLICY_ACTIVITY,
    LIST_DUE_POLICIES_ACTIVITY,
    ActivityPolicy,
    ApplyPolicyRequest,
    DuePolicies,
    DuePolicy,
    PolicyApplyOutcome,
    RetentionSweepOutcome,
    RetentionSweepRequest,
)

FAST = ActivityPolicy(initial_interval_seconds=0.01, maximum_interval_seconds=0.05)


def _policy(pid: str, action: str = "archive") -> RetentionPolicy:
    return RetentionPolicy(id=pid, name=f"pol-{pid}", namespace=None, category=None, layer=None,
                           tags_any=(), older_than_days=30, action=action, enabled=True,
                           created_at="")


# ── apply_retention(policy_ids=...) filter ─────────────────────────────

@pytest.mark.unit
def test_apply_retention_policy_ids_filter(monkeypatch):
    monkeypatch.setattr(retention, "list_retention_policies",
                        lambda mem, **kw: [_policy("p1"), _policy("p2", "delete")])
    monkeypatch.setattr(retention, "_count_matches", lambda conn, where, params: 7)

    class _Mem:
        _store = type("S", (), {"_conn": object()})()

    out = retention.apply_retention(_Mem(), dry_run=True, policy_ids=("p2",))
    assert out.policies_evaluated == 1
    assert set(out.by_policy) == {"p2"}
    assert out.memories_deleted == 7
    everything = retention.apply_retention(_Mem(), dry_run=True)
    assert everything.policies_evaluated == 2
    with pytest.raises(ValueError, match="at least one policy"):
        retention.apply_retention(_Mem(), dry_run=True, policy_ids=())


# ── activities ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_list_due_policies_maps_ids_and_actions(monkeypatch):
    from attestor.durable.activities.retention import RetentionActivities

    monkeypatch.setattr(retention, "list_retention_policies",
                        lambda mem, **kw: [_policy("p1"), _policy("p2", "delete")])
    acts = RetentionActivities(lambda: object())
    due = acts.list_due_policies(RetentionSweepRequest())
    assert due == DuePolicies(policies=(
        DuePolicy(policy_id="p1", name="pol-p1", action="archive"),
        DuePolicy(policy_id="p2", name="pol-p2", action="delete"),
    ))


@pytest.mark.unit
def test_apply_policy_wraps_apply_retention_for_one_policy(monkeypatch):
    from attestor.durable.activities.retention import RetentionActivities

    seen: dict[str, Any] = {}

    def fake_apply(mem, *, dry_run, initiated_by, policy_ids=None):
        seen.update(dry_run=dry_run, initiated_by=initiated_by, policy_ids=policy_ids)
        return RetentionApplyResult(
            policies_evaluated=1, memories_archived=0, memories_deleted=4, elapsed_ms=1.0,
            by_policy={"p2": {"name": "pol-p2", "action": "delete", "matched": 4,
                              "applied": 4, "vector_purged": 3}},
            dry_run=dry_run,
        )

    monkeypatch.setattr(retention, "apply_retention", fake_apply)
    acts = RetentionActivities(lambda: object())
    out = acts.apply_policy(ApplyPolicyRequest(policy_id="p2", initiated_by="sweep"))
    assert seen == {"dry_run": False, "initiated_by": "sweep", "policy_ids": ("p2",)}
    assert out == PolicyApplyOutcome(policy_id="p2", ok=True, action="delete", matched=4,
                                     applied=4, vector_purged=3)


@pytest.mark.unit
def test_apply_policy_disabled_meanwhile_is_a_noop(monkeypatch):
    from attestor.durable.activities.retention import RetentionActivities

    monkeypatch.setattr(retention, "apply_retention", lambda mem, **kw: RetentionApplyResult(
        policies_evaluated=0, memories_archived=0, memories_deleted=0, elapsed_ms=0.0,
    ))
    out = RetentionActivities(lambda: object()).apply_policy(ApplyPolicyRequest(policy_id="gone"))
    assert out.ok
    assert out.matched == 0
    assert out.applied == 0


@pytest.mark.unit
def test_apply_policy_errors_follow_retry_contract(monkeypatch):
    from attestor.durable.activities.retention import RetentionActivities

    def boom(mem, **kw):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(retention, "apply_retention", boom)
    with pytest.raises(ApplicationError) as exc:
        RetentionActivities(lambda: object()).apply_policy(ApplyPolicyRequest(policy_id="p1"))
    assert not exc.value.non_retryable

    def denied(mem, **kw):
        raise RuntimeError("permission denied for table memories")

    monkeypatch.setattr(retention, "apply_retention", denied)
    with pytest.raises(ApplicationError) as exc:
        RetentionActivities(lambda: object()).apply_policy(ApplyPolicyRequest(policy_id="p1"))
    assert exc.value.non_retryable


# ── workflow ───────────────────────────────────────────────────────────

class _Stubs:
    def __init__(self, due: list[DuePolicy], *, fail: dict[str, int] | None = None,
                 permanent: set[str] | None = None) -> None:
        self.due = due
        self.fail = dict(fail or {})
        self.permanent = permanent or set()
        self.applied: list[ApplyPolicyRequest] = []
        self.list_calls: list[RetentionSweepRequest] = []

    @activity.defn(name=LIST_DUE_POLICIES_ACTIVITY)
    async def list_due_policies(self, req: RetentionSweepRequest) -> DuePolicies:
        self.list_calls.append(req)
        return DuePolicies(policies=tuple(self.due))

    @activity.defn(name=APPLY_POLICY_ACTIVITY)
    async def apply_policy(self, req: ApplyPolicyRequest) -> PolicyApplyOutcome:
        pid = req.policy_id
        if pid in self.permanent:
            raise ApplicationError("permission denied", type="RetentionPermanent",
                                   non_retryable=True)
        if self.fail.get(pid, 0) > 0:
            self.fail[pid] -= 1
            raise ApplicationError("pg unavailable", type="RetentionFailed")
        self.applied.append(req)
        action = next(p.action for p in self.due if p.policy_id == pid)
        return PolicyApplyOutcome(policy_id=pid, ok=True, action=action, matched=2, applied=2,
                                  vector_purged=2 if action == "delete" else 0)

    def activities(self) -> list:
        return [self.list_due_policies, self.apply_policy]


async def _run(env, stubs: _Stubs, request: RetentionSweepRequest,
               workflow_id: str) -> RetentionSweepOutcome:
    from attestor.durable.worker import build_worker
    from attestor.durable.workflows.retention import RetentionSweep

    cfg = DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")
    async with build_worker(env.client, cfg, activities=stubs.activities()):
        return await env.client.execute_workflow(
            RetentionSweep.run, request, id=workflow_id, task_queue=cfg.task_queue,
        )


@pytest.mark.integration
async def test_sweep_applies_every_due_policy(temporal_env):
    due = [DuePolicy("p1", "a", "archive"), DuePolicy("p2", "b", "delete")]
    stubs = _Stubs(due, fail={"p2": 2})
    req = RetentionSweepRequest(initiated_by="sweep", policy=FAST)
    out = await _run(temporal_env, stubs, req, "retention-sweep-test-ok")
    assert out.ok
    assert out.policies_evaluated == 2
    assert out.memories_archived == 2
    assert out.memories_deleted == 2
    assert [r.policy_id for r in stubs.applied] == ["p1", "p2"]
    assert all(r.initiated_by == "sweep" and r.dry_run is False for r in stubs.applied)
    assert stubs.list_calls[0].initiated_by == "sweep"


@pytest.mark.integration
async def test_sweep_reports_permanent_policy_failure_without_skipping_the_rest(temporal_env):
    due = [DuePolicy("p1", "a", "archive"), DuePolicy("p2", "b", "delete"),
           DuePolicy("p3", "c", "archive")]
    stubs = _Stubs(due, permanent={"p2"})
    with pytest.raises(WorkflowFailureError) as exc:
        await _run(temporal_env, stubs, RetentionSweepRequest(policy=FAST),
                   "retention-sweep-test-fail")
    assert [r.policy_id for r in stubs.applied] == ["p1", "p3"]
    cause = exc.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    outcome = cause.details[0]
    failed = [p for p in outcome["by_policy"] if not p["ok"]]
    assert [p["policy_id"] for p in failed] == ["p2"]


@pytest.mark.integration
async def test_sweep_with_no_policies_completes(temporal_env):
    out = await _run(temporal_env, _Stubs([]), RetentionSweepRequest(policy=FAST),
                     "retention-sweep-test-empty")
    assert out.ok
    assert out.policies_evaluated == 0
    assert out.by_policy == ()
