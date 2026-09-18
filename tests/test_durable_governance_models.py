"""Phase 3 boundary models — ForgetUser saga, RetentionSweep, SessionSweep.

Frozen, ids + counts only, deterministic workflow ids. ``attestor.durable.models``
stays stdlib-only (guarded by ``tests/test_durable_models.py``).
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import pytest

from attestor.durable.models import (
    APPLY_POLICY_ACTIVITY,
    FORGET_DOC_ACTIVITY,
    FORGET_GRAPH_ACTIVITY,
    FORGET_LANE_AUDIT,
    FORGET_LANE_DOC,
    FORGET_LANE_GRAPH,
    FORGET_LANE_STATE,
    FORGET_LANE_VECTOR,
    FORGET_LANES,
    FORGET_STATE_ACTIVITY,
    FORGET_VECTOR_ACTIVITY,
    FORGET_WORKFLOW,
    LANE_STATUS_FAILED,
    LANE_STATUS_OK,
    LANE_STATUS_PENDING,
    LIST_DUE_POLICIES_ACTIVITY,
    MEMORY_CONTENT_FIELDS,
    RETENTION_SCHEDULE_ID,
    RETENTION_WORKFLOW,
    SESSION_SCHEDULE_ID,
    SESSION_WORKFLOW,
    SWEEP_ARCHIVED_ACTIVITY,
    SWEEP_ENDED_ACTIVITY,
    SWEEP_IDLE_ACTIVITY,
    WRITE_FORGET_AUDIT_ACTIVITY,
    ActivityPolicy,
    ApplyPolicyRequest,
    DuePolicies,
    DuePolicy,
    ForgetLaneOutcome,
    ForgetRequest,
    ForgetUserOutcome,
    PolicyApplyOutcome,
    RetentionSweepOutcome,
    RetentionSweepRequest,
    SessionSweepOutcome,
    SessionSweepRequest,
    SessionSweepStep,
    forget_workflow_id,
    retention_sweep_workflow_id,
    session_sweep_workflow_id,
)

pytestmark = pytest.mark.unit


def test_names_are_stable_strings():
    assert FORGET_WORKFLOW == "ForgetUser"
    assert RETENTION_WORKFLOW == "RetentionSweep"
    assert SESSION_WORKFLOW == "SessionSweep"
    assert WRITE_FORGET_AUDIT_ACTIVITY == "write_forget_audit"
    assert (FORGET_DOC_ACTIVITY, FORGET_VECTOR_ACTIVITY, FORGET_GRAPH_ACTIVITY,
            FORGET_STATE_ACTIVITY) == ("forget_doc", "forget_vector", "forget_graph",
                                       "forget_state")
    assert (LIST_DUE_POLICIES_ACTIVITY, APPLY_POLICY_ACTIVITY) == (
        "list_due_policies", "apply_policy",
    )
    assert (SWEEP_IDLE_ACTIVITY, SWEEP_ENDED_ACTIVITY, SWEEP_ARCHIVED_ACTIVITY) == (
        "sweep_idle", "sweep_ended", "sweep_archived",
    )
    assert RETENTION_SCHEDULE_ID != SESSION_SCHEDULE_ID
    assert FORGET_LANES == (FORGET_LANE_DOC, FORGET_LANE_VECTOR, FORGET_LANE_GRAPH,
                            FORGET_LANE_STATE)


def test_workflow_ids_are_deterministic():
    assert forget_workflow_id("u-1", "a-1") == "forget-u-1-a-1"
    assert retention_sweep_workflow_id(date(2026, 9, 3)) == "retention-sweep-2026-09-03"
    at = datetime(2026, 9, 3, 3, 10, 59, tzinfo=timezone.utc)
    assert session_sweep_workflow_id(at) == "session-sweep-2026-09-03T03:10"
    # naive → UTC, aware → converted
    assert session_sweep_workflow_id(at.replace(tzinfo=None)) == "session-sweep-2026-09-03T03:10"


def test_forget_request_is_frozen_and_ids_only():
    req = ForgetRequest(user_id="u-1", audit_id="a-1", initiated_by="ops")
    assert dataclasses.is_dataclass(req)
    assert req.__dataclass_params__.frozen
    assert req.policy == ActivityPolicy()
    names = {f.name for f in dataclasses.fields(ForgetRequest)}
    assert not names & set(MEMORY_CONTENT_FIELDS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.user_id = "x"  # type: ignore[misc]


def test_forget_outcome_pending_then_lanes_fill_in():
    out = ForgetUserOutcome.pending("u-1", "a-1")
    assert out.audit.lane == FORGET_LANE_AUDIT
    assert [lane.lane for lane in out.lanes] == [
        FORGET_LANE_AUDIT, FORGET_LANE_DOC, FORGET_LANE_VECTOR, FORGET_LANE_GRAPH,
        FORGET_LANE_STATE,
    ]
    assert all(lane.status == LANE_STATUS_PENDING for lane in out.lanes)
    assert not out.ok
    assert out.errors == ()

    filled = out
    for lane in (FORGET_LANE_AUDIT, FORGET_LANE_DOC, FORGET_LANE_VECTOR, FORGET_LANE_STATE):
        filled = filled.with_lane(ForgetLaneOutcome(lane=lane, status=LANE_STATUS_OK, deleted=2))
    assert out.document.status == LANE_STATUS_PENDING  # original untouched
    assert not filled.ok  # graph still pending
    done = filled.with_lane(
        ForgetLaneOutcome(lane=FORGET_LANE_GRAPH, status=LANE_STATUS_OK, deleted=3,
                          edges_deleted=4),
    )
    assert done.ok
    assert done.graph.edges_deleted == 4
    assert done.total_deleted == 2 * 3 + 3 + 4  # doc+vector+state, nodes, edges
    failed = filled.with_lane(
        ForgetLaneOutcome(lane=FORGET_LANE_GRAPH, status=LANE_STATUS_FAILED, error="neo4j down"),
    )
    assert not failed.ok
    assert failed.errors == ("graph: neo4j down",)
    assert failed.failed_lanes == (FORGET_LANE_GRAPH,)
    with pytest.raises(ValueError, match="unknown forget lane"):
        out.with_lane(ForgetLaneOutcome(lane="nope"))


def test_forget_lane_outcome_defaults():
    lane = ForgetLaneOutcome(lane=FORGET_LANE_VECTOR)
    assert lane.status == LANE_STATUS_PENDING
    assert lane.deleted == 0
    assert lane.edges_deleted == 0
    assert not lane.skipped
    assert lane.error is None
    assert not lane.ok


def test_retention_models():
    req = RetentionSweepRequest()
    assert req.dry_run is False
    assert req.initiated_by  # sweeps always sign the audit row
    assert req.policy == ActivityPolicy()
    due = DuePolicies(policies=(DuePolicy(policy_id="p1", name="n", action="delete"),))
    assert due.policies[0].policy_id == "p1"
    apply = ApplyPolicyRequest(policy_id="p1", dry_run=True, initiated_by="x")
    assert apply.policy_id == "p1"
    out = RetentionSweepOutcome(
        policies_evaluated=2, memories_archived=1, memories_deleted=0, dry_run=False,
        by_policy=(
            PolicyApplyOutcome(policy_id="p1", ok=True, action="archive", matched=1, applied=1),
            PolicyApplyOutcome(policy_id="p2", ok=False, action="delete", error="boom"),
        ),
    )
    assert not out.ok
    assert out.failed_policy_ids == ("p2",)
    assert out.errors == ("p2: boom",)


def test_session_sweep_models():
    req = SessionSweepRequest()
    assert req.idle_after_minutes > 0
    assert req.ended_after_minutes > req.idle_after_minutes
    assert req.archived_after_minutes > req.ended_after_minutes
    assert req.batch_limit > 0
    out = SessionSweepOutcome(
        idle=SessionSweepStep(transition="idle", count=2),
        ended=SessionSweepStep(transition="ended", count=1),
        archived=SessionSweepStep(transition="archived", ok=False, error="pg down"),
    )
    assert not out.ok
    assert out.total == 3
    assert out.errors == ("archived: pg down",)
    assert [s.transition for s in out.steps] == ["idle", "ended", "archived"]
