# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``ForgetUser`` — GDPR right-to-be-forgotten as a durable saga.

Order (plan §4 Phase 3):

1. ``write_forget_audit`` — append-only, RLS-exempt, retried until it
   lands. If it fails permanently NO backend is touched: the deletion
   event must be on record before any delete.
2. ``forget_doc`` — Postgres is the source of truth, so it goes first.
3. ``forget_vector``, ``forget_graph``, ``forget_state`` — concurrent,
   each retried independently (unlimited attempts). A Neo4j outage keeps
   the graph lane retrying while the others complete.

The run completes when every lane is ``ok``; otherwise it fails with a
``ForgetFailed`` ``ApplicationError`` whose details carry the per-backend
``ForgetUserOutcome`` — a regulator-visible partial state in the Temporal
UI. ``progress`` is a query for ``attestor durable status forget-...``.

Workflow rules (plan §3): only ``attestor.durable.models`` via
pass-through, no I/O, no clock, activities invoked by NAME.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        FORGET_DOC_ACTIVITY,
        FORGET_GRAPH_ACTIVITY,
        FORGET_LANE_AUDIT,
        FORGET_LANE_DOC,
        FORGET_LANE_GRAPH,
        FORGET_LANE_STATE,
        FORGET_LANE_VECTOR,
        FORGET_STATE_ACTIVITY,
        FORGET_VECTOR_ACTIVITY,
        FORGET_WORKFLOW,
        LANE_STATUS_FAILED,
        WRITE_FORGET_AUDIT_ACTIVITY,
        ForgetLaneOutcome,
        ForgetRequest,
        ForgetUserOutcome,
    )
    from attestor.durable.workflows._policy import activity_error_message, retry_policy

WORKFLOW_ERROR_TYPE = "ForgetFailed"
AUDIT_ERROR_TYPE = "ForgetAuditFailed"

_ACTIVITY_FOR_LANE: dict[str, str] = {
    FORGET_LANE_AUDIT: WRITE_FORGET_AUDIT_ACTIVITY,
    FORGET_LANE_DOC: FORGET_DOC_ACTIVITY,
    FORGET_LANE_VECTOR: FORGET_VECTOR_ACTIVITY,
    FORGET_LANE_GRAPH: FORGET_GRAPH_ACTIVITY,
    FORGET_LANE_STATE: FORGET_STATE_ACTIVITY,
}
_CONCURRENT_LANES: tuple[str, ...] = (FORGET_LANE_VECTOR, FORGET_LANE_GRAPH, FORGET_LANE_STATE)


@workflow.defn(name=FORGET_WORKFLOW)
class ForgetUser:
    """Audit → document → {vector, graph, state}; per-backend retried."""

    def __init__(self) -> None:
        self._outcome: ForgetUserOutcome | None = None

    @workflow.run
    async def run(self, request: ForgetRequest) -> ForgetUserOutcome:
        self._outcome = ForgetUserOutcome.pending(request.user_id, request.audit_id)
        audit = await self._lane(FORGET_LANE_AUDIT, request)
        if not audit.ok:
            raise ApplicationError(
                f"forget {request.user_id}: audit row not written ({audit.error}); "
                "no backend was touched",
                self._outcome, type=AUDIT_ERROR_TYPE, non_retryable=True,
            )
        await self._lane(FORGET_LANE_DOC, request)
        await asyncio.gather(*(self._lane(lane, request) for lane in _CONCURRENT_LANES))
        outcome = self._outcome
        if not outcome.ok:
            raise ApplicationError(
                f"forget {request.user_id} partial: {'; '.join(outcome.errors)}",
                outcome, type=WORKFLOW_ERROR_TYPE, non_retryable=True,
            )
        return outcome

    @workflow.query
    def progress(self) -> ForgetUserOutcome | None:
        return self._outcome

    async def _lane(self, lane: str, request: ForgetRequest) -> ForgetLaneOutcome:
        policy = request.policy
        try:
            result = await workflow.execute_activity(
                _ACTIVITY_FOR_LANE[lane],
                request,
                result_type=ForgetLaneOutcome,
                start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
                retry_policy=retry_policy(policy),
            )
        except ActivityError as exc:
            result = ForgetLaneOutcome(
                lane=lane, status=LANE_STATUS_FAILED, error=activity_error_message(exc),
            )
        self._outcome = self._outcome.with_lane(result)
        return result
