# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``RetentionSweep`` — apply every due retention policy, one activity each.

Started by the ``attestor-retention-sweep`` Schedule (cron from
``durable.schedules.retention_sweep``). Policies are applied
sequentially — retention is rare and audit correctness beats throughput
(``compliance/retention.py``). A policy that fails permanently is
recorded and the rest still run; the run then fails with the per-policy
outcome in the error details so the UI shows which policy needs a human.

Workflow rules (plan §3): models via pass-through only, no I/O, no clock.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        APPLY_POLICY_ACTIVITY,
        LIST_DUE_POLICIES_ACTIVITY,
        RETENTION_WORKFLOW,
        ApplyPolicyRequest,
        DuePolicies,
        DuePolicy,
        PolicyApplyOutcome,
        RetentionSweepOutcome,
        RetentionSweepRequest,
    )
    from attestor.durable.workflows._policy import activity_error_message, retry_policy

WORKFLOW_ERROR_TYPE = "RetentionSweepFailed"
ACTION_ARCHIVE = "archive"


def _totals(outcomes: tuple[PolicyApplyOutcome, ...]) -> tuple[int, int]:
    archived = sum(o.applied for o in outcomes if o.ok and o.action == ACTION_ARCHIVE)
    deleted = sum(o.applied for o in outcomes if o.ok and o.action and o.action != ACTION_ARCHIVE)
    return archived, deleted


@workflow.defn(name=RETENTION_WORKFLOW)
class RetentionSweep:
    """``list_due_policies`` then ``apply_policy`` per policy."""

    @workflow.run
    async def run(self, request: RetentionSweepRequest) -> RetentionSweepOutcome:
        due = await self._list(request)
        outcomes: tuple[PolicyApplyOutcome, ...] = ()
        for policy in due.policies:
            outcomes = (*outcomes, await self._apply(policy, request))
        archived, deleted = _totals(outcomes)
        outcome = RetentionSweepOutcome(
            policies_evaluated=len(due.policies), memories_archived=archived,
            memories_deleted=deleted, dry_run=request.dry_run, by_policy=outcomes,
        )
        if not outcome.ok:
            raise ApplicationError(
                f"retention sweep: {len(outcome.failed_policy_ids)} policy(ies) failed: "
                + "; ".join(outcome.errors),
                outcome, type=WORKFLOW_ERROR_TYPE, non_retryable=True,
            )
        return outcome

    async def _list(self, request: RetentionSweepRequest) -> DuePolicies:
        policy = request.policy
        return await workflow.execute_activity(
            LIST_DUE_POLICIES_ACTIVITY, request, result_type=DuePolicies,
            start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
            retry_policy=retry_policy(policy),
        )

    async def _apply(self, due: DuePolicy, request: RetentionSweepRequest) -> PolicyApplyOutcome:
        policy = request.policy
        try:
            return await workflow.execute_activity(
                APPLY_POLICY_ACTIVITY,
                ApplyPolicyRequest(
                    policy_id=due.policy_id, dry_run=request.dry_run,
                    initiated_by=request.initiated_by,
                ),
                result_type=PolicyApplyOutcome,
                start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
                retry_policy=retry_policy(policy),
            )
        except ActivityError as exc:
            return PolicyApplyOutcome(
                policy_id=due.policy_id, ok=False, action=due.action,
                error=activity_error_message(exc),
            )
