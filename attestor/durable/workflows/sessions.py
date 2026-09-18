# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``SessionSweep`` — idle → ended → archived, one activity per transition.

Started by the ``attestor-session-sweep`` Schedule (cron from
``durable.schedules.session_sweep``). Steps run in lifecycle order; a
step that fails permanently is recorded and the later steps still run,
then the run fails with the per-step outcome in the error details.

Workflow rules (plan §3): models via pass-through only, no I/O, no clock.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        SESSION_TRANSITION_ARCHIVED,
        SESSION_TRANSITION_ENDED,
        SESSION_TRANSITION_IDLE,
        SESSION_WORKFLOW,
        SWEEP_ARCHIVED_ACTIVITY,
        SWEEP_ENDED_ACTIVITY,
        SWEEP_IDLE_ACTIVITY,
        SessionSweepOutcome,
        SessionSweepRequest,
        SessionSweepStep,
    )
    from attestor.durable.workflows._policy import activity_error_message, retry_policy

WORKFLOW_ERROR_TYPE = "SessionSweepFailed"
_STEPS: tuple[tuple[str, str], ...] = (
    (SESSION_TRANSITION_IDLE, SWEEP_IDLE_ACTIVITY),
    (SESSION_TRANSITION_ENDED, SWEEP_ENDED_ACTIVITY),
    (SESSION_TRANSITION_ARCHIVED, SWEEP_ARCHIVED_ACTIVITY),
)


@workflow.defn(name=SESSION_WORKFLOW)
class SessionSweep:
    """``sweep_idle`` → ``sweep_ended`` → ``sweep_archived``."""

    @workflow.run
    async def run(self, request: SessionSweepRequest) -> SessionSweepOutcome:
        steps = [await self._step(transition, name, request) for transition, name in _STEPS]
        outcome = SessionSweepOutcome(idle=steps[0], ended=steps[1], archived=steps[2])
        if not outcome.ok:
            raise ApplicationError(
                "session sweep: " + "; ".join(outcome.errors),
                outcome, type=WORKFLOW_ERROR_TYPE, non_retryable=True,
            )
        return outcome

    async def _step(
        self, transition: str, activity_name: str, request: SessionSweepRequest,
    ) -> SessionSweepStep:
        policy = request.policy
        try:
            return await workflow.execute_activity(
                activity_name, request, result_type=SessionSweepStep,
                start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
                retry_policy=retry_policy(policy),
            )
        except ActivityError as exc:
            return SessionSweepStep(
                transition=transition, ok=False, error=activity_error_message(exc),
            )
