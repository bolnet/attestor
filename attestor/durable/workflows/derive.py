# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``DeriveMemory`` — rebuild one memory's vector + graph, durably.

Workflow rules (plan §3): Attestor imports only under
``imports_passed_through`` and only ``attestor.durable.models``; no I/O,
no clock, no randomness; every activity call carries an explicit
``start_to_close_timeout`` and ``RetryPolicy``. Activities are invoked
by NAME so this module never imports the store / embedder stack.

Lanes run in order (vector, then graph) and are retried independently
with unlimited attempts; a non-retryable activity error (dimension
mismatch, bad credentials, missing row) does not stop the other lane.
When either lane ends in a permanent error the workflow fails with a
``DeriveFailed`` ``ApplicationError`` carrying the per-lane outcome in
its details, so the Temporal UI shows exactly which lane drifted.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        DERIVE_STEP_GRAPH,
        DERIVE_STEP_VECTOR,
        DERIVE_WORKFLOW,
        EMBED_AND_UPSERT_ACTIVITY,
        GRAPH_EXTRACT_ACTIVITY,
        ActivityPolicy,
        DeriveOutcome,
        DeriveRequest,
        DeriveStepOutcome,
    )

WORKFLOW_ERROR_TYPE = "DeriveFailed"


def _retry_policy(policy: ActivityPolicy) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=timedelta(seconds=policy.initial_interval_seconds),
        backoff_coefficient=policy.backoff_coefficient,
        maximum_interval=timedelta(seconds=policy.maximum_interval_seconds),
        maximum_attempts=policy.maximum_attempts,
    )


def _activity_error_message(exc: ActivityError) -> str:
    cause = exc.cause
    if isinstance(cause, ApplicationError):
        return cause.message
    return str(cause or exc)


@workflow.defn(name=DERIVE_WORKFLOW)
class DeriveMemory:
    """``embed_and_upsert`` then ``graph_extract`` for one memory id."""

    @workflow.run
    async def run(self, request: DeriveRequest) -> DeriveOutcome:
        vector = await self._lane(EMBED_AND_UPSERT_ACTIVITY, DERIVE_STEP_VECTOR, request)
        graph = await self._lane(GRAPH_EXTRACT_ACTIVITY, DERIVE_STEP_GRAPH, request)
        outcome = DeriveOutcome(memory_id=request.memory.memory_id, vector=vector, graph=graph)
        if not outcome.ok:
            raise ApplicationError(
                f"derive {outcome.memory_id} failed: {'; '.join(outcome.errors)}",
                outcome,
                type=WORKFLOW_ERROR_TYPE,
                non_retryable=True,
            )
        return outcome

    async def _lane(
        self, activity_name: str, step: str, request: DeriveRequest,
    ) -> DeriveStepOutcome:
        policy = request.policy
        try:
            return await workflow.execute_activity(
                activity_name,
                request.memory,
                result_type=DeriveStepOutcome,
                start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
                retry_policy=_retry_policy(policy),
            )
        except ActivityError as exc:
            return DeriveStepOutcome(
                memory_id=request.memory.memory_id, step=step, ok=False,
                error=_activity_error_message(exc),
            )
