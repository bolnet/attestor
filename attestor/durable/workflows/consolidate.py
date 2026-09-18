# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``ConsolidateEpisode`` — one durable run of the per-episode consolidator.

Workflow rules (plan §3): Attestor imports only under
``imports_passed_through`` and only ``attestor.durable.models``; no I/O,
no clock, no randomness; the single activity call carries an explicit
``start_to_close_timeout`` and ``RetryPolicy``. The activity is invoked
by NAME so this module never imports the extraction / LLM stack.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        CONSOLIDATE_EPISODE_ACTIVITY,
        CONSOLIDATE_WORKFLOW,
        ActivityPolicy,
        ConsolidateRequest,
        ConsolidationOutcome,
    )


def _retry_policy(policy: ActivityPolicy) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=timedelta(seconds=policy.initial_interval_seconds),
        backoff_coefficient=policy.backoff_coefficient,
        maximum_interval=timedelta(seconds=policy.maximum_interval_seconds),
        maximum_attempts=policy.maximum_attempts,
    )


@workflow.defn(name=CONSOLIDATE_WORKFLOW)
class ConsolidateEpisode:
    """Run the ``consolidate_episode`` activity until it succeeds or is
    marked non-retryable (permanent error) / exhausts ``maximum_attempts``."""

    @workflow.run
    async def run(self, request: ConsolidateRequest) -> ConsolidationOutcome:
        policy = request.policy
        return await workflow.execute_activity(
            CONSOLIDATE_EPISODE_ACTIVITY,
            request.episode,
            result_type=ConsolidationOutcome,
            start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
            retry_policy=_retry_policy(policy),
        )
