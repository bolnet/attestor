# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``RebuildDerived`` — sliding-window batch of child ``DeriveMemory`` runs.

The first real implementation of "derived state is rebuildable from
Postgres": page over memory ids (``list_memory_ids`` activity, keyset by
id, heartbeating), start one child ``derive-{memory_id}`` per id with at
most ``window_size`` in flight, and continue-as-new every
``max_pages_per_run`` pages so history stays bounded and the run is
resumable from its cursor. A child that fails permanently is counted
(and its id reported) — it never aborts the batch. A child whose id is
already running (an ``add()`` repair in flight) is skipped, not
duplicated.

One run = one tenant: ``request.user_id`` (or the scope the list
activity resolved, echoed in ``MemoryIdPage.user_id``) is stamped on
every child ``DeriveRef`` so the worker derives as the row owner.

Workflow rules (plan §3) apply: models only via pass-through, no I/O,
no clock, explicit timeouts + retry policies.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ChildWorkflowError, WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from attestor.durable.models import (
        DERIVE_WORKFLOW,
        LIST_MEMORY_IDS_ACTIVITY,
        MAX_REPORTED_FAILED_IDS,
        REBUILD_WORKFLOW,
        ActivityPolicy,
        DeriveOutcome,
        DeriveRef,
        DeriveRequest,
        ListMemoryIdsRequest,
        MemoryIdPage,
        RebuildOutcome,
        RebuildRequest,
        derive_workflow_id,
    )

LIST_HEARTBEAT_SECONDS = 60.0


def _retry_policy(policy: ActivityPolicy) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=timedelta(seconds=policy.initial_interval_seconds),
        backoff_coefficient=policy.backoff_coefficient,
        maximum_interval=timedelta(seconds=policy.maximum_interval_seconds),
        maximum_attempts=policy.maximum_attempts,
    )


@workflow.defn(name=REBUILD_WORKFLOW)
class RebuildDerived:
    """Batch-sliding-window rebuild of vector + graph state over Postgres ids."""

    def __init__(self) -> None:
        self._inflight: set[asyncio.Task[None]] = set()
        self._progress = None
        self._failed_ids: tuple[str, ...] = ()

    @workflow.run
    async def run(self, request: RebuildRequest) -> RebuildOutcome:
        self._progress = request.progress.with_run()
        self._failed_ids = request.failed_ids
        after_id = request.after_id
        pages_this_run = 0
        while True:
            page = await self._list_page(request, after_id)
            scope = page.user_id or request.user_id
            for memory_id in page.ids:
                await workflow.wait_condition(
                    lambda: len(self._inflight) < request.window_size,
                )
                self._spawn(memory_id, request, scope)
            self._progress = self._progress.with_page(len(page.ids))
            pages_this_run += 1
            after_id = page.next_after_id
            if after_id is None:
                break
            if pages_this_run >= request.max_pages_per_run:
                await self._drain()
                workflow.continue_as_new(
                    replace(
                        request, after_id=after_id, progress=self._progress,
                        failed_ids=self._failed_ids,
                    ),
                )
        await self._drain()
        return RebuildOutcome(progress=self._progress, done=True, failed_ids=self._failed_ids)

    @workflow.query
    def progress(self) -> RebuildOutcome:
        return RebuildOutcome(
            progress=self._progress or RebuildRequest().progress,
            done=False, failed_ids=self._failed_ids,
        )

    async def _list_page(self, request: RebuildRequest, after_id: str | None) -> MemoryIdPage:
        policy = request.policy
        return await workflow.execute_activity(
            LIST_MEMORY_IDS_ACTIVITY,
            ListMemoryIdsRequest(
                since=request.since, namespace=request.namespace,
                user_id=request.user_id, after_id=after_id, limit=request.page_size,
            ),
            result_type=MemoryIdPage,
            start_to_close_timeout=timedelta(seconds=policy.start_to_close_seconds),
            heartbeat_timeout=timedelta(seconds=LIST_HEARTBEAT_SECONDS),
            retry_policy=_retry_policy(policy),
        )

    def _spawn(self, memory_id: str, request: RebuildRequest, user_id: str | None) -> None:
        task = asyncio.create_task(self._derive_one(memory_id, request, user_id))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _derive_one(
        self, memory_id: str, request: RebuildRequest, user_id: str | None,
    ) -> None:
        child = DeriveRequest(
            memory=DeriveRef(memory_id=memory_id, namespace=request.namespace, user_id=user_id),
            policy=request.policy,
        )
        try:
            await workflow.execute_child_workflow(
                DERIVE_WORKFLOW, child,
                id=derive_workflow_id(memory_id),
                result_type=DeriveOutcome,
            )
        except WorkflowAlreadyStartedError:
            self._progress = self._progress.with_child_skipped()
            return
        except ChildWorkflowError:
            self._progress = self._progress.with_child_failed()
            if len(self._failed_ids) < MAX_REPORTED_FAILED_IDS:
                self._failed_ids = (*self._failed_ids, memory_id)
            return
        self._progress = self._progress.with_child_ok()

    async def _drain(self) -> None:
        await workflow.wait_condition(lambda: not self._inflight)
