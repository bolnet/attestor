# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Declarative Temporal Schedules for the governance sweeps + idempotent ``apply``.

Cron strings come from ``configs/attestor.yaml`` ``durable.schedules``
(YAML authoritative, validated as 5-field specs in ``attestor.config``).
``apply`` is create-or-update: a missing schedule is created, an existing
one is updated in place to the declared spec/action, so re-running it
converges and never duplicates. Each action starts its workflow with an
id prefix; Temporal appends the tick timestamp, yielding the plan's
``retention-sweep-<YYYY-MM-DD…>`` / ``session-sweep-<…>`` ids.

Overlap policy is SKIP: a sweep still running when the next tick fires
is not doubled up (the next tick runs it again).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from attestor.durable.config import require_enabled, require_temporalio
from attestor.durable.models import (
    RETENTION_SCHEDULE_ID,
    RETENTION_WORKFLOW,
    RETENTION_WORKFLOW_ID_PREFIX,
    SESSION_SCHEDULE_ID,
    SESSION_WORKFLOW,
    SESSION_WORKFLOW_ID_PREFIX,
    RetentionSweepRequest,
    SessionSweepRequest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from attestor.config import DurableCfg

logger = logging.getLogger("attestor.durable.schedules")

APPLY_CREATED = "created"
APPLY_UPDATED = "updated"


@dataclass(frozen=True)
class ScheduleDecl:
    """One declared schedule (pure; no Temporal types)."""

    schedule_id: str
    workflow: str
    workflow_id_prefix: str
    cron: str
    task_queue: str
    arg: Any
    note: str = ""


@dataclass(frozen=True)
class ScheduleApplyResult:
    schedule_id: str
    cron: str
    workflow: str
    result: str


@dataclass(frozen=True)
class ScheduleView:
    """What ``attestor durable schedules list`` shows per declared schedule."""

    schedule_id: str
    workflow: str
    cron: str
    present: bool
    paused: bool = False
    num_actions: int = 0
    next_action_at: str | None = None


def declared_schedules(cfg: DurableCfg) -> tuple[ScheduleDecl, ...]:
    """The two governance schedules, cron strings from ``cfg.schedules``."""
    return (
        ScheduleDecl(
            schedule_id=RETENTION_SCHEDULE_ID, workflow=RETENTION_WORKFLOW,
            workflow_id_prefix=RETENTION_WORKFLOW_ID_PREFIX,
            cron=cfg.schedules.retention_sweep, task_queue=cfg.task_queue,
            arg=RetentionSweepRequest(), note="apply every enabled retention policy",
        ),
        ScheduleDecl(
            schedule_id=SESSION_SCHEDULE_ID, workflow=SESSION_WORKFLOW,
            workflow_id_prefix=SESSION_WORKFLOW_ID_PREFIX,
            cron=cfg.schedules.session_sweep, task_queue=cfg.task_queue,
            arg=SessionSweepRequest(), note="idle → ended → archived session transitions",
        ),
    )


def summarize(results: tuple[ScheduleApplyResult, ...]) -> Mapping[str, str]:
    return {r.schedule_id: r.result for r in results}


def _to_schedule(decl: ScheduleDecl) -> Any:
    from temporalio.client import (
        Schedule,
        ScheduleActionStartWorkflow,
        ScheduleOverlapPolicy,
        SchedulePolicy,
        ScheduleSpec,
    )

    return Schedule(
        action=ScheduleActionStartWorkflow(
            decl.workflow, decl.arg, id=decl.workflow_id_prefix, task_queue=decl.task_queue,
        ),
        spec=ScheduleSpec(cron_expressions=[decl.cron]),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
    )


async def _apply_one(client: Any, decl: ScheduleDecl) -> ScheduleApplyResult:
    from temporalio.client import ScheduleAlreadyRunningError, ScheduleUpdate

    desired = _to_schedule(decl)
    try:
        await client.create_schedule(decl.schedule_id, desired)
        result = APPLY_CREATED
    except ScheduleAlreadyRunningError:
        await client.get_schedule_handle(decl.schedule_id).update(
            lambda _inp: ScheduleUpdate(schedule=desired),
        )
        result = APPLY_UPDATED
    logger.info("schedule %s %s (cron=%r → %s)", decl.schedule_id, result, decl.cron, decl.workflow)
    return ScheduleApplyResult(
        schedule_id=decl.schedule_id, cron=decl.cron, workflow=decl.workflow, result=result,
    )


async def apply(client: Any, cfg: DurableCfg) -> tuple[ScheduleApplyResult, ...]:
    """Create-or-update every declared schedule; raises loudly when disabled."""
    require_enabled(cfg)
    require_temporalio()
    results: tuple[ScheduleApplyResult, ...] = ()
    for decl in declared_schedules(cfg):
        results = (*results, await _apply_one(client, decl))
    return results


async def _view_one(client: Any, decl: ScheduleDecl) -> ScheduleView:
    from temporalio.service import RPCError

    base = ScheduleView(
        schedule_id=decl.schedule_id, workflow=decl.workflow, cron=decl.cron, present=False,
    )
    try:
        desc = await client.get_schedule_handle(decl.schedule_id).describe()
    except RPCError as exc:
        logger.debug("schedule %s not present: %s", decl.schedule_id, exc)
        return base
    next_times = list(desc.info.next_action_times or ())
    return ScheduleView(
        schedule_id=decl.schedule_id, workflow=decl.workflow, cron=decl.cron, present=True,
        paused=bool(desc.schedule.state.paused), num_actions=int(desc.info.num_actions),
        next_action_at=next_times[0].isoformat() if next_times else None,
    )


async def list_declared(client: Any, cfg: DurableCfg) -> tuple[ScheduleView, ...]:
    """Presence + stats of every declared schedule (never raises for a missing one)."""
    require_enabled(cfg)
    require_temporalio()
    views: tuple[ScheduleView, ...] = ()
    for decl in declared_schedules(cfg):
        views = (*views, await _view_one(client, decl))
    return views
