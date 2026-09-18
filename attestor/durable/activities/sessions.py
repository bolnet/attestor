# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``SessionSweep`` activities — the three time-driven lifecycle transitions.

Each wraps one pure-SQL batch query on ``SessionRepo`` (``sweep_idle`` /
``sweep_ended`` / ``sweep_archived``) and returns a count. Thresholds
arrive in minutes on the request and are turned into an interval here.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from temporalio import activity

from attestor.compliance.forget_lanes import connection_of
from attestor.durable.activities._errors import raise_for
from attestor.durable.models import (
    SESSION_TRANSITION_ARCHIVED,
    SESSION_TRANSITION_ENDED,
    SESSION_TRANSITION_IDLE,
    SWEEP_ARCHIVED_ACTIVITY,
    SWEEP_ENDED_ACTIVITY,
    SWEEP_IDLE_ACTIVITY,
    SessionSweepRequest,
    SessionSweepStep,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("attestor.durable.activities.sessions")

ERROR_TYPE_TRANSIENT = "SessionSweepFailed"
ERROR_TYPE_PERMANENT = "SessionSweepPermanent"


class SessionActivities:
    """Activity set bound to a lazily-built ``AgentMemory``."""

    def __init__(self, memory_provider: Callable[[], Any]) -> None:
        self._provider = memory_provider

    def _get_repo(self) -> Any:
        from attestor.identity.sessions import SessionRepo

        # Rebuilt per call so a memory replaced by the provider is honoured.
        return SessionRepo(connection_of(self._provider()))

    def _step(self, transition: str, minutes: int, limit: int) -> SessionSweepStep:
        sweep = {
            SESSION_TRANSITION_IDLE: lambda r: r.sweep_idle,
            SESSION_TRANSITION_ENDED: lambda r: r.sweep_ended,
            SESSION_TRANSITION_ARCHIVED: lambda r: r.sweep_archived,
        }[transition]
        try:
            ids = sweep(self._get_repo())(timedelta(minutes=minutes), limit=limit)
        except Exception as exc:
            raise_for(
                exc, subject=f"after={minutes}m limit={limit}", step=f"sweep_{transition}",
                transient_type=ERROR_TYPE_TRANSIENT, permanent_type=ERROR_TYPE_PERMANENT,
                logger=logger,
            )
        return SessionSweepStep(transition=transition, count=len(ids))

    @activity.defn(name=SWEEP_IDLE_ACTIVITY)
    def sweep_idle(self, req: SessionSweepRequest) -> SessionSweepStep:
        return self._step(SESSION_TRANSITION_IDLE, req.idle_after_minutes, req.batch_limit)

    @activity.defn(name=SWEEP_ENDED_ACTIVITY)
    def sweep_ended(self, req: SessionSweepRequest) -> SessionSweepStep:
        return self._step(SESSION_TRANSITION_ENDED, req.ended_after_minutes, req.batch_limit)

    @activity.defn(name=SWEEP_ARCHIVED_ACTIVITY)
    def sweep_archived(self, req: SessionSweepRequest) -> SessionSweepStep:
        return self._step(
            SESSION_TRANSITION_ARCHIVED, req.archived_after_minutes, req.batch_limit,
        )
