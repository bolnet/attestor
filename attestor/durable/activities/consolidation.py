# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``consolidate_episode`` activity — wraps ``SleepTimeConsolidator.consolidate_claimed``.

Activity side = I/O side. The consolidator (AgentMemory + Postgres +
LLM clients) is built lazily by the injected provider on first call, so
constructing the activity set never opens a connection.

Payload contract: the workflow hands over an ``EpisodeRef`` (ids only).
The conversation text is re-read from the claimed Postgres row here —
it never appears in Temporal's event history (plan §2).

Failure contract:
  * ``ConsolidationResult.error`` set → ``ApplicationError`` so Temporal
    applies the workflow's ``RetryPolicy``.
  * errors matching ``NON_RETRYABLE_ERROR_PREFIXES`` → ``non_retryable``:
    ``rls:`` (a permission / config problem) and ``missing:`` (the row
    is not claimed for that user — retrying cannot make it appear).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from attestor.durable.activities._sanitize import sanitize_error_message
from attestor.durable.models import (
    CONSOLIDATE_EPISODE_ACTIVITY,
    ConsolidationOutcome,
    EpisodeRef,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from attestor.consolidation.consolidator import ConsolidationResult

logger = logging.getLogger("attestor.durable.activities.consolidation")

ERROR_TYPE_TRANSIENT = "ConsolidationFailed"
ERROR_TYPE_PERMANENT = "ConsolidationPermanent"
NON_RETRYABLE_ERROR_PREFIXES: tuple[str, ...] = ("rls:", "missing:")


def outcome_from_result(result: ConsolidationResult) -> ConsolidationOutcome:
    return ConsolidationOutcome(
        episode_id=result.episode_id,
        ok=result.ok,
        error=result.error,
        written_memory_ids=tuple(result.written_memory_ids),
        user_fact_count=len(result.user_facts),
        agent_fact_count=len(result.agent_facts),
    )


def is_permanent_error(error: str) -> bool:
    return error.startswith(NON_RETRYABLE_ERROR_PREFIXES)


class ConsolidationActivities:
    """Activity set bound to a lazily-built ``SleepTimeConsolidator``."""

    def __init__(self, consolidator_provider: Callable[[], Any]) -> None:
        self._provider = consolidator_provider

    def _get(self) -> Any:
        # Always ask the provider (see worker._memory_provider rebuild).
        return self._provider()

    @activity.defn(name=CONSOLIDATE_EPISODE_ACTIVITY)
    def consolidate_episode(self, ref: EpisodeRef) -> ConsolidationOutcome:
        result: ConsolidationResult = self._get().consolidate_claimed(
            ref.episode_id, ref.user_id,
        )
        outcome = outcome_from_result(result)
        if outcome.ok:
            return outcome
        error = sanitize_error_message(outcome.error or "consolidation failed")
        permanent = is_permanent_error(error)
        logger.warning(
            "consolidate_episode %s failed (%s): %s",
            ref.episode_id, "permanent" if permanent else "retryable", error,
        )
        raise ApplicationError(
            error,
            type=ERROR_TYPE_PERMANENT if permanent else ERROR_TYPE_TRANSIENT,
            non_retryable=permanent,
        )
