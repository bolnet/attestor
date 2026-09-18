# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``embed_and_upsert`` / ``graph_extract`` / ``list_memory_ids`` activities.

Activity side = I/O side. Every call re-reads the memory row from
Postgres by id through ``AgentMemory.derive_vector`` /
``derive_graph`` (``attestor.core.derive_service``) — the same lanes
``add()`` uses — so the Temporal payload is ids only (plan §2) and the
derived state is byte-identical to a synchronous write.

Tenant scope: ``DeriveRef.user_id`` / ``agent_id`` and
``ListMemoryIdsRequest.user_id`` are handed straight to the seam, which
sets the Postgres RLS session user before every re-read / write.

Failure contract:
  * ``MemoryNotFoundError`` → ``DeriveMissing``, non-retryable.
  * ``RebuildScopeError`` / ``RebuildUnsupportedError`` (no tenant scope
    on a multi-tenant store; store cannot list ids) → ``DerivePermanent``.
  * dimension-mismatch / auth / permission errors (see
    ``PERMANENT_ERROR_PATTERNS``) → ``DerivePermanent``, non-retryable.
  * anything else (store down, timeout, 5xx) → ``DeriveFailed``,
    retried per the workflow's ``RetryPolicy`` (unlimited attempts).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from attestor.core.derive_service import (
    MemoryNotFoundError,
    RebuildScopeError,
    RebuildUnsupportedError,
)
from attestor.durable.activities._sanitize import sanitize_error_message
from attestor.durable.models import (
    DERIVE_STEP_GRAPH,
    DERIVE_STEP_VECTOR,
    EMBED_AND_UPSERT_ACTIVITY,
    GRAPH_EXTRACT_ACTIVITY,
    LIST_MEMORY_IDS_ACTIVITY,
    DeriveRef,
    DeriveStepOutcome,
    ListMemoryIdsRequest,
    MemoryIdPage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("attestor.durable.activities.derive")

ERROR_TYPE_TRANSIENT = "DeriveFailed"
ERROR_TYPE_PERMANENT = "DerivePermanent"
ERROR_TYPE_MISSING = "DeriveMissing"
LIST_STEP = "list_memory_ids"

# Matched (case-insensitively) against ``"<ExceptionType>: <message>"``.
# Word boundaries keep HTTP status codes from matching inside uuids.
PERMANENT_ERROR_PATTERNS: tuple[str, ...] = (
    r"dimension",                # pgvector / Pinecone dim mismatch
    r"dim mismatch",
    r"embedderdimmismatch",      # attestor.store.embedder_dim_check
    r"unauthori[sz]ed",
    r"authentication",
    r"\bautherror\b",
    r"forbidden",
    r"invalid[ _]api[ _]key",
    r"permission denied",
    r"invalid input syntax",  # malformed id / user against a typed column
    r"\b401\b",
    r"\b403\b",
)
_PERMANENT_RE = re.compile("|".join(PERMANENT_ERROR_PATTERNS), re.IGNORECASE)


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def is_permanent_derive_error(exc: BaseException) -> bool:
    """True for errors a retry cannot fix (bad dim, bad credentials)."""
    return _PERMANENT_RE.search(describe_error(exc)) is not None


PERMANENT_ERROR_TYPES: tuple[type[BaseException], ...] = (
    RebuildScopeError, RebuildUnsupportedError,
)


def _classify(exc: BaseException) -> tuple[str, bool]:
    """``(error_type, non_retryable)`` for any exception from the derive seam."""
    if isinstance(exc, MemoryNotFoundError):
        return ERROR_TYPE_MISSING, True
    if isinstance(exc, PERMANENT_ERROR_TYPES) or is_permanent_derive_error(exc):
        return ERROR_TYPE_PERMANENT, True
    return ERROR_TYPE_TRANSIENT, False


def _raise_for(exc: BaseException, subject: str, step: str) -> None:
    """Sanitised message, no cause chain: both land in durable workflow history."""
    error_type, permanent = _classify(exc)
    message = sanitize_error_message(describe_error(exc))
    logger.warning(
        "%s %s failed (%s): %s",
        step, subject, "permanent" if permanent else "retryable", message,
    )
    raise ApplicationError(message, type=error_type, non_retryable=permanent) from None


def _heartbeat(*details: Any) -> None:
    if activity.in_activity():
        activity.heartbeat(*details)


class DeriveActivities:
    """Activity set bound to a lazily-built ``AgentMemory``."""

    def __init__(self, memory_provider: Callable[[], Any]) -> None:
        self._provider = memory_provider

    def _get(self) -> Any:
        # Always ask the provider: it caches the healthy instance and
        # rebuilds one whose backends failed to initialise (worker.py).
        return self._provider()

    @activity.defn(name=EMBED_AND_UPSERT_ACTIVITY)
    def embed_and_upsert(self, ref: DeriveRef) -> DeriveStepOutcome:
        try:
            derived = self._get().derive_vector(
                ref.memory_id, user_id=ref.user_id, agent_id=ref.agent_id,
            )
        except Exception as exc:
            _raise_for(exc, ref.memory_id, DERIVE_STEP_VECTOR)
        return DeriveStepOutcome(memory_id=derived.memory_id, step=DERIVE_STEP_VECTOR, ok=True)

    @activity.defn(name=GRAPH_EXTRACT_ACTIVITY)
    def graph_extract(self, ref: DeriveRef) -> DeriveStepOutcome:
        try:
            derived = self._get().derive_graph(
                ref.memory_id, user_id=ref.user_id, agent_id=ref.agent_id,
            )
        except Exception as exc:
            _raise_for(exc, ref.memory_id, DERIVE_STEP_GRAPH)
        return DeriveStepOutcome(
            memory_id=derived.memory_id, step=DERIVE_STEP_GRAPH, ok=True,
            entity_count=derived.entity_count, relation_count=derived.relation_count,
        )

    @activity.defn(name=LIST_MEMORY_IDS_ACTIVITY)
    def list_memory_ids(self, req: ListMemoryIdsRequest) -> MemoryIdPage:
        _heartbeat(req.after_id)
        try:
            page = self._get().list_derivable_ids(
                user_id=req.user_id, since=req.since, namespace=req.namespace,
                after_id=req.after_id, limit=req.limit,
            )
        except Exception as exc:
            _raise_for(exc, f"page after={req.after_id!r} user={req.user_id!r}", LIST_STEP)
        _heartbeat(page.next_after_id)
        return MemoryIdPage(ids=page.ids, next_after_id=page.next_after_id, user_id=page.user_id)
