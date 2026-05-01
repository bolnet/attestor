"""Recall context — async-safe per-recall metadata propagation.

Phase 5 of the async retrieval rollout (see
``docs/plans/async-retrieval/PLAN.md``). Provides a contextvar-backed
``recall_started_at`` primitive that activates four audit invariants
under cross-store async reads:

  - **A1**  recall(as_of=X) reproducible under concurrent writes
  - **A2**  no writes that landed after recall start are visible mid-recall
  - **A6**  per-recall metadata (user_id, recall_id, ceiling) propagates
            through ``asyncio.create_task`` / ``asyncio.gather``
  - **A8**  deletion_audit semantics preserved — pre-ceiling deletions
            still log even when the recall reads happen async

Why a contextvar instead of an explicit parameter:

  Threading a ``recall_started_at`` argument through every store
  call (Postgres, Pinecone, Neo4j) and every retrieval helper
  (HyDE, multi-query, BM25, MMR, scorer) would touch ~30 call sites
  and force every caller to know about the ceiling. ``ContextVar``
  propagates automatically across ``asyncio.create_task`` /
  ``asyncio.gather`` and is invisible to callers that don't care.

Usage:

  Sync:
      with rc.recall_started_at_scope():
          mem.recall(query="...")          # orchestrator + stores read
                                            # the ceiling automatically

  Async:
      async with rc.recall_started_at_scope_async():
          await mem.recall_async(query="...")

Reading the ceiling from inside a store backend:

      ceiling = rc.current_recall_started_at()  # → datetime or None
      if ceiling is not None:
          where.append("t_created <= %s")
          params.append(ceiling)
"""

from __future__ import annotations

import contextlib
import contextvars
from datetime import datetime, timezone
from collections.abc import Iterator

# UTC monotonic-anchored timestamp captured at recall start. Set by
# ``recall_started_at_scope``; read by store backends when constructing
# their WHERE clauses. Must use ``datetime.now(timezone.utc)`` (NOT
# ``time.monotonic``) because Postgres ``NOW()`` returns wall-clock UTC
# and the ceiling needs to be comparable across the wire.
_RECALL_STARTED_AT: contextvars.ContextVar[datetime | None] = (
    contextvars.ContextVar("attestor.recall.started_at", default=None)
)


def current_recall_started_at() -> datetime | None:
    """Return the active ``recall_started_at`` ceiling, or ``None``
    when no recall scope is open. Stores call this from their
    ``search()`` / ``list_memories()`` paths to AND a ``t_created
    <= ceiling`` filter into their query when the value is set.

    Returning ``None`` outside a recall scope is the explicit
    backwards-compat path — pre-Phase-5 callers that don't open a
    scope keep working unchanged.
    """
    return _RECALL_STARTED_AT.get()


@contextlib.contextmanager
def recall_started_at_scope(
    started_at: datetime | None = None,
) -> Iterator[datetime]:
    """Enter a recall scope. All store reads inside (sync or async,
    on this coroutine or any spawned tasks) see the same ``started_at``
    timestamp via ``current_recall_started_at()``.

    Args:
        started_at: Optional explicit timestamp. Default = ``NOW()``
            in UTC. The explicit argument exists for tests and for
            future ``recall(as_of=X)`` callers who want to force a
            specific ceiling.

    Yields the active ceiling so callers that want to log it (e.g.
    trace events) have direct access without re-reading the contextvar.
    """
    ts = started_at or datetime.now(timezone.utc)
    token = _RECALL_STARTED_AT.set(ts)
    try:
        yield ts
    finally:
        _RECALL_STARTED_AT.reset(token)


# Async sibling provided as a no-op alias so callers can use the same
# context manager from async code without ceremony. Python's ``with``
# already works fine inside ``async def``; the async-flavored helper
# below is only useful when callers want ``async with``.
@contextlib.asynccontextmanager
async def recall_started_at_scope_async(
    started_at: datetime | None = None,
) -> Iterator[datetime]:
    """Async-flavored ``recall_started_at_scope``. Identical semantics;
    contextvars propagate the same way. Provided so ``async with`` reads
    naturally in async call sites."""
    ts = started_at or datetime.now(timezone.utc)
    token = _RECALL_STARTED_AT.set(ts)
    try:
        yield ts
    finally:
        _RECALL_STARTED_AT.reset(token)
