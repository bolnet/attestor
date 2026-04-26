"""Episode consolidation queue (Phase 7.1, roadmap §E.1).

Backed by the ``consolidation_state`` columns on the v4 ``episodes``
table. State machine:

    pending --(claim)--> processing --(done|failed)--> done|failed
       ^                       |
       +---(reclaim_stale)-----+

``dequeue_batch`` claims rows atomically with FOR UPDATE SKIP LOCKED so
multiple workers can run in parallel without coordination. A row that
was claimed but never finished gets reclaimed after
``queue_lock_seconds`` (default 600) so a crashed worker doesn't strand
work forever.

RLS scoping: this queue is operated by an admin-bypass connection in
production (the worker isn't tied to a single user). For tests, an admin
fixture sets the var to whatever user owns the rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2.extras

logger = logging.getLogger("attestor.consolidation.queue")


@dataclass(frozen=True)
class QueuedEpisode:
    """A row claimed from the queue. The worker holds the lease until
    it calls ``mark_done`` or ``mark_failed``."""
    id: str
    user_id: str
    session_id: str
    thread_id: str
    user_turn_text: str
    assistant_turn_text: str
    user_ts: datetime
    assistant_ts: datetime
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> QueuedEpisode:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            import json
            meta = json.loads(meta)
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            project_id=str(row["project_id"]) if row.get("project_id") else None,
            session_id=str(row["session_id"]),
            thread_id=row["thread_id"],
            user_turn_text=row["user_turn_text"],
            assistant_turn_text=row["assistant_turn_text"],
            user_ts=row["user_ts"],
            assistant_ts=row["assistant_ts"],
            agent_id=row.get("agent_id"),
            metadata=meta,
        )


class ConsolidationQueue:
    """Pure data-access for the episode consolidation queue.

    Takes a psycopg2 connection at construction; does not own its
    lifecycle. Methods are short SQL transactions — no LLM calls, no
    business logic.
    """

    def __init__(
        self,
        conn: Any,
        *,
        queue_lock_seconds: int = 600,
    ) -> None:
        self._conn = conn
        self._lock_seconds = queue_lock_seconds

    # ── Enqueue ──────────────────────────────────────────────────────────

    def enqueue(self, episode_id: str) -> bool:
        """Mark an episode as pending consolidation. Idempotent — a
        row already pending or processing isn't reset.

        Returns True if the row transitioned to pending (newly enqueued
        or re-enqueued from done/failed); False if no change.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE episodes "
                "SET consolidation_state = 'pending', "
                "    consolidation_claimed_at = NULL, "
                "    consolidation_done_at = NULL, "
                "    consolidation_error = NULL "
                "WHERE id = %s "
                "  AND consolidation_state IN ('done', 'failed')",
                (episode_id,),
            )
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0

    # ── Dequeue ──────────────────────────────────────────────────────────

    def dequeue_batch(self, limit: int = 20) -> List[QueuedEpisode]:
        """Atomically claim up to ``limit`` pending rows. Other workers
        skip locked rows so this is safe under concurrency.

        Reclaim policy: rows stuck in 'processing' for longer than
        ``queue_lock_seconds`` are also picked up — assumed crashed.
        """
        with self._conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor,
        ) as cur:
            cur.execute(
                """
                WITH claimed AS (
                    SELECT id FROM episodes
                    WHERE consolidation_state = 'pending'
                       OR (consolidation_state = 'processing'
                           AND consolidation_claimed_at < NOW() - INTERVAL '%s seconds')
                    ORDER BY created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                UPDATE episodes e
                SET consolidation_state = 'processing',
                    consolidation_claimed_at = NOW()
                FROM claimed
                WHERE e.id = claimed.id
                RETURNING e.*
                """,
                (self._lock_seconds, limit),
            )
            rows = cur.fetchall()
        self._conn.commit()
        return [QueuedEpisode.from_row(dict(r)) for r in rows]

    # ── Lifecycle ────────────────────────────────────────────────────────

    def mark_done(self, episode_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE episodes SET consolidation_state = 'done', "
                "consolidation_done_at = NOW(), consolidation_error = NULL "
                "WHERE id = %s",
                (episode_id,),
            )
        self._conn.commit()

    def mark_failed(self, episode_id: str, error: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE episodes SET consolidation_state = 'failed', "
                "consolidation_done_at = NOW(), consolidation_error = %s "
                "WHERE id = %s",
                (error[:1000], episode_id),
            )
        self._conn.commit()

    def release(self, episode_id: str) -> None:
        """Drop a 'processing' lease back to 'pending' without marking
        done/failed. Used when a worker shuts down cleanly mid-job."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE episodes SET consolidation_state = 'pending', "
                "consolidation_claimed_at = NULL "
                "WHERE id = %s AND consolidation_state = 'processing'",
                (episode_id,),
            )
        self._conn.commit()

    # ── Introspection ────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Count rows by consolidation_state. Useful for dashboards."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT consolidation_state, COUNT(*) FROM episodes "
                "GROUP BY consolidation_state"
            )
            rows = cur.fetchall()
        out: Dict[str, int] = {
            "pending": 0, "processing": 0, "done": 0, "failed": 0,
        }
        for state, n in rows:
            out[state] = int(n)
        return out

    def pending_count(self) -> int:
        return self.stats()["pending"]
