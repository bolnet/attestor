"""User quotas + enforcement (Phase 8.3, roadmap §F).

Per-user limits that apply at the application boundary (add() /
start_session() / create_project()). Counters live in ``user_quotas``
and are maintained by Postgres triggers, so a quota check is one
SELECT, not a COUNT scan.

Limits (all NULL = unlimited):
  max_memories       — total active+superseded rows in memories
  max_sessions       — total session rows
  max_projects       — total project rows (including Inbox)
  max_writes_per_day — INSERT count rolling over at midnight UTC

Threat model:
  - Stops accidental runaway: bug in the agent loop, infinite ingest.
  - Soft DoS protection in HOSTED multi-tenant deployments.
  - NOT a security boundary: an attacker with raw DB access bypasses
    this entirely. Pair with rate limiting at the API layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psycopg2.extras

logger = logging.getLogger("attestor.quotas")


class QuotaExceeded(Exception):
    """Raised by the enforcement layer when a write would exceed a limit.

    The HTTP API maps this to 429 Too Many Requests with the field
    name in the body so clients can show a useful error.
    """

    def __init__(self, field: str, limit: int, current: int) -> None:
        self.field = field
        self.limit = limit
        self.current = current
        super().__init__(
            f"quota exceeded: {field} (limit={limit}, current={current})"
        )


@dataclass(frozen=True)
class UserQuota:
    """One user's quota row. None on a limit means unlimited."""
    user_id: str
    max_memories: Optional[int]
    max_sessions: Optional[int]
    max_projects: Optional[int]
    max_writes_per_day: Optional[int]
    memory_count: int
    session_count: int
    project_count: int
    writes_today: int

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> UserQuota:
        return cls(
            user_id=str(row["user_id"]),
            max_memories=row.get("max_memories"),
            max_sessions=row.get("max_sessions"),
            max_projects=row.get("max_projects"),
            max_writes_per_day=row.get("max_writes_per_day"),
            memory_count=int(row.get("memory_count") or 0),
            session_count=int(row.get("session_count") or 0),
            project_count=int(row.get("project_count") or 0),
            writes_today=int(row.get("writes_today") or 0),
        )

    def remaining_writes_today(self) -> Optional[int]:
        if self.max_writes_per_day is None:
            return None
        return max(0, self.max_writes_per_day - self.writes_today)


class QuotaRepo:
    """Data-access for user_quotas. Pure SQL, no business logic.

    The trigger in schema.sql auto-creates a row when a user is created;
    callers don't need to insert manually.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def get(self, user_id: str) -> Optional[UserQuota]:
        with self._conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor,
        ) as cur:
            cur.execute(
                "SELECT * FROM user_quotas WHERE user_id = %s", (user_id,),
            )
            row = cur.fetchone()
        return UserQuota.from_row(dict(row)) if row else None

    def set_limits(
        self,
        user_id: str,
        *,
        max_memories: Optional[int] = None,
        max_sessions: Optional[int] = None,
        max_projects: Optional[int] = None,
        max_writes_per_day: Optional[int] = None,
    ) -> UserQuota:
        """Update one or more limits. Counters are NOT touched."""
        with self._conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor,
        ) as cur:
            cur.execute(
                """
                INSERT INTO user_quotas (
                    user_id, max_memories, max_sessions, max_projects,
                    max_writes_per_day
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                  SET max_memories       = COALESCE(EXCLUDED.max_memories,
                                                    user_quotas.max_memories),
                      max_sessions       = COALESCE(EXCLUDED.max_sessions,
                                                    user_quotas.max_sessions),
                      max_projects       = COALESCE(EXCLUDED.max_projects,
                                                    user_quotas.max_projects),
                      max_writes_per_day = COALESCE(EXCLUDED.max_writes_per_day,
                                                    user_quotas.max_writes_per_day),
                      updated_at         = NOW()
                RETURNING *
                """,
                (user_id, max_memories, max_sessions, max_projects,
                 max_writes_per_day),
            )
            row = cur.fetchone()
        self._conn.commit()
        return UserQuota.from_row(dict(row))

    # ── Enforcement (called from AgentMemory before each write) ─────────

    def check_memory_quota(self, user_id: str) -> None:
        """Raises QuotaExceeded if adding one memory would breach a limit."""
        q = self.get(user_id)
        if q is None:
            return  # No quota row → no limits
        if q.max_memories is not None and q.memory_count >= q.max_memories:
            raise QuotaExceeded(
                "max_memories", q.max_memories, q.memory_count,
            )
        if (
            q.max_writes_per_day is not None
            and q.writes_today >= q.max_writes_per_day
        ):
            raise QuotaExceeded(
                "max_writes_per_day", q.max_writes_per_day, q.writes_today,
            )

    def check_session_quota(self, user_id: str) -> None:
        q = self.get(user_id)
        if q is None:
            return
        if q.max_sessions is not None and q.session_count >= q.max_sessions:
            raise QuotaExceeded(
                "max_sessions", q.max_sessions, q.session_count,
            )

    def check_project_quota(self, user_id: str) -> None:
        q = self.get(user_id)
        if q is None:
            return
        if q.max_projects is not None and q.project_count >= q.max_projects:
            raise QuotaExceeded(
                "max_projects", q.max_projects, q.project_count,
            )
