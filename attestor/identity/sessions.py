"""SessionRepo — CRUD against the ``sessions`` table.

Implements the lifecycle: pending → active → idle → ended → archived
(see tenancy.md §3.1). Idle/ended transitions are time-driven; this repo
exposes the explicit transitions only — a background sweeper handles the
time-driven ones.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional

import psycopg2.extras

from attestor.models import Session


class SessionStateError(Exception):
    """Raised on illegal lifecycle transitions (e.g. write to archived)."""


class SessionRepo:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── Create / autostart ────────────────────────────────────────────────

    def create(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Session:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO sessions (user_id, project_id, title, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (user_id, project_id, title, json.dumps(metadata or {})),
            )
            row = cur.fetchone()
        self._conn.commit()
        return Session.from_row(dict(row))

    def get_or_create_daily(
        self, user_id: str, project_id: str, day: str,
    ) -> Session:
        """SOLO-mode helper: one session per (user, day). ``day`` is an ISO
        date string (e.g. '2026-04-25'). Returns existing if found."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM sessions
                WHERE user_id = %s
                  AND metadata->>'daily_key' = %s
                  AND status = 'active'
                LIMIT 1
                """,
                (user_id, f"solo-daily-{day}"),
            )
            row = cur.fetchone()
            if row:
                return Session.from_row(dict(row))
        return self.create(
            user_id=user_id,
            project_id=project_id,
            title=f"Local — {day}",
            metadata={"daily_key": f"solo-daily-{day}", "created_by": "solo_daily"},
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[Session]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            row = cur.fetchone()
        return Session.from_row(dict(row)) if row else None

    def list_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        status: str = "active",
        limit: int = 20,
        before: Optional[datetime] = None,
    ) -> List[Session]:
        sql = "SELECT * FROM sessions WHERE user_id = %s AND status = %s "
        params: list = [user_id, status]
        if project_id is not None:
            sql += "AND project_id = %s "
            params.append(project_id)
        if before is not None:
            sql += "AND last_active_at < %s "
            params.append(before)
        sql += "ORDER BY last_active_at DESC LIMIT %s"
        params.append(limit)
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [Session.from_row(dict(r)) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def resume(self, session_id: str) -> Optional[Session]:
        """Bump last_active_at to now. Returns updated row, or None if missing."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "UPDATE sessions SET last_active_at = NOW(), status = 'active' "
                "WHERE id = %s AND status != 'archived' "
                "RETURNING *",
                (session_id,),
            )
            row = cur.fetchone()
        self._conn.commit()
        return Session.from_row(dict(row)) if row else None

    def bump_activity(self, session_id: str, message_increment: int = 1) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions "
                "SET last_active_at = NOW(), message_count = message_count + %s "
                "WHERE id = %s",
                (message_increment, session_id),
            )
        self._conn.commit()

    def end(self, session_id: str) -> Optional[Session]:
        """Transition active/idle → ended AND enqueue the session's
        episodes for sleep-time consolidation (Phase 7.4).

        Episodes that have already been consolidated (state='done') are
        re-enqueued — ending a session is a strong "please re-look at
        these with the stronger model" signal.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "UPDATE sessions SET status = 'ended', ended_at = NOW() "
                "WHERE id = %s AND status IN ('active', 'pending') "
                "RETURNING *",
                (session_id,),
            )
            row = cur.fetchone()
        self._conn.commit()
        if row is None:
            return None
        # Enqueue every episode in this session that's done/failed back
        # into pending. Already-pending/processing rows are untouched.
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE episodes "
                    "SET consolidation_state = 'pending', "
                    "    consolidation_claimed_at = NULL, "
                    "    consolidation_done_at = NULL, "
                    "    consolidation_error = NULL "
                    "WHERE session_id = %s "
                    "  AND consolidation_state IN ('done', 'failed')",
                    (session_id,),
                )
            self._conn.commit()
        except Exception:
            # If the episodes table is older (no consolidation_state),
            # silently skip — the lifecycle transition itself succeeded.
            pass
        return Session.from_row(dict(row))

    def archive(self, session_id: str) -> Optional[Session]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "UPDATE sessions SET status = 'archived' WHERE id = %s "
                "RETURNING *",
                (session_id,),
            )
            row = cur.fetchone()
        self._conn.commit()
        return Session.from_row(dict(row)) if row else None

    def set_title(self, session_id: str, title: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET title = %s WHERE id = %s",
                (title, session_id),
            )
        self._conn.commit()

    def assert_writable(self, session_id: str) -> Session:
        """Raises SessionStateError if the session is archived; returns the
        session row otherwise. Use before any write operation."""
        sess = self.get(session_id)
        if sess is None:
            raise SessionStateError(f"session {session_id} not found")
        if sess.status == "archived":
            raise SessionStateError(
                f"session {session_id} is archived; create a new one"
            )
        return sess
