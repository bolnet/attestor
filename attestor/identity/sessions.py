"""SessionRepo — CRUD against the ``sessions`` table.

Implements the lifecycle: pending → active → idle → ended → archived
(see tenancy.md §3.1). Idle/ended transitions are time-driven; this repo
exposes the explicit transitions only — a background sweeper handles the
time-driven ones.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

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
        project_id: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
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
        date string (e.g. '2026-04-25'). Returns existing if found.

        Race-safe under parallel callers: serializes the SELECT-then-
        INSERT critical section via a Postgres transactional advisory
        lock keyed on (user_id, day). Without this, two parallel LME
        samples sharing the SOLO user (the common case) would both see
        no daily session and both INSERT — silent duplicate sessions
        for the same day.

        The lock is transactional (auto-releases on commit/rollback)
        so we don't leak locks even if an exception fires mid-flight.
        Schema has no UNIQUE constraint on (user_id, daily_key) so the
        advisory lock IS the only protection here.
        """
        # Hash the (user, day) key into a 32-bit lock id. abs() so the
        # value fits the postgres int4 range; collision risk is
        # negligible at this volume (lock space is 2**31).
        import hashlib
        digest = hashlib.sha256(
            f"daily:{user_id}:{day}".encode(),
        ).digest()
        lock_id = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Acquire the transactional advisory lock — blocks if
            # another thread already holds it for this (user, day).
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))
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
                self._conn.commit()  # release advisory lock
                return Session.from_row(dict(row))

            # Still inside the lock — INSERT now is exclusive for this
            # (user, day) pair. Other threads block on the lock until
            # we commit.
            cur.execute(
                """
                INSERT INTO sessions (user_id, project_id, title, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (
                    user_id,
                    project_id,
                    f"Local — {day}",
                    json.dumps({
                        "daily_key": f"solo-daily-{day}",
                        "created_by": "solo_daily",
                    }),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()  # releases advisory lock
        return Session.from_row(dict(row))

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Session | None:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            row = cur.fetchone()
        return Session.from_row(dict(row)) if row else None

    def list_for_user(
        self,
        user_id: str,
        project_id: str | None = None,
        status: str = "active",
        limit: int = 20,
        before: datetime | None = None,
    ) -> list[Session]:
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

    def resume(self, session_id: str) -> Session | None:
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

    def end(self, session_id: str) -> Session | None:
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

    def archive(self, session_id: str) -> Session | None:
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
