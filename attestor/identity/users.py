"""UserRepo — CRUD against the ``users`` table.

Race-safe first-time provisioning via INSERT ... ON CONFLICT. Soft delete
for GDPR retention windows; hard ``purge`` for permanent removal across all
backing stores (graph + vector cleanup is the caller's responsibility).
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import psycopg2.extras

from attestor.models import User


class UserRepo:
    def __init__(self, conn: Any) -> None:
        """conn: a psycopg2 connection. Caller manages connection lifecycle.
        For user creation the connection should be admin/RLS-bypassing
        (or the caller must SET attestor.current_user_id appropriately)."""
        self._conn = conn

    # ── Create / first-login provisioning ────────────────────────────────

    def create_or_get(
        self,
        external_id: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> User:
        """Race-safe first-time provisioning.

        Returns the existing user if one with this external_id already exists,
        otherwise creates and returns a new user. Uses INSERT ... ON CONFLICT
        DO UPDATE so two concurrent first-time logins don't create duplicates.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO users (external_id, email, display_name, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (external_id) DO UPDATE
                  SET email = COALESCE(EXCLUDED.email, users.email),
                      display_name = COALESCE(EXCLUDED.display_name, users.display_name)
                RETURNING *
                """,
                (
                    external_id,
                    email,
                    display_name,
                    json.dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return User.from_row(dict(row))

    def create(
        self,
        external_id: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> User:
        """Strict create — raises IntegrityError if external_id already exists."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO users (external_id, email, display_name, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (external_id, email, display_name, json.dumps(metadata or {})),
            )
            row = cur.fetchone()
        self._conn.commit()
        return User.from_row(dict(row))

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> Optional[User]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        return User.from_row(dict(row)) if row else None

    def find_by_external_id(self, external_id: str) -> Optional[User]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE external_id = %s AND status = 'active'",
                (external_id,),
            )
            row = cur.fetchone()
        return User.from_row(dict(row)) if row else None

    def list_active(self, limit: int = 100) -> List[User]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE status = 'active' "
                "ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [User.from_row(dict(r)) for r in rows]

    # ── Soft delete ───────────────────────────────────────────────────────

    def soft_delete(self, user_id: str) -> bool:
        """Mark user as deleted; data remains for retention window. Returns
        True if a row was affected, False if user didn't exist."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET status = 'deleted', deleted_at = NOW() "
                "WHERE id = %s AND status != 'deleted'",
                (user_id,),
            )
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0

    # ── Hard delete ───────────────────────────────────────────────────────

    def purge(self, user_id: str) -> bool:
        """DELETE the user row. CASCADE removes their projects, sessions,
        episodes, and memories. Caller must clean up graph + vector stores
        separately (those don't CASCADE). Returns True if the user existed."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0
