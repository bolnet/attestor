"""UserRepo — CRUD against the ``users`` table.

Race-safe first-time provisioning via INSERT ... ON CONFLICT. Soft delete
for GDPR retention windows; hard ``purge`` for permanent removal across all
backing stores (graph + vector cleanup is the caller's responsibility).

RLS-aware: the create paths generate the user's UUID client-side and SET
the connection-local RLS variable to it BEFORE the INSERT. This lets the
RETURNING clause's implicit SELECT pass the policy on first creation, and
leaves the connection scoped to the new user for subsequent operations.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

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
        email: str | None = None,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> User:
        """Race-safe first-time provisioning.

        Returns the existing user if one with this external_id already exists,
        otherwise creates and returns a new user.

        RLS dance:
          1. Use SECURITY DEFINER helper to look up the id by external_id
             without needing the RLS var pre-set (chicken-and-egg).
          2. If found, set RLS var to that id, fetch the row, return.
          3. If not found, generate a new UUID, set RLS var to it, INSERT.
             Use ON CONFLICT DO NOTHING for the race where two concurrent
             callers reach step 3 — the loser falls back to step 2 on retry.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Step 1: bypass RLS to discover the id, if any
            cur.execute(
                "SELECT attestor_user_id_for_external(%s) AS id",
                (external_id,),
            )
            existing_id = cur.fetchone()["id"]

            if existing_id is not None:
                # Step 2: set var, fetch row through normal RLS path
                cur.execute(
                    "SELECT set_config('attestor.current_user_id', %s, false)",
                    (str(existing_id),),
                )
                cur.execute("SELECT * FROM users WHERE id = %s", (str(existing_id),))
                row = cur.fetchone()
                self._conn.commit()
                return User.from_row(dict(row))

            # Step 3: create new user with client-side UUID
            new_id = str(uuid.uuid4())
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (new_id,),
            )
            cur.execute(
                """
                INSERT INTO users (id, external_id, email, display_name, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (external_id) DO NOTHING
                RETURNING *
                """,
                (new_id, external_id, email, display_name,
                 json.dumps(metadata or {})),
            )
            row = cur.fetchone()
            if row is None:
                # Lost the race: another caller created the user between
                # our lookup and INSERT. Re-resolve through the helper.
                cur.execute(
                    "SELECT attestor_user_id_for_external(%s) AS id",
                    (external_id,),
                )
                actual_id = str(cur.fetchone()["id"])
                cur.execute(
                    "SELECT set_config('attestor.current_user_id', %s, false)",
                    (actual_id,),
                )
                cur.execute("SELECT * FROM users WHERE id = %s", (actual_id,))
                row = cur.fetchone()
        self._conn.commit()
        return User.from_row(dict(row))

    def create(
        self,
        external_id: str,
        email: str | None = None,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> User:
        """Strict create — raises IntegrityError if external_id already exists.

        Same RLS dance as ``create_or_get``: client-side UUID + pre-SET so
        RETURNING admits the new row."""
        new_id = str(uuid.uuid4())
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (new_id,),
            )
            cur.execute(
                """
                INSERT INTO users (id, external_id, email, display_name, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (new_id, external_id, email, display_name,
                 json.dumps(metadata or {})),
            )
            row = cur.fetchone()
        self._conn.commit()
        return User.from_row(dict(row))

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> User | None:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        return User.from_row(dict(row)) if row else None

    def find_by_external_id(self, external_id: str) -> User | None:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE external_id = %s AND status = 'active'",
                (external_id,),
            )
            row = cur.fetchone()
        return User.from_row(dict(row)) if row else None

    def list_active(self, limit: int = 100) -> list[User]:
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
