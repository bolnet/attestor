"""ProjectRepo — CRUD against the ``projects`` table.

The "Inbox" is a real project with ``metadata.is_inbox = true`` (per
defaults.md §3.3 — treat as a normal project with a flag, not a special
NULL/sentinel case). Inbox cannot be deleted or archived.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import psycopg2.extras

from attestor.models import Project

INBOX_NAME = "Inbox"
INBOX_FLAG = "is_inbox"


class InboxImmutableError(Exception):
    """Raised on attempts to delete or archive a user's Inbox."""


class ProjectRepo:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── Create ────────────────────────────────────────────────────────────

    def create(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Project:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO projects (user_id, name, description, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (user_id, name, description, json.dumps(metadata or {})),
            )
            row = cur.fetchone()
        self._conn.commit()
        return Project.from_row(dict(row))

    def ensure_inbox(self, user_id: str) -> Project:
        """Idempotent: returns the user's Inbox, creating on first call."""
        existing = self.find_by_name(user_id, INBOX_NAME)
        if existing is not None:
            return existing
        return self.create(
            user_id=user_id,
            name=INBOX_NAME,
            description="Conversations not assigned to a project",
            metadata={INBOX_FLAG: True, "created_by": "auto"},
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, project_id: str) -> Optional[Project]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
        return Project.from_row(dict(row)) if row else None

    def find_by_name(self, user_id: str, name: str) -> Optional[Project]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM projects WHERE user_id = %s AND name = %s",
                (user_id, name),
            )
            row = cur.fetchone()
        return Project.from_row(dict(row)) if row else None

    def list_for_user(
        self, user_id: str, include_inbox: bool = False, limit: int = 100,
    ) -> List[Project]:
        sql = (
            "SELECT * FROM projects "
            "WHERE user_id = %s AND status = 'active' "
        )
        params: list = [user_id]
        if not include_inbox:
            sql += "AND COALESCE(metadata->>'is_inbox', 'false') = 'false' "
        sql += "ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [Project.from_row(dict(r)) for r in rows]

    # ── Archive / Delete (Inbox protected) ────────────────────────────────

    def _ensure_not_inbox(self, project: Project) -> None:
        if project.is_inbox:
            raise InboxImmutableError(
                f"Cannot modify Inbox project (id={project.id}); it's the "
                "default destination for new sessions and must always exist."
            )

    def archive(self, project_id: str) -> bool:
        proj = self.get(project_id)
        if proj is None:
            return False
        self._ensure_not_inbox(proj)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET status = 'archived', archived_at = NOW() "
                "WHERE id = %s AND status = 'active'",
                (project_id,),
            )
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0

    def delete(self, project_id: str) -> bool:
        """Hard delete. Inbox protected. CASCADE removes referenced sessions
        (project_id is set to NULL on sessions because of ON DELETE SET NULL)."""
        proj = self.get(project_id)
        if proj is None:
            return False
        self._ensure_not_inbox(proj)
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0
