"""EpisodeRepo — verbatim user+assistant rounds in the ``episodes`` table.

Episodes are the immutable audit log: every extracted fact links back
to the episode it came from via ``memories.source_episode_id``. The
table itself is append-only — no UPDATE/DELETE except cascade.

RLS-scoped: writes go through the runtime role's connection, which
must already have the RLS variable set to the writing user's id.
``AgentMemory._resolve()`` does this before calling write paths.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2.extras

from attestor.conversation.turns import ConversationTurn


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Episode:
    """One verbatim conversational round, as persisted in ``episodes``."""

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
    created_at: datetime = field(default_factory=_now_utc)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Episode:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
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
            created_at=row["created_at"],
        )


class EpisodeRepo:
    """Pure data-access for the v4 ``episodes`` table.

    Takes a psycopg2 connection at construction time; caller manages the
    connection lifecycle. RLS variable must be set by the caller before
    any write/read."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── Create ────────────────────────────────────────────────────────────

    def write_round(
        self,
        user_id: str,
        session_id: str,
        user_turn: ConversationTurn,
        assistant_turn: ConversationTurn,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """Persist one verbatim round. Returns the created Episode.

        Validates that both turns share the same thread_id and that the
        user turn precedes the assistant turn in time. Roles must be
        user/assistant respectively (anti-foot-gun).
        """
        if user_turn.thread_id != assistant_turn.thread_id:
            raise ValueError(
                f"thread_id mismatch: user={user_turn.thread_id!r} "
                f"assistant={assistant_turn.thread_id!r}"
            )
        if not user_turn.is_user:
            raise ValueError(
                f"user_turn.role must be 'user'; got {user_turn.role!r}"
            )
        if not assistant_turn.is_assistant:
            raise ValueError(
                f"assistant_turn.role must be 'assistant'; got "
                f"{assistant_turn.role!r}"
            )
        if assistant_turn.ts < user_turn.ts:
            raise ValueError(
                "assistant_turn.ts cannot precede user_turn.ts"
            )

        new_id = str(uuid.uuid4())
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO episodes (
                    id, user_id, project_id, session_id, thread_id,
                    user_turn_text, assistant_turn_text,
                    user_ts, assistant_ts, agent_id, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING *
                """,
                (
                    new_id, user_id, project_id, session_id,
                    user_turn.thread_id,
                    user_turn.content, assistant_turn.content,
                    user_turn.ts, assistant_turn.ts,
                    agent_id,
                    json.dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return Episode.from_row(dict(row))

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, episode_id: str) -> Optional[Episode]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM episodes WHERE id = %s", (episode_id,))
            row = cur.fetchone()
        return Episode.from_row(dict(row)) if row else None

    def list_for_thread(
        self, thread_id: str, limit: int = 50,
    ) -> List[Episode]:
        """Chronological history for a thread (RLS scoped to the caller)."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM episodes WHERE thread_id = %s "
                "ORDER BY user_ts ASC LIMIT %s",
                (thread_id, limit),
            )
            rows = cur.fetchall()
        return [Episode.from_row(dict(r)) for r in rows]

    def list_for_session(
        self, session_id: str, limit: int = 100,
    ) -> List[Episode]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM episodes WHERE session_id = %s "
                "ORDER BY user_ts ASC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [Episode.from_row(dict(r)) for r in rows]

    def count_for_session(self, session_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM episodes WHERE session_id = %s",
                (session_id,),
            )
            return int(cur.fetchone()[0])
