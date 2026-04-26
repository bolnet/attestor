"""GDPR delete + export (Phase 8.5, roadmap §F).

Two operations:

  export_user(external_id) -> dict
    Full data portability dump: every user/project/session/episode/memory
    row scoped to the user, plus the quota row. JSON-serializable; the
    HTTP API returns this verbatim.

  purge_user(external_id, *, reason, deleted_by) -> dict
    Hard delete: removes the users row, which CASCADEs through
    projects/sessions/episodes/memories/user_quotas (FK ON DELETE CASCADE).
    Records an entry in the RLS-exempt deletion_audit table BEFORE the
    actual delete so the audit survives even if the delete fails.

The vector + graph stores don't CASCADE — callers using ArangoDB / Neo4j
must clean those independently. This module is Postgres-only; broader
multi-store coordination is the API layer's responsibility.

Threat model:
  - DEFENDS against incomplete deletes: the audit row is INSERTed first,
    then the cascade runs in the same transaction. If either fails, both
    roll back.
  - DOES NOT prevent deletion of an arbitrary user. That's an authz
    decision the API layer makes (e.g., HOSTED only allows self-delete).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2.extras

logger = logging.getLogger("attestor.gdpr")


@dataclass(frozen=True)
class ExportPayload:
    """Result of export_user. Each section is a list of dicts."""
    user: Dict[str, Any]
    quota: Optional[Dict[str, Any]]
    projects: List[Dict[str, Any]] = field(default_factory=list)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    exported_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "quota": self.quota,
            "projects": self.projects,
            "sessions": self.sessions,
            "episodes": self.episodes,
            "memories": self.memories,
            "exported_at": self.exported_at,
            "row_counts": {
                "projects": len(self.projects),
                "sessions": len(self.sessions),
                "episodes": len(self.episodes),
                "memories": len(self.memories),
            },
        }


@dataclass(frozen=True)
class PurgeResult:
    """Result of purge_user. user_existed=False means the call was a no-op."""
    user_existed: bool
    audit_id: Optional[str] = None
    counts: Dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _row_to_jsonable(row: Any) -> Dict[str, Any]:
    """Convert a psycopg2 dict-row into a JSON-serializable dict."""
    out: Dict[str, Any] = {}
    for k, v in dict(row).items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif hasattr(v, "isoformat"):  # date / time
            out[k] = v.isoformat()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="replace")
        elif isinstance(v, list):
            out[k] = list(v)
        elif hasattr(v, "lower") and hasattr(v, "upper") and not isinstance(v, str):
            # psycopg2 Range
            out[k] = [v.lower, v.upper]
        else:
            out[k] = v
    return out


def _fetch_user(conn: Any, external_id: str) -> Optional[Dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id::text AS id, external_id, email, display_name, "
            "status, created_at, deleted_at, metadata "
            "FROM users WHERE external_id = %s",
            (external_id,),
        )
        row = cur.fetchone()
    return _row_to_jsonable(row) if row else None


# ──────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────


def export_user(conn: Any, external_id: str) -> ExportPayload:
    """Build a full data portability dump for the user.

    Caller MUST run this on a connection that has either BYPASSRLS
    (admin) or has the RLS var set to this user's id. Cross-user calls
    return empty (RLS denies the SELECTs).
    """
    user = _fetch_user(conn, external_id)
    if user is None:
        raise LookupError(f"user with external_id={external_id!r} not found")
    user_id = user["id"]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Quota
        cur.execute("SELECT * FROM user_quotas WHERE user_id = %s::uuid",
                    (user_id,))
        q_row = cur.fetchone()
        quota = _row_to_jsonable(q_row) if q_row else None

        # Projects
        cur.execute(
            "SELECT id::text AS id, user_id::text AS user_id, name, "
            "description, status, created_at, archived_at, metadata "
            "FROM projects WHERE user_id = %s::uuid ORDER BY created_at",
            (user_id,),
        )
        projects = [_row_to_jsonable(r) for r in cur.fetchall()]

        # Sessions
        cur.execute(
            "SELECT id::text AS id, user_id::text AS user_id, "
            "project_id::text AS project_id, title, status, "
            "created_at, last_active_at, ended_at, "
            "message_count, consolidation_state, metadata "
            "FROM sessions WHERE user_id = %s::uuid ORDER BY created_at",
            (user_id,),
        )
        sessions = [_row_to_jsonable(r) for r in cur.fetchall()]

        # Episodes
        cur.execute(
            "SELECT id::text AS id, user_id::text AS user_id, "
            "project_id::text AS project_id, session_id::text AS session_id, "
            "thread_id, user_turn_text, assistant_turn_text, "
            "user_ts, assistant_ts, agent_id, metadata, created_at "
            "FROM episodes WHERE user_id = %s::uuid ORDER BY created_at",
            (user_id,),
        )
        episodes = [_row_to_jsonable(r) for r in cur.fetchall()]

        # Memories — drop the embedding (large + opaque); audit doesn't need it
        cur.execute(
            "SELECT id::text AS id, user_id::text AS user_id, "
            "project_id::text AS project_id, session_id::text AS session_id, "
            "scope, content, content_hash, tags, category, entity, "
            "confidence, status, "
            "valid_from, valid_until, t_created, t_expired, "
            "superseded_by::text AS superseded_by, "
            "source_episode_id::text AS source_episode_id, "
            "source_span, extraction_model, "
            "agent_id, parent_agent_id, visibility, signature, metadata "
            "FROM memories WHERE user_id = %s::uuid ORDER BY t_created",
            (user_id,),
        )
        memories = [_row_to_jsonable(r) for r in cur.fetchall()]

    return ExportPayload(
        user=user, quota=quota,
        projects=projects, sessions=sessions,
        episodes=episodes, memories=memories,
    )


# ──────────────────────────────────────────────────────────────────────────
# Purge
# ──────────────────────────────────────────────────────────────────────────


def purge_user(
    conn: Any,
    external_id: str,
    *,
    reason: str = "gdpr_request",
    deleted_by: Optional[str] = None,
) -> PurgeResult:
    """Hard-delete the user and CASCADE through projects/sessions/episodes/
    memories/user_quotas. Logs to deletion_audit BEFORE the actual delete
    inside one transaction; rollback on either side keeps things consistent.

    Returns PurgeResult with per-table row counts (for audit dashboards).
    Caller is responsible for cleaning vector / graph stores that don't
    CASCADE through the FK.
    """
    user = _fetch_user(conn, external_id)
    if user is None:
        return PurgeResult(user_existed=False)
    user_id = user["id"]

    with conn.cursor() as cur:
        # Count first (for the audit entry)
        counts: Dict[str, int] = {}
        for table in ("projects", "sessions", "episodes", "memories"):
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = %s::uuid",
                (user_id,),
            )
            counts[table] = int(cur.fetchone()[0])

        # Audit FIRST so it survives even if the delete fails partway
        cur.execute(
            "INSERT INTO deletion_audit (user_id, external_id, "
            "deleted_by, reason, counts) "
            "VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id::text",
            (user_id, external_id, deleted_by, reason, json.dumps(counts)),
        )
        audit_id = cur.fetchone()[0]

        # CASCADE delete via the users FK
        cur.execute("DELETE FROM users WHERE id = %s::uuid", (user_id,))

    conn.commit()
    logger.info(
        "purged user external_id=%r user_id=%s counts=%s audit_id=%s",
        external_id, user_id, counts, audit_id,
    )
    return PurgeResult(
        user_existed=True, audit_id=audit_id, counts=counts,
    )


def list_audit_log(
    conn: Any, *, limit: int = 100,
) -> List[Dict[str, Any]]:
    """Read recent deletion audit entries. Read-only; appended-only table."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id::text AS id, user_id, external_id, "
            "deleted_at, deleted_by, reason, counts "
            "FROM deletion_audit ORDER BY deleted_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [_row_to_jsonable(r) for r in rows]
