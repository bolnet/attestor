"""Postgres document-role mixin (split from postgres_backend.py).

This module is private — consumers should import ``PostgresBackend`` from
``attestor.store.postgres_backend``. The mixin is stateless: it operates on
``self._conn``, ``self._v4``, and the ``_execute`` / ``_execute_scalar``
helpers configured by ``PostgresBackend.__init__``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from attestor.models import Memory


class _PostgresDocumentMixin:
    """Document-store role for PostgresBackend (memories table CRUD)."""

    # ── DocumentStore ──

    def _memory_to_params(self, memory: Memory) -> dict[str, Any]:
        # Common params (used by both v3 and v4 paths). v4 also references
        # the additional keys below.
        params = {
            "id": memory.id,
            "content": memory.content,
            "content_hash": memory.content_hash,
            "tags": memory.tags,
            "category": memory.category,
            "entity": memory.entity,
            "namespace": memory.namespace,
            "created_at": memory.created_at,
            "event_date": memory.event_date,
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
            "superseded_by": memory.superseded_by,
            "confidence": memory.confidence,
            "status": memory.status,
            "metadata": json.dumps(memory.metadata),
        }
        # source_span on the v4 schema is INT4RANGE [start, end). psycopg2
        # serializes a Python tuple/list of two ints into a Range literal
        # via the explicit cast in the INSERT.
        span = memory.source_span
        if isinstance(span, (list, tuple)) and len(span) == 2:
            source_span_literal = f"[{int(span[0])},{int(span[1])})"
        else:
            source_span_literal = None

        # v4-only params — present when caller set them.
        params.update({
            "user_id": memory.user_id,
            "project_id": memory.project_id,
            "session_id": memory.session_id,
            "scope": memory.scope or "user",
            "source_episode_id": memory.source_episode_id,
            "source_span": source_span_literal,
            "extraction_model": memory.extraction_model,
            "agent_id": memory.agent_id,
            "parent_agent_id": memory.parent_agent_id,
            "visibility": memory.visibility or "team",
            "signature": memory.signature,
        })
        return params

    def _row_to_memory(self, row: dict[str, Any]) -> Memory:
        # Memory.from_row() handles both v3 and v4 row shapes.
        return Memory.from_row(row)

    def insert(self, memory: Memory) -> Memory:
        """Insert a memory. Routes to the v4 INSERT when running on the v4
        schema (``self._v4``); otherwise uses the v3 INSERT for backward
        compat. The Memory dataclass carries both shapes; only the relevant
        fields are written per branch.

        v4 requires user_id; raises ValueError if absent. The DB generates
        the UUID id; this method updates ``memory.id`` in place to match."""
        p = self._memory_to_params(memory)
        if self._v4:
            if not memory.user_id:
                raise ValueError(
                    "v4 schema requires Memory.user_id; pass user_id when "
                    "calling AgentMemory.add() or use a v3-mode backend."
                )
            rows = self._execute(
                """
                INSERT INTO memories (
                    user_id, project_id, session_id, scope,
                    content, content_hash, tags, category, entity,
                    confidence, status,
                    valid_from, valid_until,
                    superseded_by,
                    source_episode_id, source_span, extraction_model,
                    agent_id, parent_agent_id, visibility, signature,
                    metadata
                )
                VALUES (
                    %(user_id)s, %(project_id)s, %(session_id)s, %(scope)s,
                    %(content)s, %(content_hash)s, %(tags)s, %(category)s, %(entity)s,
                    %(confidence)s, %(status)s,
                    COALESCE(%(valid_from)s::timestamptz, NOW()),
                    %(valid_until)s::timestamptz,
                    %(superseded_by)s,
                    %(source_episode_id)s, %(source_span)s::int4range,
                    %(extraction_model)s,
                    %(agent_id)s, %(parent_agent_id)s, %(visibility)s, %(signature)s,
                    %(metadata)s::jsonb
                )
                RETURNING id, t_created
                """,
                p,
            )
            if rows:
                tc = rows[0].get("t_created")
                t_created_iso = (
                    tc.isoformat()
                    if hasattr(tc, "isoformat")
                    else str(tc) if tc else None
                )
                memory = replace(
                    memory,
                    id=str(rows[0]["id"]),
                    t_created=t_created_iso,
                )
            return memory
        # v3 path — id provided by Memory dataclass default (12-char hex)
        self._execute(
            """
            INSERT INTO memories (id, content, tags, category, entity, namespace,
                created_at, event_date, valid_from, valid_until,
                superseded_by, confidence, status, metadata)
            VALUES (%(id)s, %(content)s, %(tags)s, %(category)s, %(entity)s, %(namespace)s,
                %(created_at)s, %(event_date)s, %(valid_from)s, %(valid_until)s,
                %(superseded_by)s, %(confidence)s, %(status)s, %(metadata)s::jsonb)
            """,
            p,
        )
        return memory

    def get(self, memory_id: str) -> Memory | None:
        rows = self._execute(
            "SELECT * FROM memories WHERE id = %s", (memory_id,)
        )
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    def update(self, memory: Memory) -> Memory:
        p = self._memory_to_params(memory)
        if self._v4:
            # v4 has no namespace / created_at / event_date columns.
            # Mutates only the fields the call surface actually changes.
            self._execute(
                """
                UPDATE memories SET
                    content       = %(content)s,
                    tags          = %(tags)s,
                    category      = %(category)s,
                    entity        = %(entity)s,
                    valid_from    = COALESCE(%(valid_from)s::timestamptz, valid_from),
                    valid_until   = %(valid_until)s::timestamptz,
                    superseded_by = %(superseded_by)s,
                    confidence    = %(confidence)s,
                    status        = %(status)s,
                    metadata      = %(metadata)s::jsonb
                WHERE id = %(id)s
                """,
                p,
            )
            return memory
        # v3 path
        self._execute("""
            UPDATE memories SET
                content = %(content)s, tags = %(tags)s, category = %(category)s,
                entity = %(entity)s, namespace = %(namespace)s,
                created_at = %(created_at)s,
                event_date = %(event_date)s, valid_from = %(valid_from)s,
                valid_until = %(valid_until)s, superseded_by = %(superseded_by)s,
                confidence = %(confidence)s, status = %(status)s,
                metadata = %(metadata)s::jsonb
            WHERE id = %(id)s
        """, p)
        return memory

    def delete(self, memory_id: str) -> bool:
        rows = self._execute(
            "DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,)
        )
        return len(rows) > 0

    def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        # v4 schema replaces the v3 ``created_at`` text column with the
        # bi-temporal ``t_created`` (TIMESTAMPTZ). Use the right one per
        # active schema. v4 has no ``namespace`` column — namespace is
        # stored as ``metadata->>'_namespace'`` (PR #77 write-side fix);
        # the RLS policy at memories filters by ``user_id`` only, NOT
        # by namespace, so namespace must still be filtered here when
        # the caller passes one (e.g. LME bench writes one user with
        # many per-sample namespaces).
        time_col = "t_created" if self._v4 else "created_at"

        filters = []
        params: dict[str, Any] = {"lim": limit}

        if status:
            filters.append("status = %(status)s")
            params["status"] = status
        if category:
            filters.append("category = %(category)s")
            params["category"] = category
        if entity:
            filters.append("entity = %(entity)s")
            params["entity"] = entity
        if namespace:
            if self._v4:
                filters.append("metadata->>'_namespace' = %(namespace)s")
            else:
                filters.append("namespace = %(namespace)s")
            params["namespace"] = namespace

        # Phase 5 — recall_started_at ceiling: when an active recall
        # scope is open, no writes that landed after the recall began
        # are visible. v4 only (uses TIMESTAMPTZ t_created).
        if self._v4:
            from attestor.recall_context import current_recall_started_at
            _ceiling = current_recall_started_at()
            if _ceiling is not None:
                filters.append("t_created <= %(_recall_ceiling)s")
                params["_recall_ceiling"] = _ceiling

        if after:
            filters.append(f"{time_col} >= %(after)s")
            params["after"] = after
        if before:
            filters.append(f"{time_col} <= %(before)s")
            params["before"] = before

        where = " AND ".join(filters) if filters else "TRUE"
        rows = self._execute(
            f"SELECT * FROM memories WHERE {where} "
            f"ORDER BY {time_col} DESC LIMIT %(lim)s",
            params,
        )
        return [self._row_to_memory(r) for r in rows]

    def tag_search(
        self,
        tags: list[str],
        category: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        params: dict[str, Any] = {"tags": tags, "lim": limit}
        filters = [
            "status = 'active'",
            "valid_until IS NULL",
            "tags && %(tags)s",  # overlap operator (any tag matches)
        ]
        if category:
            filters.append("category = %(category)s")
            params["category"] = category
        if namespace:
            filters.append("namespace = %(namespace)s")
            params["namespace"] = namespace

        where = " AND ".join(filters)
        rows = self._execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT %(lim)s",
            params,
        )
        return [self._row_to_memory(r) for r in rows]

    def execute(
        self, query: str, params: Any | None = None
    ) -> list[dict[str, Any]]:
        """Execute raw SQL query."""
        return self._execute(query, params)

    def archive_before(self, date: str) -> int:
        rows = self._execute(
            "UPDATE memories SET status = 'archived' "
            "WHERE created_at < %s AND status = 'active' "
            "RETURNING id",
            (date,),
        )
        return len(rows)

    def compact(self) -> int:
        rows = self._execute(
            "DELETE FROM memories WHERE status = 'archived' RETURNING id"
        )
        return len(rows)

    def stats(self) -> dict[str, Any]:
        total = self._execute_scalar("SELECT COUNT(*) FROM memories")

        by_status: dict[str, int] = {}
        for row in self._execute(
            "SELECT status, COUNT(*) as cnt FROM memories GROUP BY status"
        ):
            by_status[row["status"]] = row["cnt"]

        by_category: dict[str, int] = {}
        for row in self._execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        ):
            by_category[row["category"]] = row["cnt"]

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }
