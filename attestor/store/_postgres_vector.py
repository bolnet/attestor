"""Postgres vector-role mixin (split from postgres_backend.py).

This module is private — consumers should import ``PostgresBackend`` from
``attestor.store.postgres_backend``. The mixin is stateless: it operates on
``self._conn``, ``self._embedder``, ``self._v4``, and the ``_execute`` /
``_execute_scalar`` helpers configured by ``PostgresBackend.__init__``.

Implements the optional pgvector path. When the embedder is unavailable or
pgvector is not installed, vector methods raise; the document path remains
operational.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import psycopg2.extras

logger = logging.getLogger("attestor")


class _PostgresVectorMixin:
    """pgvector role for PostgresBackend (embedding column + cosine search)."""

    # ── VectorStore ──

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init embedding provider via shared module."""
        if self._embedder is not None:
            return

        from attestor.store.embeddings import get_embedding_provider

        self._embedder = get_embedding_provider()
        # Backward compat: benchmark code checks _openai_client to confirm provider
        if self._embedder.provider_name == "openai":
            self._openai_client = getattr(self._embedder, "_client", True)
        self._embedding_fn = self._embedder  # backward compat marker (non-None = initialized)

    def _embed(self, text: str) -> list[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        from attestor import trace as _tr
        if not _tr.is_enabled():
            return self._embedder.embed(text)
        import time as _time
        t0 = _time.monotonic()
        vec = self._embedder.embed(text)
        _tr.event(
            "ingest.embed",
            provider=getattr(self._embedder, "provider", type(self._embedder).__name__),
            model=getattr(self._embedder, "model", "?"),
            dim=len(vec),
            text_len=len(text),
            latency_ms=round((_time.monotonic() - t0) * 1000, 2),
        )
        return vec

    def _build_embedding_text(self, memory_id: str, content: str) -> str:
        """Build the text used for embedding (Phase 4.1, roadmap §B.1).

        On v4 rows that have a source_episode_id, concatenate:

            extracted fact (content)
            ---
            user turn (verbatim)
            ---
            assistant turn (verbatim)
            ---
            Tags: tag1, tag2, ...

        This is LongMemEval Finding 2 — embedding the round, not just the
        fact, gives +4% recall@k and +5% downstream QA. On v3 rows or when
        the source episode is missing, falls back to just the fact text.
        """
        if not self._v4:
            return content
        try:
            with self._conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor,
            ) as cur:
                cur.execute(
                    """
                    SELECT m.tags, m.source_episode_id,
                           e.user_turn_text, e.assistant_turn_text
                    FROM memories m
                    LEFT JOIN episodes e ON e.id = m.source_episode_id
                    WHERE m.id = %s
                    """,
                    (memory_id,),
                )
                row = cur.fetchone()
        except Exception as e:
            logger.debug("_build_embedding_text lookup failed: %s", e)
            return content
        if not row:
            return content

        parts: list[str] = [content]
        user_text = row.get("user_turn_text")
        asst_text = row.get("assistant_turn_text")
        if user_text:
            parts.append(user_text)
        if asst_text:
            parts.append(asst_text)
        tags = row.get("tags") or []
        if tags:
            parts.append("Tags: " + ", ".join(tags))
        return "\n---\n".join(parts)

    def add(self, memory_id: str, content: str, namespace: str = "default") -> None:
        """Generate embedding and store on the memory row.

        ``namespace`` is accepted for parity with other backends. Scoping is
        enforced at the document row (memories.namespace) and propagated into
        vector search below; this method updates the embedding column on the
        existing row so no separate namespace write is required here.

        For v4 rows linked to an episode, the embedded text is the full
        round (fact + user turn + assistant turn + tags), not just the
        fact alone — see ``_build_embedding_text``.
        """
        del namespace  # reserved for future per-namespace index partitioning
        text = self._build_embedding_text(memory_id, content)
        embedding = self._embed(text)
        self._execute(
            "UPDATE memories SET embedding = %s::vector WHERE id = %s",
            (str(embedding), memory_id),
        )

    def search(
        self,
        query_text: str,
        limit: int = 20,
        namespace: str | None = None,
        as_of: datetime | None = None,
        time_window: Any | None = None,    # TimeWindow (avoid import cycle)
    ) -> list[dict[str, Any]]:
        """Vector similarity search using pgvector cosine distance.

        v4 + Phase 5.2 — bi-temporal filters (roadmap §C.2/§C.3):

          time_window  → tstzrange(valid_from, valid_until) && tstzrange(start, end)
                         pre-filters by EVENT-time overlap (when the fact
                         was true in the world)

          as_of        → t_created <= as_of AND t_expired > as_of
                         AND tstzrange(valid_from, valid_until) @> as_of
                         filters by TRANSACTION-time AND event-time:
                         "what did the system believe was true on date X"

        Both filters are additive; passing both narrows further. v3
        callers omit them — search behaves exactly as before.
        """
        query_vec = self._embed(query_text)
        params: list[Any] = [str(query_vec)]
        where: list[str] = ["embedding IS NOT NULL"]

        if namespace is not None:
            if self._v4:
                where.append("metadata->>'_namespace' = %s")
            else:
                where.append("namespace = %s")
            params.append(namespace)

        # Phase 5 — recall_started_at ceiling. Independent of explicit
        # ``as_of``: ceiling captures "when this recall started" (audit
        # invariant A2: no post-recall writes leak in), as_of captures
        # "what did the system believe on date X" (audit invariant A1).
        # Both filter on t_created; we apply them together when both
        # are active.
        if self._v4:
            from attestor.recall_context import current_recall_started_at
            _ceiling = current_recall_started_at()
            if _ceiling is not None:
                where.append("t_created <= %s")
                params.append(_ceiling)

            # Bi-temporal filters only meaningful on v4 schema
            if as_of is not None:
                where.append(
                    "t_created <= %s "
                    "AND COALESCE(t_expired, 'infinity'::timestamptz) > %s "
                    "AND tstzrange(valid_from, "
                    "  COALESCE(valid_until, 'infinity'::timestamptz)) @> %s::timestamptz"
                )
                params.extend([as_of, as_of, as_of])
            if time_window is not None:
                tw_start = getattr(time_window, "start", None)
                tw_end = getattr(time_window, "end", None)
                # tstzrange(NULL, NULL) is "any time" — Postgres treats
                # NULL bounds as -infinity / infinity inside the range
                # constructor, which is what we want for open-ended windows.
                where.append(
                    "tstzrange(valid_from, "
                    "  COALESCE(valid_until, 'infinity'::timestamptz)) "
                    "&& tstzrange(%s::timestamptz, %s::timestamptz)"
                )
                params.extend([tw_start, tw_end])

        sql = (
            "SELECT id, content, embedding <=> %s::vector AS distance "
            "FROM memories WHERE " + " AND ".join(where) +
            " ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        params.append(str(query_vec))
        params.append(limit)
        rows = self._execute(sql, params)
        return [
            {"memory_id": r["id"], "content": r["content"], "distance": r["distance"]}
            for r in rows
        ]

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        return self._execute_scalar(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        )
