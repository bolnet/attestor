"""PostgreSQL backend — document + vector (pgvector) + graph (Apache AGE) in one instance.

Public class composes three role mixins:

    _PostgresDocumentMixin   — memories table CRUD (source of truth)
    _PostgresVectorMixin     — pgvector embedding column + cosine search
    _PostgresGraphMixin      — Apache AGE Cypher (PageRank-free; AGE/PG bridge)

The mixins are stateless — they reach into ``self._conn``, ``self._v4``,
``self._embedder``, ``self._has_age``, and the SQL helpers wired up by
``__init__`` below. This split mirrors the same composition pattern used by
``azure_backend.py`` and ``aws_backend.py``.

Backward-compat: ``gcp_backend.GCPBackend`` inherits from ``PostgresBackend``
and calls ``super().__init__(config)``; the public class still owns the
constructor and the ``ROLES`` ClassVar so that path is unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

import psycopg2
import psycopg2.extras

from attestor.store._postgres_document import _PostgresDocumentMixin
from attestor.store._postgres_graph import (
    _PostgresGraphMixin,
    _CYPHER_IDENT_RE,  # noqa: F401  (preserved for backwards-compat re-export)
    _escape_cypher,  # noqa: F401  (preserved for backwards-compat re-export)
    _parse_agtype,  # noqa: F401  (preserved for backwards-compat re-export)
    _validate_cypher_identifier,  # noqa: F401  (preserved for backwards-compat re-export)
    _validate_depth,  # noqa: F401  (preserved for backwards-compat re-export)
)
from attestor.store._postgres_vector import _PostgresVectorMixin
from attestor.store.connection import CloudConnection

logger = logging.getLogger("attestor")


class PostgresBackend(
    _PostgresDocumentMixin,
    _PostgresVectorMixin,
    _PostgresGraphMixin,
):
    """Multi-role PostgreSQL backend: document + vector (pgvector) + graph (AGE).

    Accepts raw config dict. See CloudConnection.from_config() for formats.

    Requires a PostgreSQL instance with pgvector and Apache AGE extensions.
    Neon (with pgvector enabled) and any self-hosted Postgres that has both
    extensions loaded are both supported. See tests/test_postgres_live.py
    for an example live-integration configuration.
    """

    ROLES: ClassVar[set[str]] = {"document", "vector", "graph"}

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        conn_info = CloudConnection.from_config(config, backend_name="postgres")
        self._conn_info = conn_info

        # Extract host/port from URL (e.g. "postgresql://localhost:5433" -> host=localhost, port=5433)
        from urllib.parse import urlparse
        parsed = urlparse(conn_info.url)
        host = parsed.hostname or "localhost"
        port = parsed.port or conn_info.port

        # Support sslmode from config or connection options
        sslmode = config.get("sslmode") or conn_info.extra.get("sslmode")

        connect_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "dbname": conn_info.database,
            "user": conn_info.auth.username,
            "password": conn_info.auth.password,
        }
        if sslmode:
            connect_kwargs["sslmode"] = sslmode

        self._conn = psycopg2.connect(**connect_kwargs)
        self._conn.autocommit = True

        self._embedder = None  # lazy-init via shared embeddings module
        self._embedding_fn = None  # backward compat for benchmark code
        self._has_age = False  # set by _init_age()
        # Determine embedding dimension before schema init. Caller may pass
        # `embedding_dim` to skip the embedder probe entirely (useful when
        # the embedder lives in a separate service, in tests that only need
        # the document path, or when the provider is temporarily unavailable
        # but the schema is already in place).
        if config.get("embedding_dim") is not None:
            self._embedding_dim = int(config["embedding_dim"])
        else:
            self._ensure_embedding_fn()
            self._embedding_dim = self._embedder.dimension

        # v4 mode is opt-in via ATTESTOR_V4=1 env var or config["v4"]=True.
        # When ON: load attestor/store/schema.sql (greenfield v4 schema).
        # When OFF: keep the v3 inline _init_schema() so existing callers
        # continue to work unchanged.
        self._v4 = bool(
            config.get("v4", False)
            or os.environ.get("ATTESTOR_V4") in {"1", "true", "True"}
        )
        # Production deployments often run with a separate migration role
        # that owns the tables and a runtime role that does not. The runtime
        # role can't ALTER TABLE / CREATE POLICY. Same applies in tests that
        # boot AgentMemory as a non-superuser to verify RLS.
        skip_schema = bool(
            config.get("skip_schema_init", False)
            or os.environ.get("ATTESTOR_SKIP_SCHEMA_INIT") in {"1", "true", "True"}
        )
        if not skip_schema:
            if self._v4:
                self._init_schema_v4()
            else:
                self._init_schema()
        self._init_age()

    # ── Low-level SQL helpers (used by every mixin) ──

    def _execute(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        """Execute SQL and return rows as dicts."""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            return []

    def _execute_scalar(self, sql: str, params: Any = None) -> Any:
        """Execute SQL and return a single scalar value."""
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None

    # ── v4 RLS / schema helpers (Phase 1; not yet wired by default) ──

    def _set_rls_user(self, user_id: str | None) -> None:
        """Set the connection-local RLS variable so policies on v4 tables
        filter by this user. Pass None / empty to clear (fail-closed).

        Must be called on every connection checkout once the v4 schema is
        in use. No-op for v3 schema since v3 tables have no RLS policies."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (str(user_id) if user_id else "",),
            )

    def _load_v4_schema_sql(self) -> str:
        """Read attestor/store/schema.sql and substitute {embedding_dim}.

        The caller is responsible for executing the result. Used by
        higher-level v4 init paths and by tests."""
        from pathlib import Path
        path = Path(__file__).resolve().parent / "schema.sql"
        return path.read_text().replace(
            "{embedding_dim}", str(self._embedding_dim)
        )

    def _init_schema_v4(self) -> None:
        """Apply schema.sql (greenfield v4). Idempotent — uses CREATE TABLE
        IF NOT EXISTS throughout, so safe to call against an already-v4 DB.
        RLS policies are dropped + re-created to stay current."""
        sql = self._load_v4_schema_sql()
        # Use a single transaction for the whole schema apply so a partial
        # failure doesn't leave the DB half-migrated.
        was_autocommit = self._conn.autocommit
        try:
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(sql)
            self._conn.commit()
            logger.info("v4 schema initialized from schema.sql")
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._conn.autocommit = was_autocommit

    # ── Schema Init (v3 — kept until Phase 2 switches the default) ──

    def _init_schema(self) -> None:
        """Create memories table and indexes."""
        self._execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
        """)
        dim = self._embedding_dim
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT[] NOT NULL DEFAULT '{{}}'::text[],
                    category TEXT NOT NULL DEFAULT 'general',
                    entity TEXT,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL,
                    event_date TEXT,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    superseded_by TEXT,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active',
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({dim})
                );
            """)
        # Indexes
        for col in ["status", "category", "entity", "created_at"]:
            self._execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_{col}
                ON memories ({col});
            """)
        # HNSW index for vector cosine search
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
            ON memories USING hnsw (embedding vector_cosine_ops);
        """)

    def _init_age(self) -> None:
        """Initialize Apache AGE extension and graph.

        Non-fatal: if AGE is not available (e.g. Neon, Cloud SQL),
        graph methods will raise NotImplementedError but document+vector still work.
        """
        try:
            self._execute("CREATE EXTENSION IF NOT EXISTS age;")
            self._age_execute("LOAD 'age';")
            self._age_execute(
                "SET search_path = ag_catalog, \"$user\", public;"
            )
            # create_graph is not idempotent — catch if exists
            try:
                self._age_execute(
                    "SELECT create_graph('memory_graph');"
                )
            except psycopg2.errors.InvalidSchemaName:
                self._conn.rollback()
                self._conn.autocommit = True
            except Exception as e:
                if "already exists" in str(e):
                    self._conn.rollback()
                    self._conn.autocommit = True
                else:
                    raise
            self._has_age = True
            logger.info("Apache AGE graph initialized")
        except Exception as e:
            self._conn.rollback()
            self._conn.autocommit = True
            self._has_age = False
            logger.info("Apache AGE not available — graph role disabled: %s", e)

    # ── Lifecycle ──

    def save(self) -> None:
        pass  # PostgreSQL persists automatically

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
