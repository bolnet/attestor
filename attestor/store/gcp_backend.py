"""GCP AlloyDB backend — extends PostgresBackend with AlloyDB Connector + ScaNN + Vertex AI.

AlloyDB is wire-compatible with PostgreSQL, so most methods are inherited.
Only connection setup, index strategy, and embedding preference differ.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from attestor.store.postgres_backend import PostgresBackend

logger = logging.getLogger("attestor")


def _has_alloydb_connector() -> bool:
    """Check if the AlloyDB Connector package is available."""
    try:
        from google.cloud.alloydb.connector import Connector  # noqa: F401
        return True
    except ImportError:
        return False


class GCPBackend(PostgresBackend):
    """AlloyDB backend — PostgresBackend with GCP-native connection, ScaNN index, and Vertex AI embeddings.

    Connection strategies (in order):
        1. AlloyDB Connector (IAM auth via ADC) — if project_id + cluster + instance are set
        2. Direct psycopg2 via Auth Proxy URL — standard PostgreSQL URL fallback

    Config keys (beyond standard postgres keys):
        project_id: GCP project ID
        region: GCP region (default: us-central1)
        cluster: AlloyDB cluster name
        instance: AlloyDB instance name
        database: Database name (default: attestor)
    """

    ROLES: ClassVar[set[str]] = {"document", "vector", "graph"}

    def __init__(self, config: dict[str, Any]) -> None:
        gcp_fields = self._extract_gcp_fields(config)

        if self._should_use_connector(gcp_fields):
            self._init_via_connector(config, gcp_fields)
        else:
            # Fall back to standard psycopg2 connection (same as PostgresBackend)
            super().__init__(config)

    def _extract_gcp_fields(self, config: dict[str, Any]) -> dict[str, str]:
        """Extract GCP-specific fields from config."""
        return {
            "project_id": config.get("project_id", ""),
            "region": config.get("region", "us-central1"),
            "cluster": config.get("cluster", ""),
            "instance": config.get("instance", ""),
            "database": config.get("database", "attestor"),
        }

    def _should_use_connector(self, gcp_fields: dict[str, str]) -> bool:
        """Use AlloyDB Connector if project_id, cluster, and instance are all set."""
        return bool(
            gcp_fields["project_id"]
            and gcp_fields["cluster"]
            and gcp_fields["instance"]
            and _has_alloydb_connector()
        )

    def _init_via_connector(
        self, config: dict[str, Any], gcp_fields: dict[str, str]
    ) -> None:
        """Connect via AlloyDB Connector with IAM auth (ADC)."""
        from google.cloud.alloydb.connector import Connector

        self._config = config
        self._connector = Connector()

        instance_uri = (
            f"projects/{gcp_fields['project_id']}"
            f"/locations/{gcp_fields['region']}"
            f"/clusters/{gcp_fields['cluster']}"
            f"/instances/{gcp_fields['instance']}"
        )

        self._conn = self._connector.connect(
            instance_uri,
            "pg8000",
            db=gcp_fields["database"],
            enable_iam_auth=True,
        )
        self._conn.autocommit = True

        self._embedder = None
        self._embedding_fn = None
        self._ensure_embedding_fn()
        self._embedding_dim = self._embedder.dimension
        self._init_schema()
        self._init_age()

    def _init_schema(self) -> None:
        """Create schema with ScaNN index (AlloyDB-optimized), falling back to HNSW."""
        super()._init_schema()
        self._try_scann_index()

    def _try_scann_index(self) -> None:
        """Replace HNSW index with ScaNN if the extension is available."""
        try:
            self._execute("CREATE EXTENSION IF NOT EXISTS alloydb_scann;")
            self._execute("DROP INDEX IF EXISTS idx_memories_embedding_hnsw;")
            self._execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_scann
                ON memories USING scann (embedding vector_cosine_ops)
                WITH (num_leaves = 5);
            """)
            logger.info("Using ScaNN index for vector search (AlloyDB-optimized)")
        except Exception as e:
            logger.warning(
                "ScaNN extension not available, keeping HNSW index: %s", e
            )
            # Rollback any failed transaction state
            try:
                self._conn.rollback()
                self._conn.autocommit = True
            except Exception:
                pass

    def _ensure_embedding_fn(self) -> None:
        """Prefer Vertex AI embeddings on GCP."""
        if self._embedder is not None:
            return

        from attestor.store.embeddings import get_embedding_provider

        self._embedder = get_embedding_provider("vertex_ai")
        if self._embedder.provider_name == "openai":
            self._openai_client = getattr(self._embedder, "_client", True)
        self._embedding_fn = self._embedder

    def close(self) -> None:
        """Close connection and AlloyDB Connector if used."""
        super().close()
        connector = getattr(self, "_connector", None)
        if connector is not None:
            try:
                connector.close()
            except Exception:
                pass
