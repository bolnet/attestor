"""pgvector-backed vector store for semantic memory search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA_PATH = Path(__file__).parent / "schema_pg.sql"


class VectorStore:
    """PostgreSQL + pgvector store for memory embeddings."""

    def __init__(self, connection_string: str):
        import psycopg
        from pgvector.psycopg import register_vector

        self.connection_string = connection_string
        self._conn = psycopg.connect(connection_string)
        self._conn.autocommit = True
        self._init_schema()
        register_vector(self._conn)

    def _init_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text()
        with self._conn.cursor() as cur:
            cur.execute(schema_sql)

    def close(self) -> None:
        self._conn.close()

    def add(self, memory_id: str, content: str, embedding: List[float]) -> None:
        """Store a memory embedding."""
        import uuid
        import numpy as np

        vec_id = uuid.uuid4().hex[:12]
        vec = np.array(embedding, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_vectors (id, memory_id, content, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (vec_id, memory_id, content, vec),
            )

    def delete(self, memory_id: str) -> bool:
        """Remove embedding for a memory."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_vectors WHERE memory_id = %s",
                (memory_id,),
            )
            return cur.rowcount > 0

    def search(
        self, query_embedding: List[float], limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find memories most similar to the query embedding.

        Returns dicts with: memory_id, content, distance
        """
        import numpy as np

        vec = np.array(query_embedding, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT memory_id, content, embedding <-> %s AS distance "
                "FROM memory_vectors "
                "WHERE embedding IS NOT NULL "
                "ORDER BY embedding <-> %s "
                "LIMIT %s",
                (vec, vec, limit),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        """Get total number of stored vectors."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memory_vectors")
            return cur.fetchone()[0]

    def create_index(self, lists: int = 100) -> None:
        """Create IVFFlat index for fast approximate search.

        Call after bulk loading data. lists should be ~sqrt(n_vectors).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mv_embedding ON memory_vectors "
                "USING ivfflat (embedding vector_l2_ops) WITH (lists = %s)",
                (lists,),
            )
