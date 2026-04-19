"""PostgreSQL backend — document + vector (pgvector) + graph (Apache AGE) in one instance."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

import psycopg2
import psycopg2.extras

from attestor.models import Memory
from attestor.store.base import DocumentStore, VectorStore, GraphStore
from attestor.store.connection import CloudConnection

logger = logging.getLogger("attestor")


def _escape_cypher(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _parse_agtype(raw: str) -> Any:
    """Parse an AGE agtype string into a Python object.

    AGE returns values like:
        {"name": "Alice", "entity_type": "person"}::vertex
        {"id": 123, ...}::edge
        [...]::path

    Strip the ::suffix and parse the JSON.
    """
    if raw is None:
        return None
    s = str(raw)
    # Strip ::vertex, ::edge, ::path, ::numeric etc.
    s = re.sub(r"::(?:vertex|edge|path|numeric)\s*$", "", s)
    # Also handle nested ::vertex inside arrays/paths
    s = re.sub(r"::(?:vertex|edge|path|numeric)", "", s)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


class PostgresBackend(DocumentStore, VectorStore, GraphStore):
    """Multi-role PostgreSQL backend: document + vector (pgvector) + graph (AGE).

    Accepts raw config dict. See CloudConnection.from_config() for formats.

    Requires a PostgreSQL instance with pgvector and Apache AGE extensions.
    Neon (with pgvector enabled) and any self-hosted Postgres that has both
    extensions loaded are both supported. See tests/test_postgres_live.py
    for an example live-integration configuration.
    """

    ROLES: Set[str] = {"document", "vector", "graph"}

    def __init__(self, config: Dict[str, Any]) -> None:
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

        connect_kwargs: Dict[str, Any] = {
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
        # Determine embedding dimension before schema init
        self._ensure_embedding_fn()
        self._embedding_dim = self._embedder.dimension
        self._init_schema()
        self._init_age()

    def _execute(self, sql: str, params: Any = None) -> List[Dict[str, Any]]:
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

    # ── Schema Init ──

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

    # ── AGE Helpers ──

    def _require_age(self) -> None:
        """Raise if AGE is not available."""
        if not self._has_age:
            raise NotImplementedError(
                "Graph operations require the Apache AGE extension on this "
                "PostgreSQL instance. Configure a Neo4j backend for the graph "
                "role instead."
            )

    def _age_execute(self, sql: str, params: Any = None) -> None:
        """Execute a non-query AGE/SQL statement."""
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _age_query(self, cypher: str, cols: str = "result agtype") -> List[Any]:
        """Execute a Cypher query via AGE and return parsed results.

        Args:
            cypher: openCypher query string.
            cols: Column definition for the AS clause (default: "result agtype").
        """
        with self._conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            sql = f"SELECT * FROM cypher('memory_graph', $$ {cypher} $$) AS ({cols});"
            cur.execute(sql)
            rows = cur.fetchall()
        return [_parse_agtype(row[0]) if len(row) == 1 else tuple(_parse_agtype(c) for c in row) for row in rows]

    # ── DocumentStore ──

    def _memory_to_params(self, memory: Memory) -> Dict[str, Any]:
        return {
            "id": memory.id,
            "content": memory.content,
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

    def _row_to_memory(self, row: Dict[str, Any]) -> Memory:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Memory(
            id=row["id"],
            content=row["content"],
            tags=row.get("tags", []),
            category=row.get("category", "general"),
            entity=row.get("entity"),
            namespace=row.get("namespace", "default"),
            created_at=row["created_at"],
            event_date=row.get("event_date"),
            valid_from=row["valid_from"],
            valid_until=row.get("valid_until"),
            superseded_by=row.get("superseded_by"),
            confidence=row.get("confidence", 1.0),
            status=row.get("status", "active"),
            metadata=metadata,
        )

    def insert(self, memory: Memory) -> Memory:
        p = self._memory_to_params(memory)
        self._execute("""
            INSERT INTO memories (id, content, tags, category, entity, namespace,
                created_at, event_date, valid_from, valid_until,
                superseded_by, confidence, status, metadata)
            VALUES (%(id)s, %(content)s, %(tags)s, %(category)s, %(entity)s, %(namespace)s,
                %(created_at)s, %(event_date)s, %(valid_from)s, %(valid_until)s,
                %(superseded_by)s, %(confidence)s, %(status)s, %(metadata)s::jsonb)
        """, p)
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        rows = self._execute(
            "SELECT * FROM memories WHERE id = %s", (memory_id,)
        )
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    def update(self, memory: Memory) -> Memory:
        p = self._memory_to_params(memory)
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
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        namespace: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        filters = []
        params: Dict[str, Any] = {"lim": limit}

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
            filters.append("namespace = %(namespace)s")
            params["namespace"] = namespace
        if after:
            filters.append("created_at >= %(after)s")
            params["after"] = after
        if before:
            filters.append("created_at <= %(before)s")
            params["before"] = before

        where = " AND ".join(filters) if filters else "TRUE"
        rows = self._execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT %(lim)s",
            params,
        )
        return [self._row_to_memory(r) for r in rows]

    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        params: Dict[str, Any] = {"tags": tags, "lim": limit}
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
        self, query: str, params: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
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

    def stats(self) -> Dict[str, Any]:
        total = self._execute_scalar("SELECT COUNT(*) FROM memories")

        by_status: Dict[str, int] = {}
        for row in self._execute(
            "SELECT status, COUNT(*) as cnt FROM memories GROUP BY status"
        ):
            by_status[row["status"]] = row["cnt"]

        by_category: Dict[str, int] = {}
        for row in self._execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        ):
            by_category[row["category"]] = row["cnt"]

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }

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

    def _embed(self, text: str) -> List[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        return self._embedder.embed(text)

    def add(self, memory_id: str, content: str, namespace: str = "default") -> None:
        """Generate embedding and store on the memory row.

        ``namespace`` is accepted for parity with other backends. Scoping is
        enforced at the document row (memories.namespace) and propagated into
        vector search below; this method updates the embedding column on the
        existing row so no separate namespace write is required here.
        """
        del namespace  # reserved for future per-namespace index partitioning
        embedding = self._embed(content)
        self._execute(
            "UPDATE memories SET embedding = %s::vector WHERE id = %s",
            (str(embedding), memory_id),
        )

    def search(
        self,
        query_text: str,
        limit: int = 20,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search using pgvector cosine distance."""
        query_vec = self._embed(query_text)
        if namespace is None:
            rows = self._execute(
                "SELECT id, content, embedding <=> %s::vector AS distance "
                "FROM memories WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(query_vec), str(query_vec), limit),
            )
        else:
            rows = self._execute(
                "SELECT id, content, embedding <=> %s::vector AS distance "
                "FROM memories "
                "WHERE embedding IS NOT NULL AND namespace = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(query_vec), namespace, str(query_vec), limit),
            )
        return [
            {"memory_id": r["id"], "content": r["content"], "distance": r["distance"]}
            for r in rows
        ]

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        return self._execute_scalar(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        )

    # ── GraphStore (Apache AGE) ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._require_age()
        key = name.lower()
        escaped_key = _escape_cypher(key)
        escaped_name = _escape_cypher(name)

        # Check if entity exists
        results = self._age_query(
            f"MATCH (e:Entity {{key: '{escaped_key}'}}) RETURN e"
        )

        if results:
            existing = results[0]
            existing_type = existing.get("properties", {}).get("entity_type", "general")
            new_type = entity_type if (existing_type == "general" and entity_type != "general") else existing_type

            set_parts = [f"e.display_name = '{escaped_name}'", f"e.entity_type = '{_escape_cypher(new_type)}'"]
            if attributes:
                for k, v in attributes.items():
                    escaped_v = _escape_cypher(str(v))
                    set_parts.append(f"e.{k} = '{escaped_v}'")

            set_clause = ", ".join(set_parts)
            self._age_query(
                f"MATCH (e:Entity {{key: '{escaped_key}'}}) SET {set_clause} RETURN e"
            )
        else:
            props = f"key: '{escaped_key}', display_name: '{escaped_name}', entity_type: '{_escape_cypher(entity_type)}'"
            if attributes:
                for k, v in attributes.items():
                    escaped_v = _escape_cypher(str(v))
                    props += f", {k}: '{escaped_v}'"
            self._age_query(f"CREATE (e:Entity {{{props}}}) RETURN e")

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._require_age()
        from_key = _escape_cypher(from_entity.lower())
        to_key = _escape_cypher(to_entity.lower())
        from_name = _escape_cypher(from_entity)
        to_name = _escape_cypher(to_entity)

        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", relation_type).upper()

        # Ensure both entities exist (AGE doesn't support MERGE ... ON CREATE SET)
        for ek, en in [(from_key, from_name), (to_key, to_name)]:
            existing = self._age_query(
                f"MATCH (e:Entity {{key: '{ek}'}}) RETURN e"
            )
            if not existing:
                self._age_query(
                    f"CREATE (e:Entity {{key: '{ek}', display_name: '{en}', entity_type: 'general'}}) RETURN e"
                )

        # Create edge
        meta_props = f", relation_type: '{sanitized}'"
        if metadata:
            for k, v in metadata.items():
                meta_props += f", {k}: '{_escape_cypher(str(v))}'"

        self._age_query(
            f"MATCH (a:Entity {{key: '{from_key}'}}), (b:Entity {{key: '{to_key}'}}) "
            f"CREATE (a)-[r:RELATION {{{meta_props[2:]}}}]->(b) RETURN r"
        )

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        self._require_age()
        key = _escape_cypher(entity.lower())

        results = self._age_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return []

        results = self._age_query(
            f"MATCH (start:Entity {{key: '{key}'}})-[*1..{depth}]-(connected:Entity) "
            f"RETURN DISTINCT connected"
        )
        names = []
        for r in results:
            if isinstance(r, dict):
                name = r.get("properties", {}).get("display_name")
                if name:
                    names.append(name)
        return names

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        self._require_age()
        key = _escape_cypher(entity.lower())

        # Check entity exists
        results = self._age_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return {"entity": entity, "nodes": [], "edges": []}

        # Get nodes
        node_results = self._age_query(
            f"MATCH (start:Entity {{key: '{key}'}})-[*0..{depth}]-(n:Entity) "
            f"RETURN DISTINCT n"
        )

        nodes = []
        node_keys = set()
        for r in node_results:
            if isinstance(r, dict):
                props = r.get("properties", {})
                nk = props.get("key")
                if nk and nk not in node_keys:
                    node_keys.add(nk)
                    nodes.append({
                        "name": props.get("display_name", nk),
                        "type": props.get("entity_type", "general"),
                        "key": nk,
                    })

        # Get edges between those nodes
        edges = []
        if len(node_keys) > 1:
            edge_results = self._age_query(
                f"MATCH (start:Entity {{key: '{key}'}})-[*0..{depth}]-(a:Entity)-[r:RELATION]->(b:Entity) "
                f"RETURN DISTINCT a, r, b",
                cols="a agtype, r agtype, b agtype",
            )
            for row in edge_results:
                if isinstance(row, tuple) and len(row) == 3:
                    a_props = row[0].get("properties", {}) if isinstance(row[0], dict) else {}
                    r_props = row[1].get("properties", {}) if isinstance(row[1], dict) else {}
                    b_props = row[2].get("properties", {}) if isinstance(row[2], dict) else {}
                    a_key = a_props.get("key")
                    b_key = b_props.get("key")
                    if a_key in node_keys and b_key in node_keys:
                        edges.append({
                            "source": a_key,
                            "target": b_key,
                            "type": r_props.get("relation_type", "RELATION"),
                        })

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        self._require_age()
        if entity_type:
            et = _escape_cypher(entity_type)
            results = self._age_query(
                f"MATCH (e:Entity) WHERE e.entity_type = '{et}' RETURN e"
            )
        else:
            results = self._age_query("MATCH (e:Entity) RETURN e")

        entities = []
        for r in results:
            if isinstance(r, dict):
                props = r.get("properties", {})
                key = props.get("key", "")
                attrs = {
                    k: v for k, v in props.items()
                    if k not in ("key", "display_name", "entity_type")
                }
                entities.append({
                    "name": props.get("display_name", key),
                    "type": props.get("entity_type", "general"),
                    "key": key,
                    "attributes": attrs,
                })
        return entities

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        self._require_age()
        key = _escape_cypher(entity.lower())

        results = self._age_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return []

        edge_results = self._age_query(
            f"MATCH (a:Entity)-[r:RELATION]-(b:Entity) "
            f"WHERE a.key = '{key}' OR b.key = '{key}' "
            f"RETURN a, r, b",
            cols="a agtype, r agtype, b agtype",
        )

        seen = set()
        edges = []
        for row in edge_results:
            if isinstance(row, tuple) and len(row) == 3:
                a_props = row[0].get("properties", {}) if isinstance(row[0], dict) else {}
                r_props = row[1].get("properties", {}) if isinstance(row[1], dict) else {}
                b_props = row[2].get("properties", {}) if isinstance(row[2], dict) else {}

                subject = a_props.get("display_name", "")
                predicate = r_props.get("relation_type", "RELATION")
                obj = b_props.get("display_name", "")
                event_date = r_props.get("event_date", "")

                triple = (subject, predicate, obj)
                if triple not in seen:
                    seen.add(triple)
                    edges.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "event_date": event_date,
                    })
        return edges

    def graph_stats(self) -> Dict[str, Any]:
        """Graph statistics: node/edge counts, entity types."""
        self._require_age()
        node_results = self._age_query(
            "MATCH (e:Entity) RETURN count(e)",
        )
        nodes = node_results[0] if node_results else 0

        edge_results = self._age_query(
            "MATCH ()-[r:RELATION]->() RETURN count(r)",
        )
        edge_count = edge_results[0] if edge_results else 0

        type_results = self._age_query(
            "MATCH (e:Entity) RETURN e.entity_type, count(e)",
            cols="et agtype, cnt agtype",
        )
        types: Dict[str, int] = {}
        for row in type_results:
            if isinstance(row, tuple) and len(row) == 2:
                et = str(row[0]).strip('"') if row[0] else "general"
                types[et] = int(row[1]) if row[1] else 0

        return {"nodes": nodes, "edges": edge_count, "types": types}

    def save(self) -> None:
        pass  # PostgreSQL persists automatically

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
