"""ArangoDB backend — talks to ArangoDB Community Edition via python-arango.

Provides document, vector, and graph roles in one backend. Uses only features
present in ArangoDB CE (the C++ database server): doc collections, edge
collections, named graphs, AQL, and the COSINE_SIMILARITY function.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from arango import ArangoClient

from attestor.models import Memory
from attestor.store._graph_utils import sanitize_rel_type as _sanitize_rel_type
from attestor.store.connection import CloudConnection

logger = logging.getLogger("attestor")


class ArangoBackend:
    """Multi-role ArangoDB backend: document + vector + graph in one DB.

    Talks to ArangoDB Community Edition via python-arango. Uses only CE
    features: doc/edge collections, named graphs, AQL, COSINE_SIMILARITY.

    Accepts raw config dict. See CloudConnection.from_config() for formats.

    Supports:
        - Local Docker (mode=local): auto-creates database via _system
        - ArangoGraph cloud: TLS with CA cert (base64 or file path)
        - Self-hosted cloud (AWS/Azure/GCP): standard TLS
    """

    ROLES: ClassVar[set[str]] = {"document", "vector", "graph"}

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        conn = CloudConnection.from_config(config, backend_name="arangodb")
        self._conn = conn

        username = conn.auth.username
        password = conn.auth.password

        # Build raw python-arango client against ArangoDB CE
        verify = bool(config.get("tls", {}).get("verify", False))
        self._client = ArangoClient(hosts=conn.url, verify_override=verify)

        # Ensure target database exists (create via _system if missing)
        sys_db = self._client.db("_system", username=username, password=password)
        if not sys_db.has_database(conn.database):
            sys_db.create_database(conn.database)

        self._db = self._client.db(conn.database, username=username, password=password)

        # Ensure collections and graph exist (same schema as before)
        self._init_collections()
        self._init_graph()

        self._embedder = None  # lazy-init via shared embeddings module
        self._embedding_fn = None  # backward compat for benchmark code
        self._vector_index_created = False

    def _init_collections(self) -> None:
        """Create document collections and indexes."""
        if not self._db.has_collection("memories"):
            self._db.create_collection("memories")
        col = self._db.collection("memories")
        existing = set()
        for idx in col.indexes():
            if idx["type"] == "persistent":
                existing.update(idx.get("fields", []))
        for field_name in ["category", "entity", "status", "created_at"]:
            if field_name not in existing:
                col.add_index({"type": "persistent", "fields": [field_name]})

    def _init_graph(self) -> None:
        """Create graph structure: entities (vertices) + relations (edges)."""
        if not self._db.has_collection("entities"):
            self._db.create_collection("entities")
        if not self._db.has_collection("relations"):
            self._db.create_collection("relations", edge=True)
        if not self._db.has_graph("memory_graph"):
            self._db.create_graph(
                "memory_graph",
                edge_definitions=[{
                    "edge_collection": "relations",
                    "from_vertex_collections": ["entities"],
                    "to_vertex_collections": ["entities"],
                }],
            )

    # ── DocumentStore ──

    @staticmethod
    def _memory_to_doc(memory: Memory) -> dict[str, Any]:
        return {
            "_key": memory.id,
            "memory_id": memory.id,
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
            "metadata": memory.metadata,
        }

    def insert(self, memory: Memory) -> Memory:
        col = self._db.collection("memories")
        col.insert(self._memory_to_doc(memory))
        return memory

    def get(self, memory_id: str) -> Memory | None:
        col = self._db.collection("memories")
        doc = col.get(memory_id)
        if doc is None:
            return None
        return self._doc_to_memory(doc)

    def update(self, memory: Memory) -> Memory:
        col = self._db.collection("memories")
        col.replace(self._memory_to_doc(memory))
        return memory

    def delete(self, memory_id: str) -> bool:
        col = self._db.collection("memories")
        if not col.has(memory_id):
            return False
        col.delete(memory_id)
        return True

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
        filters = []
        bind_vars: dict[str, Any] = {"@col": "memories", "lim": limit}

        if status:
            filters.append("doc.status == @status")
            bind_vars["status"] = status
        if category:
            filters.append("doc.category == @category")
            bind_vars["category"] = category
        if entity:
            filters.append("doc.entity == @entity")
            bind_vars["entity"] = entity
        if namespace:
            filters.append("doc.namespace == @namespace")
            bind_vars["namespace"] = namespace
        if after:
            filters.append("doc.created_at >= @after")
            bind_vars["after"] = after
        if before:
            filters.append("doc.created_at <= @before")
            bind_vars["before"] = before

        where = " AND ".join(filters) if filters else "true"
        aql = f"FOR doc IN @@col FILTER {where} SORT doc.created_at DESC LIMIT @lim RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def tag_search(
        self,
        tags: list[str],
        category: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        bind_vars: dict[str, Any] = {"@col": "memories", "lim": limit, "tags": tags}
        filters = [
            "doc.status == 'active'",
            "doc.valid_until == null",
            "LENGTH(INTERSECTION(doc.tags, @tags)) > 0",
        ]
        if category:
            filters.append("doc.category == @category")
            bind_vars["category"] = category
        if namespace:
            filters.append("doc.namespace == @namespace")
            bind_vars["namespace"] = namespace

        where = " AND ".join(filters)
        aql = f"FOR doc IN @@col FILTER {where} SORT doc.created_at DESC LIMIT @lim RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def execute(
        self, query: str, params: Any | None = None
    ) -> list[dict[str, Any]]:
        """Execute raw AQL query."""
        bind_vars = params if isinstance(params, dict) else {}
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)
        return list(cursor)

    def archive_before(self, date: str) -> int:
        aql = """
        FOR doc IN memories
            FILTER doc.created_at < @date AND doc.status == 'active'
            UPDATE doc WITH { status: 'archived' } IN memories
            RETURN 1
        """
        cursor = self._db.aql.execute(aql, bind_vars={"date": date})
        return sum(1 for _ in cursor)

    def compact(self) -> int:
        aql = """
        FOR doc IN memories
            FILTER doc.status == 'archived'
            REMOVE doc IN memories
            RETURN 1
        """
        cursor = self._db.aql.execute(aql, bind_vars={})
        return sum(1 for _ in cursor)

    def stats(self) -> dict[str, Any]:
        total_cursor = self._db.aql.execute("RETURN LENGTH(memories)")
        total = next(total_cursor)

        by_status: dict[str, int] = {}
        status_cursor = self._db.aql.execute(
            "FOR doc IN memories COLLECT s = doc.status WITH COUNT INTO c RETURN {status: s, count: c}"
        )
        for row in status_cursor:
            by_status[row["status"]] = row["count"]

        by_category: dict[str, int] = {}
        cat_cursor = self._db.aql.execute(
            "FOR doc IN memories COLLECT c = doc.category WITH COUNT INTO cnt RETURN {category: c, count: cnt}"
        )
        for row in cat_cursor:
            by_category[row["category"]] = row["count"]

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }

    @staticmethod
    def _doc_to_memory(doc: dict[str, Any]) -> Memory:
        """Convert raw AQL document → Attestor Memory."""
        return Memory(
            id=doc["_key"],
            content=doc["content"],
            tags=doc.get("tags", []),
            category=doc.get("category", "general"),
            entity=doc.get("entity"),
            namespace=doc.get("namespace", "default"),
            created_at=doc["created_at"],
            event_date=doc.get("event_date"),
            valid_from=doc["valid_from"],
            valid_until=doc.get("valid_until"),
            superseded_by=doc.get("superseded_by"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "active"),
            metadata=doc.get("metadata", {}),
        )

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
        return self._embedder.embed(text)

    def add(self, memory_id: str, content: str, namespace: str = "default") -> None:
        """Generate embedding and store as vector_data on the memory doc."""
        embedding = self._embed(content)
        col = self._db.collection("memories")
        if col.has(memory_id):
            col.update({"_key": memory_id, "vector_data": embedding, "namespace": namespace})

    def search(self, query_text: str, limit: int = 20, namespace: str | None = None) -> list[dict[str, Any]]:
        """Vector similarity search using cosine similarity."""
        query_vec = self._embed(query_text)

        bind_vars: dict[str, Any] = {"query_vec": query_vec, "lim": limit}
        namespace_filter = ""
        if namespace:
            namespace_filter = "FILTER doc.namespace == @namespace"
            bind_vars["namespace"] = namespace

        aql = f"""
        FOR doc IN memories
            FILTER doc.vector_data != null
            {namespace_filter}
            LET score = COSINE_SIMILARITY(doc.vector_data, @query_vec)
            FILTER score != null
            SORT score DESC
            LIMIT @lim
            RETURN {{memory_id: doc._key, content: doc.content, distance: 1.0 - score}}
        """
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        cursor = self._db.aql.execute(
            "RETURN LENGTH(FOR doc IN memories FILTER doc.vector_data != null RETURN 1)"
        )
        return next(cursor)

    # ── GraphStore ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        key = name.lower()
        attrs = dict(attributes) if attributes else {}
        graph = self._db.graph("memory_graph")
        entities = graph.vertex_collection("entities")

        if entities.has(key):
            existing = entities.get(key)
            update_doc: dict[str, Any] = {"_key": key, "display_name": name}
            for k, v in attrs.items():
                update_doc[k] = v
            if existing.get("entity_type", "general") == "general" and entity_type != "general":
                update_doc["entity_type"] = entity_type
            entities.update(update_doc)
        else:
            doc = {"_key": key, "display_name": name, "entity_type": entity_type}
            doc.update(attrs)
            entities.insert(doc)

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from_key = from_entity.lower()
        to_key = to_entity.lower()

        graph = self._db.graph("memory_graph")
        entities = graph.vertex_collection("entities")
        relations = graph.edge_collection("relations")

        if not entities.has(from_key):
            entities.insert({"_key": from_key, "display_name": from_entity, "entity_type": "general"})
        if not entities.has(to_key):
            entities.insert({"_key": to_key, "display_name": to_entity, "entity_type": "general"})

        sanitized = _sanitize_rel_type(relation_type)
        edge_doc = {
            "_from": f"entities/{from_key}",
            "_to": f"entities/{to_key}",
            "relation_type": sanitized,
        }
        if metadata:
            edge_doc.update(metadata)
        relations.insert(edge_doc)

    def get_related(self, entity: str, depth: int = 2) -> list[str]:
        start = entity.lower()
        entities = self._db.collection("entities")
        if not entities.has(start):
            return []

        aql = """
        FOR v IN 1..@depth ANY @start GRAPH 'memory_graph'
            OPTIONS { bfs: true, uniqueVertices: 'global' }
            RETURN v.display_name
        """
        cursor = self._db.aql.execute(
            aql, bind_vars={"depth": depth, "start": f"entities/{start}"}
        )
        return [name for name in cursor if name]

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
        start = entity.lower()
        entities = self._db.collection("entities")
        if not entities.has(start):
            return {"entity": entity, "nodes": [], "edges": []}

        v_aql = """
        LET start_v = DOCUMENT(@start)
        LET traversed = (
            FOR v IN 1..@depth ANY @start GRAPH 'memory_graph'
                OPTIONS { bfs: true, uniqueVertices: 'global' }
                RETURN v
        )
        RETURN APPEND([start_v], traversed)
        """
        v_cursor = self._db.aql.execute(
            v_aql, bind_vars={"depth": depth, "start": f"entities/{start}"}
        )
        all_vertices = next(v_cursor, [])

        nodes = []
        node_keys = set()
        for v in all_vertices:
            if v and v.get("_key") not in node_keys:
                node_keys.add(v["_key"])
                nodes.append({
                    "name": v.get("display_name", v["_key"]),
                    "type": v.get("entity_type", "general"),
                    "key": v["_key"],
                })

        edges = []
        if node_keys:
            e_aql = """
            FOR e IN relations
                FILTER PARSE_IDENTIFIER(e._from).key IN @keys
                   AND PARSE_IDENTIFIER(e._to).key IN @keys
                RETURN {
                    source: PARSE_IDENTIFIER(e._from).key,
                    target: PARSE_IDENTIFIER(e._to).key,
                    type: e.relation_type
                }
            """
            e_cursor = self._db.aql.execute(e_aql, bind_vars={"keys": list(node_keys)})
            edges = list(e_cursor)

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
        if entity_type:
            aql = "FOR doc IN entities FILTER doc.entity_type == @et RETURN doc"
            cursor = self._db.aql.execute(aql, bind_vars={"et": entity_type})
        else:
            cursor = self._db.aql.execute("FOR doc IN entities RETURN doc")

        result = []
        for doc in cursor:
            attrs = {
                k: v for k, v in doc.items()
                if k not in ("_key", "_id", "_rev", "display_name", "entity_type")
            }
            result.append({
                "name": doc.get("display_name", doc["_key"]),
                "type": doc.get("entity_type", "general"),
                "key": doc["_key"],
                "attributes": attrs,
            })
        return result

    def get_edges(self, entity: str) -> list[dict[str, Any]]:
        key = entity.lower()
        entities_col = self._db.collection("entities")
        if not entities_col.has(key):
            return []

        aql = """
        LET outgoing = (
            FOR v, e IN 1..1 OUTBOUND @start GRAPH 'memory_graph'
                RETURN {
                    subject: DOCUMENT(e._from).display_name,
                    predicate: e.relation_type,
                    object: DOCUMENT(e._to).display_name,
                    event_date: e.event_date || ""
                }
        )
        LET incoming = (
            FOR v, e IN 1..1 INBOUND @start GRAPH 'memory_graph'
                RETURN {
                    subject: DOCUMENT(e._from).display_name,
                    predicate: e.relation_type,
                    object: DOCUMENT(e._to).display_name,
                    event_date: e.event_date || ""
                }
        )
        RETURN APPEND(outgoing, incoming)
        """
        cursor = self._db.aql.execute(aql, bind_vars={"start": f"entities/{key}"})
        all_edges = next(cursor, [])
        seen = set()
        result = []
        for edge in all_edges:
            triple = (edge["subject"], edge["predicate"], edge["object"])
            if triple not in seen:
                seen.add(triple)
                result.append(edge)
        return result

    def graph_stats(self) -> dict[str, Any]:
        """Graph-specific stats (separate from DocumentStore.stats)."""
        nodes_cursor = self._db.aql.execute("RETURN LENGTH(entities)")
        edges_cursor = self._db.aql.execute("RETURN LENGTH(relations)")
        nodes = next(nodes_cursor)
        edges = next(edges_cursor)

        types: dict[str, int] = {}
        type_cursor = self._db.aql.execute(
            "FOR doc IN entities COLLECT t = doc.entity_type WITH COUNT INTO c RETURN {type: t, count: c}"
        )
        for row in type_cursor:
            types[row["type"]] = row["count"]

        return {"nodes": nodes, "edges": edges, "types": types}

    def save(self) -> None:
        pass  # ArangoDB persists automatically

    def close(self) -> None:
        self._client.close()
