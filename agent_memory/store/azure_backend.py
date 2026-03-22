"""Azure Cosmos DB backend — document + vector (DiskANN) + graph (NetworkX) in one account.

Uses Cosmos DB NoSQL API for document and vector storage, with an in-memory
NetworkX graph persisted to Cosmos containers for the graph role.

Requires: azure-cosmos, azure-identity (optional, for DefaultAzureCredential)
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore, GraphStore, VectorStore

logger = logging.getLogger("agent_memory")


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type: uppercase, safe characters only."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return sanitized.upper()


class AzureBackend(DocumentStore, VectorStore, GraphStore):
    """Multi-role Azure Cosmos DB backend: document + vector + graph.

    Architecture:
        - DocumentStore: Cosmos DB NoSQL API, container "memories", PK /category
        - VectorStore: DiskANN vector index on /embedding field (same container)
        - GraphStore: Cosmos containers "graph_entities" (PK /entity_type) and
          "graph_edges" (PK /from_key) + in-memory NetworkX MultiDiGraph

    Config keys:
        cosmos_endpoint: Cosmos DB account endpoint (or env AZURE_COSMOS_ENDPOINT)
        cosmos_key: Cosmos DB account key (or env AZURE_COSMOS_KEY)
        cosmos_database: Database name (default "memwright")

    When no cosmos_key is provided, falls back to DefaultAzureCredential.
    """

    ROLES: Set[str] = {"document", "vector", "graph"}

    def __init__(self, config: Dict[str, Any]) -> None:
        import os

        self._config = config

        # Resolve endpoint and key
        self._endpoint = (
            config.get("cosmos_endpoint")
            or os.environ.get("AZURE_COSMOS_ENDPOINT", "")
        )
        if not self._endpoint:
            raise ValueError(
                "Azure Cosmos DB endpoint required. Set cosmos_endpoint in config "
                "or AZURE_COSMOS_ENDPOINT environment variable."
            )

        cosmos_key = (
            config.get("cosmos_key")
            or config.get("auth", {}).get("api_key")
            or os.environ.get("AZURE_COSMOS_KEY", "")
        )
        self._database_name = config.get("cosmos_database", "memwright")

        # Lazy import azure.cosmos
        from azure.cosmos import CosmosClient, PartitionKey

        self._PartitionKey = PartitionKey

        if cosmos_key:
            self._client = CosmosClient(self._endpoint, credential=cosmos_key)
        else:
            # Fall back to DefaultAzureCredential (managed identity, CLI, etc.)
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            self._client = CosmosClient(self._endpoint, credential=credential)

        self._init_database()
        self._init_containers()
        self._init_graph_memory()

        # Embedding provider — lazy init
        self._embedder = None

    # ── Initialization ──

    def _init_database(self) -> None:
        """Ensure database exists (create if not)."""
        self._database = self._client.create_database_if_not_exists(
            id=self._database_name
        )

    def _init_containers(self) -> None:
        """Create containers with appropriate partition keys and vector policies."""
        # Memories container — PK on /category
        # Vector embedding policy for DiskANN index on /embedding
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536,  # text-embedding-3-small default
                }
            ]
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/embedding/*"}],
            "vectorIndexes": [
                {"path": "/embedding", "type": "diskANN"}
            ],
        }
        self._memories_container = self._database.create_container_if_not_exists(
            id="memories",
            partition_key=self._PartitionKey("/category"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
        )

        # Graph entities container — PK on /entity_type
        self._entities_container = self._database.create_container_if_not_exists(
            id="graph_entities",
            partition_key=self._PartitionKey("/entity_type"),
        )

        # Graph edges container — PK on /from_key
        self._edges_container = self._database.create_container_if_not_exists(
            id="graph_edges",
            partition_key=self._PartitionKey("/from_key"),
        )

    def _init_graph_memory(self) -> None:
        """Load all entities and edges from Cosmos into an in-memory NetworkX graph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for the graph role. Install with: pip install networkx")

        self._graph = nx.MultiDiGraph()

        # Load entities
        try:
            entities = list(self._entities_container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            ))
            for entity in entities:
                key = entity.get("key", "")
                attrs = {
                    k: v for k, v in entity.items()
                    if k not in ("id", "key", "display_name", "entity_type", "_rid", "_self", "_etag", "_attachments", "_ts")
                }
                self._graph.add_node(
                    key,
                    display_name=entity.get("display_name", key),
                    entity_type=entity.get("entity_type", "general"),
                    **attrs,
                )
        except Exception as e:
            logger.debug("Could not load graph entities: %s", e)

        # Load edges
        try:
            edges = list(self._edges_container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            ))
            for edge in edges:
                from_key = edge.get("from_key", "")
                to_key = edge.get("to_key", "")
                rel_type = edge.get("relation_type", "RELATED_TO")
                meta = {
                    k: v for k, v in edge.items()
                    if k not in ("id", "from_key", "to_key", "relation_type", "_rid", "_self", "_etag", "_attachments", "_ts")
                }
                meta["relation_type"] = rel_type
                self._graph.add_edge(from_key, to_key, key=rel_type, **meta)
        except Exception as e:
            logger.debug("Could not load graph edges: %s", e)

    # ── Embedding Helpers ──

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init embedding provider via shared module."""
        if self._embedder is not None:
            return
        from agent_memory.store.embeddings import get_embedding_provider

        self._embedder = get_embedding_provider("azure_openai")

    def _embed(self, text: str) -> List[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        return self._embedder.embed(text)

    # ── DocumentStore ──

    def _memory_to_doc(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory to Cosmos DB document. id field is Cosmos PK."""
        return {
            "id": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "entity": memory.entity,
            "created_at": memory.created_at,
            "event_date": memory.event_date,
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
            "superseded_by": memory.superseded_by,
            "confidence": memory.confidence,
            "status": memory.status,
            "metadata": memory.metadata,
        }

    def _doc_to_memory(self, doc: Dict[str, Any]) -> Memory:
        """Convert Cosmos DB document to Memory."""
        return Memory(
            id=doc["id"],
            content=doc["content"],
            tags=doc.get("tags", []),
            category=doc.get("category", "general"),
            entity=doc.get("entity"),
            created_at=doc["created_at"],
            event_date=doc.get("event_date"),
            valid_from=doc["valid_from"],
            valid_until=doc.get("valid_until"),
            superseded_by=doc.get("superseded_by"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "active"),
            metadata=doc.get("metadata", {}),
        )

    def insert(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        self._memories_container.create_item(body=doc)
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        try:
            # Cross-partition point read when category unknown
            items = list(self._memories_container.query_items(
                query="SELECT * FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": memory_id}],
                enable_cross_partition_query=True,
            ))
            if not items:
                return None
            return self._doc_to_memory(items[0])
        except Exception:
            return None

    def update(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        # Preserve existing embedding if present
        existing = self.get(memory.id)
        if existing:
            try:
                existing_items = list(self._memories_container.query_items(
                    query="SELECT * FROM c WHERE c.id = @id",
                    parameters=[{"name": "@id", "value": memory.id}],
                    enable_cross_partition_query=True,
                ))
                if existing_items and "embedding" in existing_items[0]:
                    doc["embedding"] = existing_items[0]["embedding"]
            except Exception:
                pass
        self._memories_container.upsert_item(body=doc)
        return memory

    def delete(self, memory_id: str) -> bool:
        try:
            items = list(self._memories_container.query_items(
                query="SELECT c.id, c.category FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": memory_id}],
                enable_cross_partition_query=True,
            ))
            if not items:
                return False
            category = items[0]["category"]
            self._memories_container.delete_item(item=memory_id, partition_key=category)
            return True
        except Exception:
            return False

    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        filters: List[str] = []
        params: List[Dict[str, Any]] = []

        if status:
            filters.append("c.status = @status")
            params.append({"name": "@status", "value": status})
        if category:
            filters.append("c.category = @category")
            params.append({"name": "@category", "value": category})
        if entity:
            filters.append("c.entity = @entity")
            params.append({"name": "@entity", "value": entity})
        if after:
            filters.append("c.created_at >= @after")
            params.append({"name": "@after", "value": after})
        if before:
            filters.append("c.created_at <= @before")
            params.append({"name": "@before", "value": before})

        where = " AND ".join(filters) if filters else "1=1"
        query = f"SELECT * FROM c WHERE {where} ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})

        items = list(self._memories_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        ))
        return [self._doc_to_memory(item) for item in items]

    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        # Build tag filter: check if any tag in the list is in c.tags
        tag_conditions = []
        params: List[Dict[str, Any]] = []
        for i, tag in enumerate(tags):
            param_name = f"@tag{i}"
            tag_conditions.append(f"ARRAY_CONTAINS(c.tags, {param_name})")
            params.append({"name": param_name, "value": tag})

        tag_filter = f"({' OR '.join(tag_conditions)})"
        filters = [
            "c.status = 'active'",
            "NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until)",
            tag_filter,
        ]

        if category:
            filters.append("c.category = @category")
            params.append({"name": "@category", "value": category})

        where = " AND ".join(filters)
        query = f"SELECT * FROM c WHERE {where} ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})

        items = list(self._memories_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        ))
        return [self._doc_to_memory(item) for item in items]

    def execute(
        self, query: str, params: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw Cosmos SQL query."""
        parameters = params if isinstance(params, list) else []
        items = list(self._memories_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))
        return items

    def archive_before(self, date: str) -> int:
        """Archive memories created before the given date."""
        items = list(self._memories_container.query_items(
            query="SELECT * FROM c WHERE c.created_at < @date AND c.status = 'active'",
            parameters=[{"name": "@date", "value": date}],
            enable_cross_partition_query=True,
        ))
        count = 0
        for item in items:
            item["status"] = "archived"
            self._memories_container.upsert_item(body=item)
            count += 1
        return count

    def compact(self) -> int:
        """Delete all archived memories."""
        items = list(self._memories_container.query_items(
            query="SELECT c.id, c.category FROM c WHERE c.status = 'archived'",
            parameters=[],
            enable_cross_partition_query=True,
        ))
        count = 0
        for item in items:
            try:
                self._memories_container.delete_item(
                    item=item["id"], partition_key=item["category"]
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to delete archived item %s: %s", item["id"], e)
        return count

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total_items = list(self._memories_container.query_items(
            query="SELECT VALUE COUNT(1) FROM c",
            enable_cross_partition_query=True,
        ))
        total = total_items[0] if total_items else 0

        by_status: Dict[str, int] = {}
        status_items = list(self._memories_container.query_items(
            query="SELECT c.status, COUNT(1) AS cnt FROM c GROUP BY c.status",
            enable_cross_partition_query=True,
        ))
        for row in status_items:
            by_status[row["status"]] = row["cnt"]

        by_category: Dict[str, int] = {}
        cat_items = list(self._memories_container.query_items(
            query="SELECT c.category, COUNT(1) AS cnt FROM c GROUP BY c.category",
            enable_cross_partition_query=True,
        ))
        for row in cat_items:
            by_category[row["category"]] = row["cnt"]

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }

    # ── VectorStore ──

    def add(self, memory_id: str, content: str) -> None:
        """Generate embedding and patch onto the memory document."""
        embedding = self._embed(content)
        # Read the existing document, add embedding, upsert
        items = list(self._memories_container.query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": memory_id}],
            enable_cross_partition_query=True,
        ))
        if not items:
            return
        doc = items[0]
        doc["embedding"] = embedding
        self._memories_container.upsert_item(body=doc)

    def search(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Vector similarity search using Cosmos DB DiskANN VectorDistance."""
        query_vec = self._embed(query_text)
        query = (
            "SELECT TOP @limit c.id, c.content, "
            "VectorDistance(c.embedding, @queryVector, false, "
            "{'distanceFunction': 'cosine'}) AS distance "
            "FROM c WHERE IS_DEFINED(c.embedding) "
            "ORDER BY VectorDistance(c.embedding, @queryVector, false, "
            "{'distanceFunction': 'cosine'})"
        )
        items = list(self._memories_container.query_items(
            query=query,
            parameters=[
                {"name": "@limit", "value": limit},
                {"name": "@queryVector", "value": query_vec},
            ],
            enable_cross_partition_query=True,
        ))
        return [
            {
                "memory_id": item["id"],
                "content": item["content"],
                "distance": item.get("distance", 0.0),
            }
            for item in items
        ]

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        items = list(self._memories_container.query_items(
            query="SELECT VALUE COUNT(1) FROM c WHERE IS_DEFINED(c.embedding)",
            enable_cross_partition_query=True,
        ))
        return items[0] if items else 0

    # ── GraphStore ──

    def _persist_entity(self, key: str) -> None:
        """Write-through entity from NetworkX to Cosmos."""
        if not self._graph.has_node(key):
            return
        data = dict(self._graph.nodes[key])
        doc = {
            "id": key,
            "key": key,
            "display_name": data.get("display_name", key),
            "entity_type": data.get("entity_type", "general"),
        }
        # Include extra attributes
        for k, v in data.items():
            if k not in ("display_name", "entity_type"):
                doc[k] = v
        self._entities_container.upsert_item(body=doc)

    def _persist_edge(self, from_key: str, to_key: str, rel_type: str, metadata: Dict[str, Any]) -> None:
        """Write-through edge from NetworkX to Cosmos."""
        import hashlib

        # Deterministic edge id from from/to/type
        edge_id = hashlib.md5(
            f"{from_key}:{to_key}:{rel_type}".encode()
        ).hexdigest()[:16]

        doc = {
            "id": edge_id,
            "from_key": from_key,
            "to_key": to_key,
            "relation_type": rel_type,
        }
        doc.update(metadata)
        self._edges_container.upsert_item(body=doc)

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = name.lower()
        attrs = dict(attributes) if attributes else {}

        if self._graph.has_node(key):
            existing = self._graph.nodes[key]
            for k, v in attrs.items():
                existing[k] = v
            existing["display_name"] = name
            if existing.get("entity_type", "general") == "general" and entity_type != "general":
                existing["entity_type"] = entity_type
        else:
            self._graph.add_node(
                key,
                display_name=name,
                entity_type=entity_type,
                **attrs,
            )

        self._persist_entity(key)

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        from_key = from_entity.lower()
        to_key = to_entity.lower()

        # Auto-create missing nodes
        if not self._graph.has_node(from_key):
            self._graph.add_node(
                from_key, display_name=from_entity, entity_type="general",
            )
            self._persist_entity(from_key)
        if not self._graph.has_node(to_key):
            self._graph.add_node(
                to_key, display_name=to_entity, entity_type="general",
            )
            self._persist_entity(to_key)

        sanitized = _sanitize_rel_type(relation_type)
        edge_attrs = dict(metadata) if metadata else {}
        edge_attrs["relation_type"] = sanitized
        self._graph.add_edge(from_key, to_key, key=sanitized, **edge_attrs)
        self._persist_edge(from_key, to_key, sanitized, edge_attrs)

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        """BFS traversal in both directions up to depth hops."""
        start = entity.lower()
        if not self._graph.has_node(start):
            return []

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: List[str] = []

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            neighbors = set(self._graph.successors(node)) | set(self._graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(self._graph.nodes[neighbor].get("display_name", neighbor))
                    queue.append((neighbor, d + 1))

        return result

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """Return subgraph around entity as {entity, nodes, edges}."""
        start = entity.lower()
        if not self._graph.has_node(start):
            return {"entity": entity, "nodes": [], "edges": []}

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            neighbors = set(self._graph.successors(node)) | set(self._graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        nodes = []
        for nid in visited:
            data = self._graph.nodes[nid]
            nodes.append({
                "name": data.get("display_name", nid),
                "type": data.get("entity_type", "general"),
                "key": nid,
            })

        edges = []
        for u, v, _key, data in self._graph.edges(keys=True, data=True):
            if u in visited and v in visited:
                edges.append({
                    "source": u,
                    "target": v,
                    "type": data.get("relation_type", "RELATED_TO"),
                })

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for nid, data in self._graph.nodes(data=True):
            if entity_type and data.get("entity_type") != entity_type:
                continue
            attrs = {
                k: v for k, v in data.items()
                if k not in ("display_name", "entity_type")
            }
            result.append({
                "name": data.get("display_name", nid),
                "type": data.get("entity_type", "general"),
                "key": nid,
                "attributes": attrs,
            })
        return result

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        key = entity.lower()
        if not self._graph.has_node(key):
            return []

        result = []
        seen: set[tuple[str, str, str]] = set()

        # Outgoing edges
        for _, target, data in self._graph.edges(key, data=True):
            rel = data.get("relation_type", "RELATED_TO")
            triple = (key, rel, target)
            if triple not in seen:
                seen.add(triple)
                result.append({
                    "subject": self._graph.nodes[key].get("display_name", key),
                    "predicate": rel,
                    "object": self._graph.nodes[target].get("display_name", target),
                    "event_date": data.get("event_date", ""),
                })

        # Incoming edges
        for source, _, data in self._graph.in_edges(key, data=True):
            rel = data.get("relation_type", "RELATED_TO")
            triple = (source, rel, key)
            if triple not in seen:
                seen.add(triple)
                result.append({
                    "subject": self._graph.nodes[source].get("display_name", source),
                    "predicate": rel,
                    "object": self._graph.nodes[key].get("display_name", key),
                    "event_date": data.get("event_date", ""),
                })

        return result

    def graph_stats(self) -> Dict[str, Any]:
        types: Dict[str, int] = {}
        for _nid, data in self._graph.nodes(data=True):
            t = data.get("entity_type", "general")
            types[t] = types.get(t, 0) + 1
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "types": types,
        }

    def save(self) -> None:
        """No-op — writes are immediate (write-through to Cosmos)."""
        pass

    def close(self) -> None:
        """Close the Cosmos client."""
        try:
            self._client.close()
        except Exception:
            pass
