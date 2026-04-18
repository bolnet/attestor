"""Tests for Azure Cosmos DB backend — fully mocked, no Azure services required.

Run with: .venv/bin/pytest tests/test_azure_backend.py -v
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from attestor.models import Memory

try:
    import azure.cosmos  # noqa: F401
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

azure_required = pytest.mark.skipif(
    not HAS_AZURE, reason="azure-cosmos not installed"
)


# ═══════════════════════════════════════════════════════════════════════
# Mock Cosmos infrastructure
# ═══════════════════════════════════════════════════════════════════════


class MockContainer:
    """In-memory mock of a Cosmos DB container."""

    def __init__(self, id: str, partition_key_path: str = "/id") -> None:
        self.id = id
        self._pk_path = partition_key_path
        self._items: Dict[str, Dict[str, Any]] = {}

    def create_item(self, body: Dict[str, Any]) -> Dict[str, Any]:
        item_id = body["id"]
        if item_id in self._items:
            raise Exception(f"Item {item_id} already exists")
        self._items[item_id] = dict(body)
        return body

    def upsert_item(self, body: Dict[str, Any]) -> Dict[str, Any]:
        self._items[body["id"]] = dict(body)
        return body

    def read_item(self, item: str, partition_key: Any) -> Dict[str, Any]:
        if item not in self._items:
            raise Exception(f"Item {item} not found")
        return dict(self._items[item])

    def delete_item(self, item: str, partition_key: Any) -> None:
        if item not in self._items:
            raise Exception(f"Item {item} not found")
        del self._items[item]

    def query_items(
        self,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        enable_cross_partition_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """Simple query engine for testing — handles basic WHERE clauses."""
        params = {p["name"]: p["value"] for p in (parameters or [])}
        items = list(self._items.values())

        # Parse SELECT VALUE COUNT(1) queries
        if "COUNT(1)" in query.upper() and "VALUE" in query.upper():
            filtered = self._apply_where(items, query, params)
            return [len(filtered)]

        # Parse SELECT ... GROUP BY queries
        if "GROUP BY" in query.upper():
            return self._handle_group_by(items, query, params)

        # Parse SELECT TOP queries
        limit = None
        if "TOP @limit" in query:
            limit = params.get("@limit")
        elif "LIMIT @limit" in query:
            limit = params.get("@limit")

        # Filter items
        filtered = self._apply_where(items, query, params)

        # Apply ORDER BY created_at DESC if present
        if "ORDER BY c.created_at DESC" in query:
            filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply limit
        if limit is not None:
            filtered = filtered[:limit]

        # Handle field projection (SELECT c.id, c.category)
        if "SELECT *" not in query and "SELECT VALUE" not in query and "SELECT TOP" not in query:
            fields = self._parse_select_fields(query)
            if fields:
                filtered = [{f: item.get(f) for f in fields} for item in filtered]

        return [dict(item) for item in filtered]

    def _apply_where(
        self, items: List[Dict[str, Any]], query: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply WHERE clause filters."""
        result = list(items)

        if "WHERE" not in query.upper():
            return result

        where_part = query.upper().split("WHERE", 1)[1]
        # Remove ORDER BY, LIMIT, OFFSET parts
        for keyword in ["ORDER BY", "OFFSET", "LIMIT", "GROUP BY"]:
            if keyword in where_part:
                where_part = where_part.split(keyword)[0]

        filtered = []
        for item in result:
            if self._matches_where(item, query, params):
                filtered.append(item)
        return filtered

    def _matches_where(
        self, item: Dict[str, Any], query: str, params: Dict[str, Any]
    ) -> bool:
        """Check if an item matches WHERE conditions."""
        # Extract the WHERE clause
        if "WHERE" not in query.upper():
            return True

        where_idx = query.upper().index("WHERE")
        where_clause = query[where_idx + 5:]
        # Remove trailing clauses
        for keyword in ["ORDER BY", "OFFSET", "LIMIT", "GROUP BY"]:
            idx = where_clause.upper().find(keyword)
            if idx >= 0:
                where_clause = where_clause[:idx]

        conditions = where_clause.strip()

        # Split on AND
        parts = [p.strip() for p in conditions.split("AND")]

        for part in parts:
            part = part.strip()
            if not part or part == "1=1":
                continue

            # Handle c.field = @param
            if "=" in part and "@" in part and "!" not in part and ">" not in part and "<" not in part:
                field = part.split("=")[0].strip().replace("c.", "").strip()
                param = part.split("=")[1].strip()
                # Handle string literals
                if param.startswith("'") and param.endswith("'"):
                    expected = param[1:-1]
                elif param in params:
                    expected = params[param]
                else:
                    continue
                if item.get(field) != expected:
                    return False

            # Handle c.field < @param
            elif "<" in part and "@" in part and "=" not in part.replace("<=", ""):
                if "<=" in part:
                    field = part.split("<=")[0].strip().replace("c.", "").strip()
                    param = part.split("<=")[1].strip()
                    if param in params and str(item.get(field, "")) > str(params[param]):
                        return False
                else:
                    field = part.split("<")[0].strip().replace("c.", "").strip()
                    param = part.split("<")[1].strip()
                    if param in params and str(item.get(field, "")) >= str(params[param]):
                        return False

            # Handle c.field >= @param
            elif ">=" in part and "@" in part:
                field = part.split(">=")[0].strip().replace("c.", "").strip()
                param = part.split(">=")[1].strip()
                if param in params and str(item.get(field, "")) < str(params[param]):
                    return False

            # Handle IS_DEFINED(c.embedding)
            elif "IS_DEFINED(c.embedding)" in part:
                if "embedding" not in item:
                    return False

            # Handle NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until)
            elif "IS_NULL(c.valid_until)" in part:
                if item.get("valid_until") is not None:
                    return False

            # Handle ARRAY_CONTAINS
            elif "ARRAY_CONTAINS" in part:
                for pname, pval in params.items():
                    if pname in part:
                        tags = item.get("tags", [])
                        if pval not in tags:
                            # This is an OR condition among tag checks,
                            # handled at a higher level
                            pass

        # Special case: tag search with OR
        if "ARRAY_CONTAINS" in conditions:
            tags = item.get("tags", [])
            any_match = False
            for pname, pval in params.items():
                if pname.startswith("@tag") and pval in tags:
                    any_match = True
                    break
            tag_params = [p for p in params if p.startswith("@tag")]
            if tag_params and not any_match:
                return False

        return True

    def _handle_group_by(
        self, items: List[Dict[str, Any]], query: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle simple GROUP BY queries."""
        # Extract group field
        group_part = query.upper().split("GROUP BY")[1].strip()
        group_field = group_part.split()[0].replace("C.", "").strip()

        groups: Dict[str, int] = {}
        for item in items:
            key = item.get(group_field, "unknown")
            groups[key] = groups.get(key, 0) + 1

        return [
            {group_field: k, "cnt": v}
            for k, v in groups.items()
        ]

    def _parse_select_fields(self, query: str) -> List[str]:
        """Parse field names from SELECT clause."""
        select_part = query.split("FROM")[0].replace("SELECT", "").strip()
        fields = []
        for f in select_part.split(","):
            f = f.strip().replace("c.", "")
            if f and not f.startswith("VectorDistance"):
                fields.append(f)
        return fields


class MockDatabase:
    """In-memory mock of a Cosmos DB database."""

    def __init__(self, id: str) -> None:
        self.id = id
        self._containers: Dict[str, MockContainer] = {}

    def create_container_if_not_exists(
        self,
        id: str,
        partition_key: Any = None,
        indexing_policy: Any = None,
        vector_embedding_policy: Any = None,
    ) -> MockContainer:
        if id not in self._containers:
            pk_path = "/id"
            if partition_key and hasattr(partition_key, "path"):
                pk_path = partition_key.path
            self._containers[id] = MockContainer(id, pk_path)
        return self._containers[id]


class MockCosmosClient:
    """In-memory mock of CosmosClient."""

    def __init__(self, endpoint: str, credential: Any = None) -> None:
        self.endpoint = endpoint
        self._databases: Dict[str, MockDatabase] = {}

    def create_database_if_not_exists(self, id: str) -> MockDatabase:
        if id not in self._databases:
            self._databases[id] = MockDatabase(id)
        return self._databases[id]

    def close(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_cosmos():
    """Patch azure.cosmos.CosmosClient with our mock."""
    with patch("azure.cosmos.CosmosClient", MockCosmosClient):
        with patch("azure.cosmos.PartitionKey") as pk_mock:
            pk_mock.side_effect = lambda path: MagicMock(path=path)
            yield


@pytest.fixture
def azure_backend(mock_cosmos):
    """Create an AzureBackend with mocked Cosmos client."""
    from attestor.store.azure_backend import AzureBackend

    config = {
        "cosmos_endpoint": "https://test.documents.azure.com:443",
        "cosmos_key": "test-key-abc123==",
        "cosmos_database": "memwright_test",
    }
    backend = AzureBackend(config)
    return backend


# ═══════════════════════════════════════════════════════════════════════
# DocumentStore Tests
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureDocumentStore:
    def test_insert_and_get(self, azure_backend):
        m = Memory(content="test fact", tags=["a"], category="test")
        azure_backend.insert(m)
        retrieved = azure_backend.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "test fact"
        assert retrieved.tags == ["a"]
        assert retrieved.category == "test"

    def test_get_nonexistent(self, azure_backend):
        result = azure_backend.get("nonexistent-id")
        assert result is None

    def test_update(self, azure_backend):
        m = Memory(content="original", tags=["a"])
        azure_backend.insert(m)

        updated = Memory(
            id=m.id, content="updated", tags=["b"],
            category=m.category, created_at=m.created_at, valid_from=m.valid_from,
        )
        azure_backend.update(updated)
        retrieved = azure_backend.get(m.id)
        assert retrieved.content == "updated"
        assert retrieved.tags == ["b"]

    def test_delete(self, azure_backend):
        m = Memory(content="to delete")
        azure_backend.insert(m)
        assert azure_backend.delete(m.id)
        assert azure_backend.get(m.id) is None

    def test_delete_nonexistent(self, azure_backend):
        assert not azure_backend.delete("nonexistent-id")

    def test_list_with_status_filter(self, azure_backend):
        azure_backend.insert(Memory(content="a", category="career", status="active"))
        azure_backend.insert(Memory(content="b", category="pref", status="active"))
        azure_backend.insert(Memory(content="c", category="career", status="archived"))

        active = azure_backend.list_memories(status="active")
        assert len(active) == 2
        assert all(m.status == "active" for m in active)

    def test_list_with_category_filter(self, azure_backend):
        azure_backend.insert(Memory(content="a", category="career", status="active"))
        azure_backend.insert(Memory(content="b", category="pref", status="active"))

        career = azure_backend.list_memories(category="career")
        assert len(career) == 1
        assert career[0].content == "a"

    def test_tag_search(self, azure_backend):
        azure_backend.insert(
            Memory(content="python stuff", tags=["python", "coding"], status="active")
        )
        azure_backend.insert(
            Memory(content="career stuff", tags=["career"], status="active")
        )

        results = azure_backend.tag_search(["python"])
        assert len(results) >= 1
        assert any(r.content == "python stuff" for r in results)

    def test_tag_search_multiple_tags(self, azure_backend):
        azure_backend.insert(
            Memory(content="python", tags=["python"], status="active")
        )
        azure_backend.insert(
            Memory(content="java", tags=["java"], status="active")
        )
        azure_backend.insert(
            Memory(content="rust", tags=["rust"], status="active")
        )

        results = azure_backend.tag_search(["python", "java"])
        assert len(results) == 2

    def test_stats(self, azure_backend):
        azure_backend.insert(Memory(content="a", category="career"))
        azure_backend.insert(Memory(content="b", category="preference"))
        s = azure_backend.stats()
        assert s["total_memories"] == 2
        assert "by_status" in s
        assert "by_category" in s

    def test_archive_before(self, azure_backend):
        azure_backend.insert(Memory(content="old fact"))
        count = azure_backend.archive_before("2099-01-01")
        assert count >= 1

    def test_compact(self, azure_backend):
        m = Memory(content="archived fact", status="archived")
        azure_backend.insert(m)
        removed = azure_backend.compact()
        assert removed >= 1
        assert azure_backend.get(m.id) is None

    def test_execute_raw_query(self, azure_backend):
        azure_backend.insert(Memory(id="exec1", content="exec test"))
        results = azure_backend.execute(
            "SELECT * FROM c WHERE c.id = @id",
            [{"name": "@id", "value": "exec1"}],
        )
        assert len(results) >= 1
        assert results[0]["content"] == "exec test"


# ═══════════════════════════════════════════════════════════════════════
# VectorStore Tests
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureVectorStore:
    def test_add_embedding(self, azure_backend):
        """Test that add() patches the document with an embedding field."""
        azure_backend.insert(Memory(id="v1", content="test content"))

        # Mock the embedder
        azure_backend._embedder = MagicMock()
        azure_backend._embedder.embed.return_value = [0.1] * 1536

        azure_backend.add("v1", "test content")

        # Verify embedding was stored
        doc = azure_backend._memories_container._items.get("v1")
        assert doc is not None
        assert "embedding" in doc
        assert len(doc["embedding"]) == 1536

    def test_add_nonexistent_memory(self, azure_backend):
        """Adding embedding for nonexistent memory is a no-op."""
        azure_backend._embedder = MagicMock()
        azure_backend._embedder.embed.return_value = [0.1] * 1536
        azure_backend.add("nonexistent", "content")
        # Should not raise

    def test_count(self, azure_backend):
        """Count only documents with embeddings."""
        azure_backend.insert(Memory(id="vc1", content="has embedding"))
        azure_backend.insert(Memory(id="vc2", content="no embedding"))

        # Manually add embedding to one
        azure_backend._memories_container._items["vc1"]["embedding"] = [0.1] * 10

        count = azure_backend.count()
        assert count == 1

    def test_search_query_construction(self, azure_backend):
        """Verify search calls embed and queries with VectorDistance."""
        azure_backend._embedder = MagicMock()
        azure_backend._embedder.embed.return_value = [0.1] * 1536

        # Insert a memory with embedding
        azure_backend.insert(Memory(id="vs1", content="searchable content"))
        azure_backend._memories_container._items["vs1"]["embedding"] = [0.1] * 1536

        # The mock container doesn't implement VectorDistance, but we can
        # verify the embed call happens
        azure_backend._embedder.embed.assert_not_called()
        azure_backend.search("find this", limit=5)
        azure_backend._embedder.embed.assert_called_once_with("find this")


# ═══════════════════════════════════════════════════════════════════════
# GraphStore Tests
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureGraphStore:
    def test_add_entity(self, azure_backend):
        azure_backend.add_entity("Alice", entity_type="person")
        entities = azure_backend.get_entities(entity_type="person")
        assert any(e["name"] == "Alice" for e in entities)

    def test_add_entity_merge(self, azure_backend):
        azure_backend.add_entity("Bob", entity_type="general")
        azure_backend.add_entity("Bob", entity_type="person", attributes={"age": "30"})
        entities = azure_backend.get_entities()
        bob = next(e for e in entities if e["name"] == "Bob")
        assert bob["type"] == "person"

    def test_add_entity_preserves_specific_type(self, azure_backend):
        """If entity already has a specific type, general should not overwrite."""
        azure_backend.add_entity("Charlie", entity_type="person")
        azure_backend.add_entity("Charlie", entity_type="general")
        entities = azure_backend.get_entities()
        charlie = next(e for e in entities if e["name"] == "Charlie")
        assert charlie["type"] == "person"

    def test_add_relation(self, azure_backend):
        azure_backend.add_entity("Alice", entity_type="person")
        azure_backend.add_entity("Bob", entity_type="person")
        azure_backend.add_relation("Alice", "Bob", relation_type="knows")
        edges = azure_backend.get_edges("Alice")
        assert any(e["predicate"] == "KNOWS" for e in edges)

    def test_add_relation_auto_creates_nodes(self, azure_backend):
        azure_backend.add_relation("X", "Y", relation_type="links_to")
        entities = azure_backend.get_entities()
        names = [e["name"] for e in entities]
        assert "X" in names
        assert "Y" in names

    def test_get_related_bfs(self, azure_backend):
        azure_backend.add_entity("A")
        azure_backend.add_entity("B")
        azure_backend.add_entity("C")
        azure_backend.add_relation("A", "B")
        azure_backend.add_relation("B", "C")
        related = azure_backend.get_related("A", depth=2)
        assert "B" in related
        assert "C" in related

    def test_get_related_nonexistent(self, azure_backend):
        result = azure_backend.get_related("nonexistent")
        assert result == []

    def test_get_related_depth_limit(self, azure_backend):
        azure_backend.add_relation("D1", "D2")
        azure_backend.add_relation("D2", "D3")
        azure_backend.add_relation("D3", "D4")
        related_d1 = azure_backend.get_related("D1", depth=1)
        assert "D2" in related_d1
        assert "D3" not in related_d1

    def test_get_subgraph(self, azure_backend):
        azure_backend.add_entity("X")
        azure_backend.add_entity("Y")
        azure_backend.add_relation("X", "Y")
        subgraph = azure_backend.get_subgraph("X", depth=1)
        assert len(subgraph["nodes"]) >= 2
        assert len(subgraph["edges"]) >= 1

    def test_get_subgraph_nonexistent(self, azure_backend):
        subgraph = azure_backend.get_subgraph("nonexistent")
        assert subgraph == {"entity": "nonexistent", "nodes": [], "edges": []}

    def test_graph_stats(self, azure_backend):
        azure_backend.add_entity("StatsNode", entity_type="test")
        stats = azure_backend.graph_stats()
        assert stats["nodes"] >= 1
        assert "types" in stats
        assert stats["types"].get("test", 0) >= 1

    def test_get_edges_incoming(self, azure_backend):
        azure_backend.add_relation("Source", "Target", relation_type="points_to")
        edges = azure_backend.get_edges("Target")
        assert len(edges) >= 1
        assert edges[0]["predicate"] == "POINTS_TO"

    def test_save_noop(self, azure_backend):
        """save() should not raise (it's a no-op for write-through)."""
        azure_backend.save()

    def test_entity_persisted_to_cosmos(self, azure_backend):
        """Verify write-through to Cosmos container."""
        azure_backend.add_entity("Persisted", entity_type="person")
        items = azure_backend._entities_container._items
        assert "persisted" in items
        assert items["persisted"]["entity_type"] == "person"

    def test_edge_persisted_to_cosmos(self, azure_backend):
        """Verify write-through edge to Cosmos container."""
        azure_backend.add_relation("From", "To", relation_type="connects")
        items = azure_backend._edges_container._items
        assert len(items) >= 1
        edge = list(items.values())[0]
        assert edge["from_key"] == "from"
        assert edge["to_key"] == "to"
        assert edge["relation_type"] == "CONNECTS"


# ═══════════════════════════════════════════════════════════════════════
# Container Auto-creation Tests
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureContainerCreation:
    def test_containers_created_on_init(self, azure_backend):
        """Verify all three containers are created during init."""
        assert azure_backend._memories_container is not None
        assert azure_backend._entities_container is not None
        assert azure_backend._edges_container is not None

    def test_database_created_on_init(self, azure_backend):
        """Verify database is created if not exists."""
        assert azure_backend._database is not None
        assert azure_backend._database.id == "memwright_test"


# ═══════════════════════════════════════════════════════════════════════
# Auth Tests
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureAuth:
    def test_api_key_auth(self, mock_cosmos):
        """API key auth uses CosmosClient with key credential."""
        from attestor.store.azure_backend import AzureBackend

        backend = AzureBackend({
            "cosmos_endpoint": "https://test.documents.azure.com:443",
            "cosmos_key": "my-api-key==",
        })
        assert backend._client is not None

    def test_default_credential_auth(self, mock_cosmos):
        """When no key is provided, DefaultAzureCredential is used."""
        from attestor.store.azure_backend import AzureBackend

        with patch("azure.identity.DefaultAzureCredential") as mock_cred:
            mock_cred.return_value = MagicMock()
            backend = AzureBackend({
                "cosmos_endpoint": "https://test.documents.azure.com:443",
            })
            mock_cred.assert_called_once()

    def test_endpoint_from_env(self, mock_cosmos):
        """Endpoint can come from AZURE_COSMOS_ENDPOINT env var."""
        import os
        from attestor.store.azure_backend import AzureBackend

        with patch.dict(os.environ, {"AZURE_COSMOS_ENDPOINT": "https://env.documents.azure.com:443"}):
            backend = AzureBackend({
                "cosmos_key": "test-key==",
            })
            assert backend._endpoint == "https://env.documents.azure.com:443"

    def test_key_from_env(self, mock_cosmos):
        """Key can come from AZURE_COSMOS_KEY env var."""
        import os
        from attestor.store.azure_backend import AzureBackend

        with patch.dict(os.environ, {"AZURE_COSMOS_KEY": "env-key=="}):
            backend = AzureBackend({
                "cosmos_endpoint": "https://test.documents.azure.com:443",
            })
            assert backend._client is not None

    def test_missing_endpoint_raises(self, mock_cosmos):
        """Missing endpoint raises ValueError."""
        import os
        from attestor.store.azure_backend import AzureBackend

        with patch.dict(os.environ, {}, clear=True):
            # Remove any env vars that might provide endpoint
            env = dict(os.environ)
            env.pop("AZURE_COSMOS_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="endpoint required"):
                    AzureBackend({"cosmos_key": "test-key=="})

    def test_key_from_auth_config(self, mock_cosmos):
        """Key can come from nested auth.api_key."""
        from attestor.store.azure_backend import AzureBackend

        backend = AzureBackend({
            "cosmos_endpoint": "https://test.documents.azure.com:443",
            "auth": {"api_key": "nested-key=="},
        })
        assert backend._client is not None


# ═══════════════════════════════════════════════════════════════════════
# Graph Load/Persist Cycle
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureGraphPersistence:
    def test_graph_loads_from_cosmos_on_init(self, mock_cosmos):
        """Entities and edges stored in Cosmos are loaded into NetworkX on init."""
        from attestor.store.azure_backend import AzureBackend

        # Create a backend and add some graph data
        config = {
            "cosmos_endpoint": "https://test.documents.azure.com:443",
            "cosmos_key": "test-key==",
            "cosmos_database": "persist_test",
        }
        backend1 = AzureBackend(config)
        backend1.add_entity("Alice", entity_type="person")
        backend1.add_entity("Bob", entity_type="person")
        backend1.add_relation("Alice", "Bob", relation_type="knows")

        # Verify data is in Cosmos containers
        assert "alice" in backend1._entities_container._items
        assert "bob" in backend1._entities_container._items
        assert len(backend1._edges_container._items) >= 1

        # Simulate creating a new backend instance that reads the same containers
        # by passing the same mock containers
        backend2 = AzureBackend(config)
        # backend2 gets fresh containers (different mock), so it won't have the data
        # This tests the init path -- in production the Cosmos containers persist

        # For a true persistence test, manually set the items before init
        backend2._entities_container._items = dict(backend1._entities_container._items)
        backend2._edges_container._items = dict(backend1._edges_container._items)
        backend2._init_graph_memory()

        # Verify graph was loaded
        assert backend2._graph.has_node("alice")
        assert backend2._graph.has_node("bob")
        related = backend2.get_related("Alice", depth=1)
        assert "Bob" in related


# ═══════════════════════════════════════════════════════════════════════
# Registry Integration
# ═══════════════════════════════════════════════════════════════════════


class TestAzureRegistry:
    def test_azure_in_backend_registry(self):
        from attestor.store.registry import BACKEND_REGISTRY
        assert "azure" in BACKEND_REGISTRY
        entry = BACKEND_REGISTRY["azure"]
        assert entry["roles"] == {"document", "vector", "graph"}
        assert entry["init_style"] == "config"
        assert entry["class"] == "AzureBackend"

    def test_azure_in_engine_defaults(self):
        from attestor.store.connection import ENGINE_DEFAULTS
        assert "azure" in ENGINE_DEFAULTS
        defaults = ENGINE_DEFAULTS["azure"]
        assert defaults["cosmos_database"] == "memwright"
        assert defaults["tls"]["verify"] is True

    def test_resolve_backends_azure(self):
        from attestor.store.registry import resolve_backends
        roles = resolve_backends(["azure"])
        assert roles == {"document": "azure", "vector": "azure", "graph": "azure"}


# ═══════════════════════════════════════════════════════════════════════
# Close
# ═══════════════════════════════════════════════════════════════════════


@azure_required
class TestAzureClose:
    def test_close(self, azure_backend):
        azure_backend.close()
        # Should not raise on double close
        azure_backend.close()
