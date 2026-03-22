"""Live integration tests for Azure Cosmos DB backend.

Requires:
    AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY environment variables.

Run:
    .venv/bin/pytest tests/test_azure_live.py -v

Skip when credentials are not available (CI-safe).
"""

import os
import uuid

import pytest

COSMOS_ENDPOINT = os.environ.get("AZURE_COSMOS_ENDPOINT", "")
COSMOS_KEY = os.environ.get("AZURE_COSMOS_KEY", "")

pytestmark = pytest.mark.skipif(
    not COSMOS_ENDPOINT or not COSMOS_KEY,
    reason="AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY required",
)


@pytest.fixture(scope="module")
def backend():
    """Create AzureBackend connected to live Cosmos DB."""
    from agent_memory.store.azure_backend import AzureBackend

    config = {
        "cosmos_endpoint": COSMOS_ENDPOINT,
        "cosmos_key": COSMOS_KEY,
        "cosmos_database": "memwright_test",
    }
    be = AzureBackend(config)
    yield be
    # Cleanup: delete test database
    try:
        be._client.delete_database("memwright_test")
    except Exception:
        pass
    be.close()


@pytest.fixture
def memory_id():
    return f"test-{uuid.uuid4().hex[:12]}"


class TestAzureLiveDocument:
    def test_insert_and_get(self, backend, memory_id):
        from agent_memory.models import Memory

        mem = Memory(
            id=memory_id,
            content="Azure live test memory",
            tags=["azure", "test"],
            category="test",
        )
        result = backend.insert(mem)
        assert result.id == memory_id

        fetched = backend.get(memory_id)
        assert fetched is not None
        assert fetched.content == "Azure live test memory"
        assert "azure" in fetched.tags

    def test_update(self, backend, memory_id):
        from agent_memory.models import Memory

        mem = Memory(
            id=memory_id,
            content="original content",
            tags=["azure"],
            category="test",
        )
        backend.insert(mem)

        updated = Memory(
            id=memory_id,
            content="updated content",
            tags=["azure", "updated"],
            category="test",
        )
        backend.update(updated)

        fetched = backend.get(memory_id)
        assert fetched is not None
        assert fetched.content == "updated content"
        assert "updated" in fetched.tags

    def test_delete(self, backend, memory_id):
        from agent_memory.models import Memory

        mem = Memory(
            id=memory_id,
            content="to be deleted",
            tags=["azure"],
            category="test",
        )
        backend.insert(mem)
        assert backend.delete(memory_id) is True
        assert backend.get(memory_id) is None

    def test_delete_nonexistent(self, backend):
        assert backend.delete("nonexistent-id-12345") is False

    def test_list_memories(self, backend):
        from agent_memory.models import Memory

        ids = []
        for i in range(3):
            mid = f"list-test-{uuid.uuid4().hex[:8]}"
            ids.append(mid)
            backend.insert(Memory(
                id=mid,
                content=f"list test {i}",
                tags=["list-test"],
                category="test",
            ))

        memories = backend.list_memories(category="test")
        found_ids = {m.id for m in memories}
        for mid in ids:
            assert mid in found_ids

    def test_tag_search(self, backend):
        from agent_memory.models import Memory

        mid = f"tag-search-{uuid.uuid4().hex[:8]}"
        backend.insert(Memory(
            id=mid,
            content="tagged memory for search",
            tags=["unique-tag-xyz"],
            category="test",
        ))

        results = backend.tag_search(tags=["unique-tag-xyz"])
        assert any(m.id == mid for m in results)

    def test_stats(self, backend):
        stats = backend.stats()
        assert "total_memories" in stats
        assert isinstance(stats["total_memories"], int)


class TestAzureLiveGraph:
    def test_add_entity_and_relation(self, backend):
        backend.add_entity("Azure", entity_type="cloud_provider")
        backend.add_entity("CosmosDB", entity_type="database")
        backend.add_relation("Azure", "CosmosDB", "provides")

        related = backend.get_related("Azure", depth=1)
        assert "CosmosDB" in related

    def test_get_subgraph(self, backend):
        backend.add_entity("TestNode1", entity_type="test")
        backend.add_entity("TestNode2", entity_type="test")
        backend.add_relation("TestNode1", "TestNode2", "linked_to")

        subgraph = backend.get_subgraph("TestNode1", depth=1)
        assert len(subgraph["nodes"]) >= 2
        assert len(subgraph["edges"]) >= 1

    def test_get_entities(self, backend):
        backend.add_entity("FilterTest", entity_type="test_filter")
        entities = backend.get_entities(entity_type="test_filter")
        assert any(e["name"] == "FilterTest" for e in entities)

    def test_get_edges(self, backend):
        backend.add_entity("EdgeFrom", entity_type="test")
        backend.add_entity("EdgeTo", entity_type="test")
        backend.add_relation("EdgeFrom", "EdgeTo", "connects")

        edges = backend.get_edges("EdgeFrom")
        assert any(e["predicate"] == "CONNECTS" for e in edges)

    def test_graph_stats(self, backend):
        stats = backend.graph_stats()
        assert stats["nodes"] >= 0
        assert stats["edges"] >= 0


class TestAzureLiveHealth:
    def test_connection_alive(self, backend):
        """Verify we can query Cosmos DB."""
        result = backend.execute("SELECT VALUE 1")
        assert result == [1]

    def test_containers_exist(self, backend):
        """Verify all 3 containers were created."""
        assert backend._memories_container is not None
        assert backend._entities_container is not None
        assert backend._edges_container is not None
