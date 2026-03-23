"""Live integration tests for ArangoDB backend (Oasis or local).

Requires:
    ARANGO_URL and ARANGO_PASSWORD environment variables.
    Optionally ARANGO_DATABASE (defaults to memwright_test).

Run:
    .venv/bin/pytest tests/test_arango_live.py -v

Skip when credentials are not available (CI-safe).
"""

import os
import uuid

import pytest

ARANGO_URL = os.environ.get("ARANGO_URL", "")
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
ARANGO_DATABASE = os.environ.get("ARANGO_DATABASE", "memwright_test")

pytestmark = pytest.mark.skipif(
    not ARANGO_URL or not ARANGO_PASSWORD,
    reason="ARANGO_URL and ARANGO_PASSWORD required",
)


@pytest.fixture(scope="module")
def backend():
    """Create ArangoBackend connected to live ArangoDB."""
    from agent_memory.store.arango_backend import ArangoBackend

    config = {
        "url": ARANGO_URL,
        "database": ARANGO_DATABASE,
        "auth": {"username": "root", "password": ARANGO_PASSWORD},
        "tls": {"verify": False},
    }
    be = ArangoBackend(config)
    yield be
    # Cleanup: drop test database
    try:
        from arango import ArangoClient
        client = ArangoClient(hosts=ARANGO_URL, verify_override=False)
        sys_db = client.db("_system", username="root", password=ARANGO_PASSWORD)
        if sys_db.has_database(ARANGO_DATABASE):
            sys_db.delete_database(ARANGO_DATABASE)
    except Exception:
        pass
    be.close()


@pytest.fixture
def memory_id():
    return f"test-{uuid.uuid4().hex[:12]}"


class TestArangoLiveDocument:
    def test_insert_and_get(self, backend, memory_id):
        from agent_memory.models import Memory

        mem = Memory(
            id=memory_id,
            content="ArangoDB live test memory",
            tags=["arango", "test"],
            category="test",
        )
        result = backend.insert(mem)
        assert result.id == memory_id

        fetched = backend.get(memory_id)
        assert fetched is not None
        assert fetched.content == "ArangoDB live test memory"
        assert "arango" in fetched.tags

    def test_update(self, backend, memory_id):
        from agent_memory.models import Memory

        mem = Memory(
            id=memory_id,
            content="original content",
            tags=["arango"],
            category="test",
        )
        backend.insert(mem)

        updated = Memory(
            id=memory_id,
            content="updated content",
            tags=["arango", "updated"],
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
            tags=["arango"],
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
            tags=["unique-tag-arango-xyz"],
            category="test",
        ))

        results = backend.tag_search(tags=["unique-tag-arango-xyz"])
        assert any(m.id == mid for m in results)

    def test_stats(self, backend):
        stats = backend.stats()
        assert "total_memories" in stats
        assert isinstance(stats["total_memories"], int)


class TestArangoLiveGraph:
    def test_add_entity_and_relation(self, backend):
        backend.add_entity("ArangoDB", entity_type="database")
        backend.add_entity("Oasis", entity_type="cloud_service")
        backend.add_relation("ArangoDB", "Oasis", "hosted_on")

        related = backend.get_related("ArangoDB", depth=1)
        assert "Oasis" in related

    def test_get_subgraph(self, backend):
        backend.add_entity("GraphNode1", entity_type="test")
        backend.add_entity("GraphNode2", entity_type="test")
        backend.add_relation("GraphNode1", "GraphNode2", "linked_to")

        subgraph = backend.get_subgraph("GraphNode1", depth=1)
        assert len(subgraph["nodes"]) >= 2
        assert len(subgraph["edges"]) >= 1

    def test_get_entities(self, backend):
        backend.add_entity("FilterTestArango", entity_type="test_filter")
        entities = backend.get_entities(entity_type="test_filter")
        assert any(e["name"] == "FilterTestArango" for e in entities)

    def test_get_edges(self, backend):
        backend.add_entity("EdgeFromA", entity_type="test")
        backend.add_entity("EdgeToA", entity_type="test")
        backend.add_relation("EdgeFromA", "EdgeToA", "connects")

        edges = backend.get_edges("EdgeFromA")
        assert any(e["predicate"] == "CONNECTS" for e in edges)

    def test_graph_stats(self, backend):
        stats = backend.graph_stats()
        assert stats["nodes"] >= 0
        assert stats["edges"] >= 0


class TestArangoLiveHealth:
    def test_connection_alive(self, backend):
        """Verify we can query ArangoDB."""
        result = backend.execute("RETURN 1")
        assert result == [1]

    def test_collections_exist(self, backend):
        """Verify all collections were created."""
        assert backend._db.has_collection("memories")
        assert backend._db.has_collection("entities")
        assert backend._db.has_collection("relations")
