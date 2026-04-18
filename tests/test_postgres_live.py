"""Live integration tests for PostgresBackend on Neon (or any PostgreSQL with pgvector).

Requires:
    NEON_DATABASE_URL environment variable (or any PostgreSQL connection URL).

Run:
    NEON_DATABASE_URL='postgresql://...' .venv/bin/pytest tests/test_postgres_live.py -v

Skip when credentials are not available (CI-safe).
"""

import os
import uuid

import pytest

DATABASE_URL = os.environ.get("NEON_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="NEON_DATABASE_URL required",
)


@pytest.fixture(scope="module")
def backend():
    """Create PostgresBackend connected to Neon."""
    from attestor.store.postgres_backend import PostgresBackend

    config = {"url": DATABASE_URL, "sslmode": "require"}
    be = PostgresBackend(config)
    yield be
    # Cleanup: delete all test memories
    try:
        be._execute("DELETE FROM memories WHERE category = 'test'")
    except Exception:
        pass
    be.close()


@pytest.fixture
def memory_id():
    return f"test-{uuid.uuid4().hex[:12]}"


class TestPostgresLiveDocument:
    def test_insert_and_get(self, backend, memory_id):
        from attestor.models import Memory

        mem = Memory(
            id=memory_id,
            content="Neon live test memory",
            tags=["neon", "test"],
            category="test",
        )
        result = backend.insert(mem)
        assert result.id == memory_id

        fetched = backend.get(memory_id)
        assert fetched is not None
        assert fetched.content == "Neon live test memory"
        assert "neon" in fetched.tags

    def test_update(self, backend, memory_id):
        from attestor.models import Memory

        mem = Memory(
            id=memory_id,
            content="original",
            tags=["neon"],
            category="test",
        )
        backend.insert(mem)

        updated = Memory(
            id=memory_id,
            content="updated",
            tags=["neon", "updated"],
            category="test",
        )
        backend.update(updated)

        fetched = backend.get(memory_id)
        assert fetched is not None
        assert fetched.content == "updated"

    def test_delete(self, backend, memory_id):
        from attestor.models import Memory

        mem = Memory(
            id=memory_id,
            content="to be deleted",
            tags=["neon"],
            category="test",
        )
        backend.insert(mem)
        assert backend.delete(memory_id) is True
        assert backend.get(memory_id) is None

    def test_list_memories(self, backend):
        from attestor.models import Memory

        ids = []
        for i in range(3):
            mid = f"list-{uuid.uuid4().hex[:8]}"
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
        from attestor.models import Memory

        mid = f"tag-{uuid.uuid4().hex[:8]}"
        backend.insert(Memory(
            id=mid,
            content="tagged for search",
            tags=["unique-neon-tag"],
            category="test",
        ))

        results = backend.tag_search(tags=["unique-neon-tag"])
        assert any(m.id == mid for m in results)

    def test_stats(self, backend):
        stats = backend.stats()
        assert "total_memories" in stats
        assert isinstance(stats["total_memories"], int)


class TestPostgresLiveVector:
    def test_add_and_search(self, backend):
        from attestor.models import Memory

        mid = f"vec-{uuid.uuid4().hex[:8]}"
        backend.insert(Memory(
            id=mid,
            content="machine learning and neural networks",
            tags=["ml"],
            category="test",
        ))
        backend.add(mid, "machine learning and neural networks")

        results = backend.search("deep learning AI", limit=5)
        assert len(results) > 0
        assert any(r["memory_id"] == mid for r in results)

    def test_vector_count(self, backend):
        count = backend.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_similar_texts_rank_higher(self, backend):
        from attestor.models import Memory

        mid1 = f"sim-{uuid.uuid4().hex[:8]}"
        mid2 = f"diff-{uuid.uuid4().hex[:8]}"
        backend.insert(Memory(id=mid1, content="python programming language", tags=["code"], category="test"))
        backend.insert(Memory(id=mid2, content="chocolate cake recipe", tags=["food"], category="test"))
        backend.add(mid1, "python programming language")
        backend.add(mid2, "chocolate cake recipe")

        results = backend.search("coding in python", limit=5)
        ids = [r["memory_id"] for r in results]
        # Python content should rank before cake recipe
        if mid1 in ids and mid2 in ids:
            assert ids.index(mid1) < ids.index(mid2)


class TestPostgresLiveGraphDisabled:
    """Verify graph methods raise NotImplementedError when AGE is unavailable."""

    def test_age_not_available(self, backend):
        assert backend._has_age is False

    def test_add_entity_raises(self, backend):
        with pytest.raises(NotImplementedError, match="Apache AGE"):
            backend.add_entity("test")

    def test_add_relation_raises(self, backend):
        with pytest.raises(NotImplementedError, match="Apache AGE"):
            backend.add_relation("a", "b")

    def test_get_related_raises(self, backend):
        with pytest.raises(NotImplementedError, match="Apache AGE"):
            backend.get_related("test")

    def test_graph_stats_raises(self, backend):
        with pytest.raises(NotImplementedError, match="Apache AGE"):
            backend.graph_stats()
