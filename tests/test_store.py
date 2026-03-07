"""Tests for SQLite store."""

import tempfile

import pytest

from agent_memory.models import Memory
from agent_memory.store.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        s = SQLiteStore(f"{d}/memory.db")
        yield s
        s.close()


class TestMemoryCRUD:
    def test_insert_and_get(self, store):
        m = Memory(content="test fact", tags=["a"], category="test")
        store.insert(m)
        retrieved = store.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "test fact"
        assert retrieved.tags == ["a"]

    def test_update(self, store):
        m = Memory(content="original", tags=["a"])
        store.insert(m)
        m.content = "updated"
        store.update(m)
        retrieved = store.get(m.id)
        assert retrieved.content == "updated"

    def test_delete(self, store):
        m = Memory(content="to delete")
        store.insert(m)
        assert store.delete(m.id)
        assert store.get(m.id) is None

    def test_delete_nonexistent(self, store):
        assert not store.delete("nonexistent")

    def test_list_with_filters(self, store):
        store.insert(Memory(content="a", category="career", status="active"))
        store.insert(Memory(content="b", category="preference", status="active"))
        store.insert(Memory(content="c", category="career", status="archived"))

        active = store.list_memories(status="active")
        assert len(active) == 2

        career = store.list_memories(category="career")
        assert len(career) == 2

        active_career = store.list_memories(status="active", category="career")
        assert len(active_career) == 1


class TestTagSearch:
    def test_tag_search(self, store):
        store.insert(Memory(content="a", tags=["python", "coding"], status="active"))
        store.insert(Memory(content="b", tags=["career", "job"], status="active"))

        results = store.tag_search(["python"])
        assert len(results) >= 1
        assert results[0].content == "a"

    def test_tag_search_multiple(self, store):
        store.insert(Memory(content="a", tags=["python", "coding"], status="active"))
        store.insert(Memory(content="b", tags=["career"], status="active"))

        results = store.tag_search(["python", "career"])
        assert len(results) == 2


class TestStats:
    def test_stats(self, store):
        store.insert(Memory(content="a", category="career"))
        store.insert(Memory(content="b", category="preference"))
        s = store.stats()
        assert s["total_memories"] == 2
        assert "active" in s["by_status"]


class TestBulkOps:
    def test_archive_before(self, store):
        store.insert(Memory(content="old"))
        count = store.archive_before("2099-01-01")
        assert count >= 1

    def test_compact(self, store):
        m = Memory(content="archived", status="archived")
        store.insert(m)
        removed = store.compact()
        assert removed == 1
