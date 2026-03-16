"""Tests for ChromaDB vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory.store.chroma_store import ChromaStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    """Create a ChromaStore with a temporary directory."""
    return ChromaStore(tmp_path)


class TestChromaStoreInit:
    def test_creates_chroma_directory(self, tmp_path: Path):
        ChromaStore(tmp_path)
        assert (tmp_path / "chroma").exists()

    def test_auto_provisions_collection(self, store: ChromaStore):
        # Should be usable immediately without setup
        assert store.count() == 0


class TestChromaStoreAdd:
    def test_add_stores_content(self, store: ChromaStore):
        store.add("mem-1", "The user prefers dark mode")
        assert store.count() == 1

    def test_add_multiple(self, store: ChromaStore):
        store.add("mem-1", "First memory")
        store.add("mem-2", "Second memory")
        assert store.count() == 2

    def test_add_upsert_same_id(self, store: ChromaStore):
        store.add("mem-1", "Original content")
        store.add("mem-1", "Updated content")
        assert store.count() == 1
        results = store.search("Updated", limit=1)
        assert results[0]["content"] == "Updated content"


class TestChromaStoreSearch:
    def test_search_returns_matching_results(self, store: ChromaStore):
        store.add("mem-1", "The user prefers Python for backend development")
        store.add("mem-2", "The user likes dark mode in their editor")
        results = store.search("What programming language does the user prefer?", limit=5)
        assert len(results) > 0
        assert any(r["memory_id"] == "mem-1" for r in results)

    def test_search_result_keys(self, store: ChromaStore):
        store.add("mem-1", "Some content here")
        results = store.search("content", limit=5)
        assert len(results) == 1
        r = results[0]
        assert "memory_id" in r
        assert "content" in r
        assert "distance" in r

    def test_search_sorted_by_distance(self, store: ChromaStore):
        store.add("mem-1", "Python is a programming language")
        store.add("mem-2", "The weather is sunny today")
        store.add("mem-3", "JavaScript is used for web development")
        results = store.search("programming languages", limit=3)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_search_respects_limit(self, store: ChromaStore):
        for i in range(10):
            store.add(f"mem-{i}", f"Memory number {i} about things")
        results = store.search("memory", limit=3)
        assert len(results) == 3

    def test_search_empty_store(self, store: ChromaStore):
        results = store.search("anything", limit=5)
        assert results == []


class TestChromaStoreDelete:
    def test_delete_removes_embedding(self, store: ChromaStore):
        store.add("mem-1", "Some content")
        store.delete("mem-1")
        assert store.count() == 0

    def test_delete_nonexistent_returns_false(self, store: ChromaStore):
        result = store.delete("nonexistent")
        assert result is False


class TestChromaStorePersistence:
    def test_data_persists_across_instances(self, tmp_path: Path):
        store1 = ChromaStore(tmp_path)
        store1.add("mem-1", "Persistent data")
        store1.close()

        store2 = ChromaStore(tmp_path)
        assert store2.count() == 1
        results = store2.search("Persistent", limit=1)
        assert len(results) == 1
        assert results[0]["memory_id"] == "mem-1"
