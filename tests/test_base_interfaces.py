"""Tests for abstract base interfaces."""

import pytest
from agent_memory.store.base import DocumentStore, VectorStore, GraphStore


class TestABCsCannotInstantiate:
    def test_document_store_is_abstract(self):
        with pytest.raises(TypeError):
            DocumentStore()

    def test_vector_store_is_abstract(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_graph_store_is_abstract(self):
        with pytest.raises(TypeError):
            GraphStore()


class ConcreteDocumentStore(DocumentStore):
    """Minimal concrete implementation for testing."""
    ROLES = {"document"}
    def insert(self, memory): return memory
    def get(self, memory_id): return None
    def update(self, memory): return memory
    def delete(self, memory_id): return False
    def list_memories(self, **kwargs): return []
    def tag_search(self, tags, **kwargs): return []
    def execute(self, query, params=None): return []
    def archive_before(self, date): return 0
    def compact(self): return 0
    def stats(self): return {}
    def close(self): pass


class ConcreteVectorStore(VectorStore):
    ROLES = {"vector"}
    def add(self, memory_id, content): pass
    def search(self, query_text, limit=20): return []
    def delete(self, memory_id): return False
    def count(self): return 0
    def close(self): pass


class ConcreteGraphStore(GraphStore):
    ROLES = {"graph"}
    def add_entity(self, name, entity_type="general", attributes=None): pass
    def add_relation(self, from_entity, to_entity, relation_type="related_to", metadata=None): pass
    def get_related(self, entity, depth=2): return []
    def get_subgraph(self, entity, depth=2): return {}
    def get_entities(self, entity_type=None): return []
    def get_edges(self, entity): return []
    def graph_stats(self): return {}
    def save(self): pass
    def close(self): pass


class TestConcreteImplementations:
    def test_document_store_instantiates(self):
        store = ConcreteDocumentStore()
        assert "document" in store.ROLES

    def test_vector_store_instantiates(self):
        store = ConcreteVectorStore()
        assert "vector" in store.ROLES

    def test_graph_store_instantiates(self):
        store = ConcreteGraphStore()
        assert "graph" in store.ROLES
