"""Abstract base interfaces for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from attestor.models import Memory


class DocumentStore(ABC):
    """Abstract document storage for Memory objects."""

    ROLES: Set[str] = set()

    @abstractmethod
    def insert(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]: ...

    @abstractmethod
    def update(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        namespace: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]: ...

    # NOTE: tag_search predates the semantic-first cascade and is no longer
    # called by the canonical recall pipeline (vector → BM25 → RRF → graph
    # → MMR → fit). It is kept on the interface for direct admin/UI lookup
    # by tag-set on backends that have a cheap tag index, and for backwards
    # compat with v3 callers. New callers should prefer recall() / search()
    # — those are what the orchestrator routes through.
    @abstractmethod
    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]: ...

    @abstractmethod
    def execute(
        self, query: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def archive_before(self, date: str) -> int: ...

    @abstractmethod
    def compact(self) -> int: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


class VectorStore(ABC):
    """Abstract vector embedding storage and similarity search."""

    ROLES: Set[str] = set()

    @abstractmethod
    def add(self, memory_id: str, content: str, namespace: str = "default") -> None: ...

    @abstractmethod
    def search(
        self, query_text: str, limit: int = 20, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...


class GraphStore(ABC):
    """Abstract entity graph with traversal."""

    ROLES: Set[str] = set()

    @abstractmethod
    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def get_related(self, entity: str, depth: int = 2) -> List[str]: ...

    @abstractmethod
    def get_subgraph(
        self, entity: str, depth: int = 2
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def get_entities(
        self, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_edges(self, entity: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def graph_stats(self) -> Dict[str, Any]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
