"""Structural interfaces for storage backends.

These are :class:`typing.Protocol` definitions, not :class:`abc.ABC`
hierarchies — backends satisfy them via duck typing (matching method
shapes) and are *not* required to inherit from them. The registry at
``attestor/store/registry.py`` dispatches on each backend's ``ROLES``
class attribute, never on ``isinstance`` against these protocols.

``@runtime_checkable`` is applied so ``isinstance(obj, DocumentStore)``
remains a valid (if discouraged) check; it is included defensively
even though no caller in the tree currently relies on it.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from attestor.models import Memory


@runtime_checkable
class DocumentStore(Protocol):
    """Document role: source-of-truth for content, tags, entity, ts, provenance, confidence."""

    # Subclasses MUST define ROLES (e.g., {"document"} or
    # {"document", "vector", "graph"}). The empty default is removed so
    # that a backend forgetting to declare its roles fails loudly at
    # registry validation time rather than silently claiming nothing.
    ROLES: ClassVar[set[str]]

    def insert(self, memory: Memory) -> Memory: ...

    # ``requester_agent_id`` (Gap A1) — optional visibility filter:
    # when set, the read drops rows where
    # ``visibility='private' AND agent_id != requester_agent_id``.
    # Backends that don't carry visibility (v3 schema) ignore it.
    def get(
        self,
        memory_id: str,
        requester_agent_id: str | None = None,
    ) -> Memory | None: ...

    def update(self, memory: Memory) -> Memory: ...

    def delete(self, memory_id: str) -> bool: ...

    def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
        requester_agent_id: str | None = None,
    ) -> list[Memory]: ...

    # NOTE: tag_search predates the semantic-first cascade and is no longer
    # called by the canonical recall pipeline (vector → BM25 → RRF → graph
    # → MMR → fit). It is kept on the interface for direct admin/UI lookup
    # by tag-set on backends that have a cheap tag index, and for backwards
    # compat with v3 callers. New callers should prefer recall() / search()
    # — those are what the orchestrator routes through.
    def tag_search(
        self,
        tags: list[str],
        category: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        requester_agent_id: str | None = None,
    ) -> list[Memory]: ...

    def execute(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def archive_before(self, date: str) -> int: ...

    def compact(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector role: dense embedding storage and similarity search."""

    # Subclasses MUST define ROLES (e.g., {"vector"} or
    # {"document", "vector", "graph"}).
    ROLES: ClassVar[set[str]]

    def add(self, memory_id: str, content: str, namespace: str = "default") -> None: ...

    def search(
        self, query_text: str, limit: int = 20, namespace: str | None = None
    ) -> list[dict[str, Any]]: ...

    def delete(self, memory_id: str) -> bool: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    """Graph role: entity nodes + typed edges with traversal."""

    # Subclasses MUST define ROLES (e.g., {"graph"}).
    ROLES: ClassVar[set[str]]

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: dict[str, Any] | None = None,
    ) -> None: ...

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def get_related(self, entity: str, depth: int = 2) -> list[str]: ...

    def get_subgraph(
        self, entity: str, depth: int = 2
    ) -> dict[str, Any]: ...

    def get_entities(
        self, entity_type: str | None = None
    ) -> list[dict[str, Any]]: ...

    def get_edges(self, entity: str) -> list[dict[str, Any]]: ...

    def graph_stats(self) -> dict[str, Any]: ...

    def save(self) -> None: ...

    def close(self) -> None: ...
