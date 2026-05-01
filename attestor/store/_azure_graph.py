"""NetworkX in-process graph-role mixin for AzureBackend (split from azure_backend.py).

This module is private — consumers should import ``AzureBackend`` from
``attestor.store.azure_backend``. The mixin is stateless: it operates on
``self._graph`` (a NetworkX MultiDiGraph) and ``self._entities_container`` /
``self._edges_container`` configured by ``AzureBackend.__init__``.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from attestor.store._graph_utils import sanitize_rel_type as _sanitize_rel_type

logger = logging.getLogger("attestor")


class _AzureGraphMixin:
    """NetworkX in-process graph role for AzureBackend (write-through to Cosmos)."""

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

    def _persist_edge(self, from_key: str, to_key: str, rel_type: str, metadata: dict[str, Any]) -> None:
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
        attributes: dict[str, Any] | None = None,
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
        metadata: dict[str, Any] | None = None,
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

    def get_related(self, entity: str, depth: int = 2) -> list[str]:
        """BFS traversal in both directions up to depth hops."""
        start = entity.lower()
        if not self._graph.has_node(start):
            return []

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: list[str] = []

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

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
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

    def get_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
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

    def get_edges(self, entity: str) -> list[dict[str, Any]]:
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

    def graph_stats(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for _nid, data in self._graph.nodes(data=True):
            t = data.get("entity_type", "general")
            types[t] = types.get(t, 0) + 1
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "types": types,
        }
