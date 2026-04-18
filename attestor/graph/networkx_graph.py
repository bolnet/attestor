"""NetworkX entity graph — zero-config in-process graph with JSON persistence."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from attestor.store.base import GraphStore


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type: uppercase, safe characters only."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return sanitized.upper()


class NetworkXGraph(GraphStore):
    """In-process entity graph using NetworkX MultiDiGraph with JSON persistence.

    Drop-in replacement for the old Neo4j graph backend.
    Node ids are lowercased; display_name preserves original case.
    """

    ROLES = {"graph"}

    def __init__(self, store_path: Path) -> None:
        self._path = store_path / "graph.json"
        self._graph = nx.MultiDiGraph()
        if self._path.exists():
            self.load()

    def _invalidate_pr_cache(self) -> None:
        """Mark PageRank cache as dirty after graph mutation."""
        self._pr_dirty = True

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or merge an entity node. Key is lowercased; display_name preserved."""
        self._invalidate_pr_cache()
        key = name.lower()
        attrs = dict(attributes) if attributes else {}
        if self._graph.has_node(key):
            existing = self._graph.nodes[key]
            # Merge attributes (new values overwrite)
            for k, v in attrs.items():
                existing[k] = v
            # Update display_name
            existing["display_name"] = name
            # Only overwrite entity_type if currently "general" and new is more specific
            if existing.get("entity_type", "general") == "general" and entity_type != "general":
                existing["entity_type"] = entity_type
        else:
            self._graph.add_node(
                key,
                display_name=name,
                entity_type=entity_type,
                **attrs,
            )
        self.save()

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add directed edge. Auto-creates nodes if missing."""
        self._invalidate_pr_cache()
        from_key = from_entity.lower()
        to_key = to_entity.lower()

        # Auto-create missing nodes
        if not self._graph.has_node(from_key):
            self._graph.add_node(
                from_key, display_name=from_entity, entity_type="general",
            )
        if not self._graph.has_node(to_key):
            self._graph.add_node(
                to_key, display_name=to_entity, entity_type="general",
            )

        sanitized = _sanitize_rel_type(relation_type)
        edge_attrs = dict(metadata) if metadata else {}
        edge_attrs["relation_type"] = sanitized
        self._graph.add_edge(from_key, to_key, key=sanitized, **edge_attrs)
        self.save()

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        """BFS traversal in both directions up to depth hops. Returns display names."""
        start = entity.lower()
        if not self._graph.has_node(start):
            return []

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: List[str] = []

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            # Successors (outgoing) and predecessors (incoming)
            neighbors = set(self._graph.successors(node)) | set(self._graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(self._graph.nodes[neighbor].get("display_name", neighbor))
                    queue.append((neighbor, d + 1))

        return result

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """Return subgraph around entity as {entity, nodes, edges}."""
        start = entity.lower()
        if not self._graph.has_node(start):
            return {"entity": entity, "nodes": [], "edges": []}

        # Collect reachable nodes via BFS (both directions)
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

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all entities, optionally filtered by type."""
        result = []
        for nid, data in self._graph.nodes(data=True):
            if entity_type and data.get("entity_type") != entity_type:
                continue
            # Separate known fields from extra attributes
            attrs = {
                k: v
                for k, v in data.items()
                if k not in ("display_name", "entity_type")
            }
            result.append({
                "name": data.get("display_name", nid),
                "type": data.get("entity_type", "general"),
                "key": nid,
                "attributes": attrs,
            })
        return result

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        """Return typed edges involving entity (both directions).

        Returns list of {subject, predicate, object, event_date}.
        """
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

    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Compute PageRank scores for all nodes.

        Returns dict mapping node keys (lowercased entity names) to scores.
        Uses cached result if graph hasn't changed since last computation.
        """
        if not hasattr(self, "_pr_cache"):
            self._pr_cache: Optional[Dict[str, float]] = None
            self._pr_dirty: bool = True

        if self._pr_cache is not None and not self._pr_dirty:
            return self._pr_cache

        if self._graph.number_of_nodes() == 0:
            self._pr_cache = {}
        else:
            self._pr_cache = nx.pagerank(self._graph, alpha=alpha)
        self._pr_dirty = False
        return self._pr_cache

    def graph_stats(self) -> Dict[str, Any]:
        """Return node count, edge count, and type breakdown."""
        types: Dict[str, int] = {}
        for _nid, data in self._graph.nodes(data=True):
            t = data.get("entity_type", "general")
            types[t] = types.get(t, 0) + 1
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "types": types,
        }

    def save(self, path: Optional[str] = None) -> None:
        """Serialize graph to JSON file."""
        target = Path(path) if path else self._path
        data = nx.node_link_data(self._graph)
        target.write_text(json.dumps(data, default=str, indent=2))

    def load(self, path: Optional[str] = None) -> None:
        """Load graph from JSON file."""
        target = Path(path) if path else self._path
        raw = json.loads(target.read_text())
        self._graph = nx.node_link_graph(raw, directed=True, multigraph=True)

    def close(self) -> None:
        """Save graph on close."""
        self.save()
