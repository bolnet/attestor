"""Tests for NetworkX entity graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory.graph.networkx_graph import NetworkXGraph


@pytest.fixture
def graph(tmp_path: Path) -> NetworkXGraph:
    """Create a NetworkXGraph with a temporary directory."""
    return NetworkXGraph(tmp_path)


class TestNetworkXGraphInit:
    def test_initializes_empty_graph(self, graph: NetworkXGraph):
        stats = graph.graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_loads_existing_graph(self, tmp_path: Path):
        g1 = NetworkXGraph(tmp_path)
        g1.add_entity("Python", "tool")
        g1.save()

        g2 = NetworkXGraph(tmp_path)
        stats = g2.graph_stats()
        assert stats["nodes"] == 1


class TestAddEntity:
    def test_creates_node(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool", {"version": "3.14"})
        entities = graph.get_entities()
        assert len(entities) == 1
        assert entities[0]["name"] == "Python"
        assert entities[0]["type"] == "tool"

    def test_lowercase_key_preserves_display_name(self, graph: NetworkXGraph):
        graph.add_entity("FastAPI", "tool")
        entities = graph.get_entities()
        assert entities[0]["name"] == "FastAPI"
        assert entities[0]["key"] == "fastapi"

    def test_merge_updates_attributes(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool", {"version": "3.12"})
        graph.add_entity("Python", "tool", {"version": "3.14"})
        entities = graph.get_entities()
        assert len(entities) == 1
        assert entities[0]["attributes"]["version"] == "3.14"


class TestAddRelation:
    def test_creates_directed_edge(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Acme", "organization")
        graph.add_relation("Alice", "Acme", "works_at")
        stats = graph.graph_stats()
        assert stats["edges"] == 1

    def test_auto_creates_missing_nodes(self, graph: NetworkXGraph):
        graph.add_relation("Alice", "Acme", "works_at")
        stats = graph.graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_stores_metadata(self, graph: NetworkXGraph):
        graph.add_relation("Alice", "Acme", "works_at", {"since": "2020"})
        edges = graph.get_edges("Alice")
        assert len(edges) == 1
        assert edges[0]["predicate"] == "WORKS_AT"


class TestGetRelated:
    def test_depth_1_direct_connections(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Bob", "person")
        graph.add_entity("Carol", "person")
        graph.add_relation("Alice", "Bob", "knows")
        graph.add_relation("Bob", "Carol", "knows")

        related = graph.get_related("Alice", depth=1)
        assert "Bob" in related
        assert "Carol" not in related

    def test_depth_2_multi_hop(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Bob", "person")
        graph.add_entity("Carol", "person")
        graph.add_relation("Alice", "Bob", "knows")
        graph.add_relation("Bob", "Carol", "knows")

        related = graph.get_related("Alice", depth=2)
        assert "Bob" in related
        assert "Carol" in related

    def test_undirected_traversal(self, graph: NetworkXGraph):
        """get_related should traverse edges in both directions."""
        graph.add_entity("Alice", "person")
        graph.add_entity("Bob", "person")
        graph.add_relation("Alice", "Bob", "knows")

        # Bob should find Alice even though edge points Alice -> Bob
        related = graph.get_related("Bob", depth=1)
        assert "Alice" in related

    def test_unknown_entity_returns_empty(self, graph: NetworkXGraph):
        related = graph.get_related("nonexistent", depth=2)
        assert related == []

    def test_does_not_include_self(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Bob", "person")
        graph.add_relation("Alice", "Bob", "knows")

        related = graph.get_related("Alice", depth=1)
        assert "Alice" not in related


class TestGetSubgraph:
    def test_returns_expected_structure(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Bob", "person")
        graph.add_relation("Alice", "Bob", "knows")

        subgraph = graph.get_subgraph("Alice", depth=1)
        assert subgraph["entity"] == "Alice"
        assert len(subgraph["nodes"]) == 2  # Alice + Bob
        assert len(subgraph["edges"]) == 1


class TestGetEntities:
    def test_returns_all_entities(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool")
        graph.add_entity("Alice", "person")
        entities = graph.get_entities()
        assert len(entities) == 2

    def test_filters_by_type(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool")
        graph.add_entity("Alice", "person")
        graph.add_entity("JavaScript", "tool")
        tools = graph.get_entities(entity_type="tool")
        assert len(tools) == 2
        names = {e["name"] for e in tools}
        assert names == {"Python", "JavaScript"}


class TestGetEdges:
    def test_returns_typed_edges(self, graph: NetworkXGraph):
        graph.add_entity("Alice", "person")
        graph.add_entity("Acme", "organization")
        graph.add_relation("Alice", "Acme", "works_at", {"event_date": "2020-01"})

        edges = graph.get_edges("Alice")
        assert len(edges) == 1
        e = edges[0]
        assert e["subject"] == "Alice"
        assert e["predicate"] == "WORKS_AT"
        assert e["object"] == "Acme"
        assert e["event_date"] == "2020-01"

    def test_includes_incoming_edges(self, graph: NetworkXGraph):
        graph.add_relation("Alice", "Bob", "manages")
        edges = graph.get_edges("Bob")
        assert len(edges) == 1
        assert edges[0]["subject"] == "Alice"
        assert edges[0]["object"] == "Bob"


class TestPersistence:
    def test_save_creates_json_file(self, tmp_path: Path):
        g = NetworkXGraph(tmp_path)
        g.add_entity("Python", "tool")
        g.save()
        assert (tmp_path / "graph.json").exists()

    def test_round_trip_persistence(self, tmp_path: Path):
        g1 = NetworkXGraph(tmp_path)
        g1.add_entity("Alice", "person")
        g1.add_entity("Bob", "person")
        g1.add_relation("Alice", "Bob", "knows")
        g1.save()

        g2 = NetworkXGraph(tmp_path)
        stats = g2.graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        related = g2.get_related("Alice", depth=1)
        assert "Bob" in related

    def test_close_saves(self, tmp_path: Path):
        g = NetworkXGraph(tmp_path)
        g.add_entity("Python", "tool")
        g.close()
        assert (tmp_path / "graph.json").exists()


class TestStats:
    def test_node_and_edge_counts(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool")
        graph.add_entity("Alice", "person")
        graph.add_relation("Alice", "Python", "uses")
        stats = graph.graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_type_breakdown(self, graph: NetworkXGraph):
        graph.add_entity("Python", "tool")
        graph.add_entity("JavaScript", "tool")
        graph.add_entity("Alice", "person")
        stats = graph.graph_stats()
        assert stats["types"]["tool"] == 2
        assert stats["types"]["person"] == 1
