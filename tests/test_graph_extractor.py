"""Tests for the graph entity/relation extractor."""

import pytest

from attestor.graph.extractor import extract_entities_and_relations


class TestExtraction:
    def test_extract_entity_from_entity_field(self):
        nodes, edges = extract_entities_and_relations(
            content="Works at Anthropic",
            tags=[],
            entity="Alice",
            category="career",
        )
        # Should have at least Alice as a node
        names = [n["name"] for n in nodes]
        assert "Alice" in names

    def test_extract_uses_pattern(self):
        nodes, edges = extract_entities_and_relations(
            content="Team uses Python for backend",
            tags=[],
            entity="team",
            category="technical",
        )
        edge_types = [e["type"] for e in edges]
        assert "uses" in edge_types

    def test_extract_prefers_pattern(self):
        nodes, edges = extract_entities_and_relations(
            content="User prefers vim over vscode",
            tags=[],
            entity="user",
            category="preference",
        )
        edge_types = [e["type"] for e in edges]
        assert "prefers" in edge_types

    def test_extract_works_at_pattern(self):
        nodes, edges = extract_entities_and_relations(
            content="Alice works at Google",
            tags=[],
            entity="Alice",
            category="career",
        )
        edge_types = [e["type"] for e in edges]
        assert "works_at" in edge_types

    def test_extract_from_tags(self):
        nodes, edges = extract_entities_and_relations(
            content="Some content",
            tags=["Python", "React"],
            entity="project",
            category="technical",
        )
        names = [n["name"] for n in nodes]
        assert "Python" in names
        assert "React" in names

    def test_extract_known_tools_from_tags(self):
        nodes, edges = extract_entities_and_relations(
            content="Some content",
            tags=["python", "react"],
            entity=None,
            category="general",
        )
        # Lowercase known tools should be recognized
        names = [n["name"] for n in nodes]
        assert "python" in names

    def test_no_entity_no_crash(self):
        nodes, edges = extract_entities_and_relations(
            content="Just some text",
            tags=[],
            entity=None,
            category="general",
        )
        # Should not crash, may return empty
        assert isinstance(nodes, list)
        assert isinstance(edges, list)

    def test_tag_relation_to_entity(self):
        nodes, edges = extract_entities_and_relations(
            content="Some content",
            tags=["Python"],
            entity="Alice",
            category="technical",
        )
        # Should create a relation from Alice to Python
        rel_edges = [e for e in edges if e["from"] == "Alice" and e["to"] == "Python"]
        assert len(rel_edges) >= 1
