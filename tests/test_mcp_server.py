"""Tests for MCP server: tools, resources, and prompts."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from attestor.mcp.server import _handle_tool
from attestor.models import Memory, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_memory(**overrides) -> Memory:
    defaults = dict(
        id="mem1",
        content="User prefers Python",
        tags=["preference"],
        category="preference",
        entity="user",
        created_at="2026-01-01T00:00:00Z",
        event_date="2026-01-01",
        status="active",
    )
    defaults.update(overrides)
    return Memory(**defaults)


def _mock_agent_memory(*, graph_available=True):
    """Create a mock AgentMemory with controllable _graph presence."""
    mem = MagicMock()

    # Graph setup
    if graph_available:
        mem._graph = MagicMock()
        mem._graph.get_entities.return_value = [
            {"name": "Python", "type": "language", "key": "python", "attributes": {}},
            {"name": "Alice", "type": "person", "key": "alice", "attributes": {"role": "dev"}},
        ]
        mem._graph.get_related.return_value = ["Alice", "FastAPI"]
    else:
        mem._graph = None

    # Memories for search
    m1 = _make_memory(id="mem1", content="User prefers Python")
    m2 = _make_memory(id="mem2", content="Alice works on backend")
    mem.search.return_value = [m1, m2]

    # Single memory get
    mem.get.return_value = m1

    # Recall
    mem.recall.return_value = [
        RetrievalResult(memory=m1, score=0.95, match_source="tag"),
        RetrievalResult(memory=m2, score=0.80, match_source="vector"),
    ]

    # Timeline
    mem.timeline.return_value = [m1, m2]

    return mem


# ---------------------------------------------------------------------------
# Tool handler tests (existing functionality, MCP-05 regression)
# ---------------------------------------------------------------------------

class TestToolHandler:
    def test_memory_add(self):
        mem = _mock_agent_memory()
        mem.add.return_value = _make_memory()
        result = _handle_tool(mem, "memory_add", {"content": "test fact"})
        assert result["status"] == "stored"
        assert result["id"] == "mem1"

    def test_memory_get(self):
        mem = _mock_agent_memory()
        result = _handle_tool(mem, "memory_get", {"memory_id": "mem1"})
        assert result["id"] == "mem1"
        assert result["content"] == "User prefers Python"

    def test_memory_get_not_found(self):
        mem = _mock_agent_memory()
        mem.get.return_value = None
        result = _handle_tool(mem, "memory_get", {"memory_id": "nope"})
        assert "error" in result

    def test_memory_recall(self):
        mem = _mock_agent_memory()
        result = _handle_tool(mem, "memory_recall", {"query": "python"})
        assert result["count"] == 2
        assert result["memories"][0]["score"] == 0.95

    def test_memory_search(self):
        mem = _mock_agent_memory()
        result = _handle_tool(mem, "memory_search", {"query": "python"})
        assert result["count"] == 2

    def test_memory_forget(self):
        mem = _mock_agent_memory()
        mem.forget.return_value = True
        result = _handle_tool(mem, "memory_forget", {"memory_id": "mem1"})
        assert result["success"] is True

    def test_memory_timeline(self):
        mem = _mock_agent_memory()
        result = _handle_tool(mem, "memory_timeline", {"entity": "user"})
        assert result["count"] == 2
        assert result["entity"] == "user"

    def test_memory_stats(self):
        mem = _mock_agent_memory()
        mem.stats.return_value = {"total": 10}
        result = _handle_tool(mem, "memory_stats", {})
        assert result["total"] == 10

    def test_memory_health(self):
        mem = _mock_agent_memory()
        mem.health.return_value = {"healthy": True}
        result = _handle_tool(mem, "memory_health", {})
        assert result["healthy"] is True

    def test_unknown_tool(self):
        mem = _mock_agent_memory()
        result = _handle_tool(mem, "no_such_tool", {})
        assert "error" in result


# ---------------------------------------------------------------------------
# Resource tests (MCP-01, MCP-02)
# ---------------------------------------------------------------------------

class TestResources:
    def test_list_resources_with_graph(self):
        mem = _mock_agent_memory(graph_available=True)
        resources = asyncio.run(_list_resources(mem))
        # 2 entities + 2 recent memories = 4 resources
        assert len(resources) == 4
        entity_uris = [r.uri for r in resources if "entity" in str(r.uri)]
        memory_uris = [r.uri for r in resources if "memory" in str(r.uri)]
        assert len(entity_uris) == 2
        assert len(memory_uris) == 2

    def test_list_resources_without_graph(self):
        mem = _mock_agent_memory(graph_available=False)
        resources = asyncio.run(_list_resources(mem))
        # Only memory resources when graph is None
        assert len(resources) == 2
        for r in resources:
            assert "memory" in str(r.uri)

    def test_read_resource_entity(self):
        mem = _mock_agent_memory(graph_available=True)
        result = asyncio.run(_read_resource(mem, "attestor://entity/python"))
        data = json.loads(result)
        assert data["name"] == "Python"
        assert "related" in data

    def test_read_resource_entity_no_graph(self):
        mem = _mock_agent_memory(graph_available=False)
        with pytest.raises(ValueError, match="Graph not available"):
            asyncio.run(_read_resource(mem, "attestor://entity/python"))

    def test_read_resource_memory(self):
        mem = _mock_agent_memory(graph_available=True)
        result = asyncio.run(_read_resource(mem, "attestor://memory/mem1"))
        data = json.loads(result)
        assert data["id"] == "mem1"
        assert data["content"] == "User prefers Python"

    def test_read_resource_unknown_uri(self):
        mem = _mock_agent_memory(graph_available=True)
        with pytest.raises(ValueError, match="Unknown resource"):
            asyncio.run(_read_resource(mem, "attestor://unknown/thing"))

    def test_list_resource_templates(self):
        mem = _mock_agent_memory()
        templates = asyncio.run(_list_resource_templates(mem))
        assert len(templates) == 2
        template_uris = {t.uriTemplate for t in templates}
        assert "attestor://entity/{name}" in template_uris
        assert "attestor://memory/{id}" in template_uris

    def test_list_resources_emits_new_scheme_only(self):
        mem = _mock_agent_memory(graph_available=True)
        resources = asyncio.run(_list_resources(mem))
        for r in resources:
            assert str(r.uri).startswith("attestor://"), (
                f"emitted URIs must use attestor:// scheme, got {r.uri}"
            )


# ---------------------------------------------------------------------------
# Prompt tests (MCP-03, MCP-04)
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_list_prompts(self):
        mem = _mock_agent_memory()
        prompts = asyncio.run(_list_prompts(mem))
        assert len(prompts) == 2
        names = {p.name for p in prompts}
        assert names == {"recall", "timeline"}

    def test_get_prompt_recall(self):
        mem = _mock_agent_memory()
        result = asyncio.run(_get_prompt(mem, "recall", {"query": "python preferences"}))
        assert result.messages
        assert len(result.messages) >= 1
        mem.recall.assert_called_once_with("python preferences")

    def test_get_prompt_timeline(self):
        mem = _mock_agent_memory()
        result = asyncio.run(_get_prompt(mem, "timeline", {"entity": "user"}))
        assert result.messages
        assert len(result.messages) >= 1
        mem.timeline.assert_called_once_with("user")

    def test_get_prompt_unknown(self):
        mem = _mock_agent_memory()
        with pytest.raises(ValueError, match="Unknown prompt"):
            asyncio.run(_get_prompt(mem, "nonexistent", {}))


# ---------------------------------------------------------------------------
# Helper: extract handler functions from create_server
# ---------------------------------------------------------------------------

def _extract_handlers(mem):
    """Create server with mock mem and extract handler functions.

    We patch AgentMemory to inject our mock, then extract the registered handlers.
    """
    from attestor.mcp.server import _build_handlers
    return _build_handlers(mem)


async def _list_resources(mem):
    handlers = _extract_handlers(mem)
    return await handlers["list_resources"]()


async def _read_resource(mem, uri):
    from pydantic import AnyUrl
    handlers = _extract_handlers(mem)
    return await handlers["read_resource"](AnyUrl(uri))


async def _list_resource_templates(mem):
    handlers = _extract_handlers(mem)
    return await handlers["list_resource_templates"]()


async def _list_prompts(mem):
    handlers = _extract_handlers(mem)
    return await handlers["list_prompts"]()


async def _get_prompt(mem, name, arguments):
    handlers = _extract_handlers(mem)
    return await handlers["get_prompt"](name, arguments)
