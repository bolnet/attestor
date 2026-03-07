"""MCP server exposing AgentMemory as tools for Claude Code / Claude Desktop."""

from __future__ import annotations

import json
import sys
from typing import Any

from agent_memory.core import AgentMemory


def create_server(memory_path: str):
    """Create an MCP server exposing AgentMemory tools.

    Requires: pip install agent-memory[mcp]
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        print(
            "MCP package required. Install with: pip install agent-memory[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    mem = AgentMemory(memory_path)
    server = Server("agent-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory_add",
                description=(
                    "Store a new memory/fact about the user or project. "
                    "Use this to remember preferences, decisions, context, "
                    "and important facts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact to remember. Should be a single, atomic statement.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization (e.g., ['preference', 'python'])",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["career", "project", "preference", "personal", "technical", "general"],
                            "description": "Category of the memory",
                            "default": "general",
                        },
                        "entity": {
                            "type": "string",
                            "description": "Primary entity this fact is about (company, tool, person, etc.)",
                        },
                        "event_date": {
                            "type": "string",
                            "description": "When this fact occurred (ISO date, e.g. '2025-03-15'). Used for timeline ordering.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level 0.0-1.0 (default: 1.0). Lower confidence for uncertain facts.",
                            "default": 1.0,
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="memory_get",
                description="Retrieve a single memory by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to retrieve",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_recall",
                description=(
                    "Smart retrieval: finds the most relevant memories using "
                    "multi-layer search (tags, graph connections, "
                    "and semantic vectors) with score fusion. Use this as the "
                    "primary way to find relevant context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search memories",
                        },
                        "budget": {
                            "type": "integer",
                            "description": "Max tokens to return (default: 2000)",
                            "default": 2000,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="memory_search",
                description=(
                    "Simple keyword search with filters. Use this when you need "
                    "to filter by category, entity, status, or date range. "
                    "For general relevance queries, prefer memory_recall instead."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text search query (semantic vector search)"},
                        "category": {"type": "string", "description": "Filter by category"},
                        "entity": {"type": "string", "description": "Filter by entity"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "superseded", "archived"],
                            "default": "active",
                        },
                        "after": {"type": "string", "description": "Only memories after this date (ISO format)"},
                        "before": {"type": "string", "description": "Only memories before this date (ISO format)"},
                        "limit": {"type": "integer", "default": 10},
                    },
                },
            ),
            Tool(
                name="memory_forget",
                description="Archive a specific memory by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "ID of the memory to archive"},
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_timeline",
                description="Get the chronological history of memories about an entity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity name to get timeline for"},
                    },
                    "required": ["entity"],
                },
            ),
            Tool(
                name="memory_stats",
                description="Get statistics about the memory store.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_health",
                description=(
                    "Check health of all memory system components. "
                    "Reports status of SQLite, vector store, graph database, "
                    "embeddings API, and retrieval pipeline. "
                    "Call this first to verify the system is working."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = _handle_tool(mem, name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _handle_tool(mem: AgentMemory, name: str, args: dict) -> dict:
    if name == "memory_add":
        memory = mem.add(
            content=args["content"],
            tags=args.get("tags", []),
            category=args.get("category", "general"),
            entity=args.get("entity"),
            event_date=args.get("event_date"),
            confidence=args.get("confidence", 1.0),
        )
        return {"id": memory.id, "content": memory.content, "status": "stored"}

    elif name == "memory_get":
        memory = mem.get(args["memory_id"])
        if not memory:
            return {"error": f"Memory {args['memory_id']} not found"}
        return {
            "id": memory.id,
            "content": memory.content,
            "category": memory.category,
            "entity": memory.entity,
            "tags": memory.tags,
            "status": memory.status,
            "event_date": memory.event_date,
            "created_at": memory.created_at,
        }

    elif name == "memory_recall":
        results = mem.recall(args["query"], budget=args.get("budget", 2000))
        return {
            "count": len(results),
            "memories": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "score": round(r.score, 3),
                    "source": r.match_source,
                    "category": r.memory.category,
                    "entity": r.memory.entity,
                    "tags": r.memory.tags,
                }
                for r in results
            ],
        }

    elif name == "memory_search":
        memories = mem.search(
            query=args.get("query"),
            category=args.get("category"),
            entity=args.get("entity"),
            status=args.get("status", "active"),
            after=args.get("after"),
            before=args.get("before"),
            limit=args.get("limit", 10),
        )
        return {
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "category": m.category,
                    "entity": m.entity,
                    "tags": m.tags,
                    "status": m.status,
                }
                for m in memories
            ],
        }

    elif name == "memory_forget":
        success = mem.forget(args["memory_id"])
        return {"success": success, "memory_id": args["memory_id"]}

    elif name == "memory_timeline":
        memories = mem.timeline(args["entity"])
        return {
            "entity": args["entity"],
            "count": len(memories),
            "timeline": [
                {
                    "id": m.id,
                    "content": m.content,
                    "date": m.event_date or m.created_at,
                    "status": m.status,
                }
                for m in memories
            ],
        }

    elif name == "memory_stats":
        return mem.stats()

    elif name == "memory_health":
        return mem.health()

    else:
        return {"error": f"Unknown tool: {name}"}


async def run_server(memory_path: str):
    """Run the MCP server over stdio."""
    from mcp.server.stdio import stdio_server

    server = create_server(memory_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
