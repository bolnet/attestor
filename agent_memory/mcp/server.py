"""MCP server exposing AgentMemory as tools, resources, and prompts."""

from __future__ import annotations

import json
import sys
from typing import Any

from agent_memory.core import AgentMemory


def _build_handlers(mem: AgentMemory) -> dict:
    """Build resource and prompt handler functions for an AgentMemory instance.

    Returns a dict of async handler functions that can be registered with
    an MCP server or called directly in tests.
    """
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        Resource,
        ResourceTemplate,
        TextContent,
    )

    async def list_resources() -> list[Resource]:
        resources: list[Resource] = []

        # Entity resources (only if graph is available)
        if mem._graph is not None:
            for entity in mem._graph.get_entities():
                resources.append(
                    Resource(
                        uri=f"memwright://entity/{entity['key']}",
                        name=entity["name"],
                        description=f"{entity['type']} entity",
                        mimeType="application/json",
                    )
                )

        # Recent memory resources (last 50 active)
        for m in mem.search(limit=50):
            resources.append(
                Resource(
                    uri=f"memwright://memory/{m.id}",
                    name=m.content[:80],
                    description=f"{m.category} memory",
                    mimeType="application/json",
                )
            )

        return resources

    async def read_resource(uri) -> str:
        uri_str = str(uri)

        if uri_str.startswith("memwright://entity/"):
            if mem._graph is None:
                raise ValueError("Graph not available")
            key = uri_str[len("memwright://entity/"):]
            # Find matching entity
            entities = mem._graph.get_entities()
            match = next((e for e in entities if e["key"] == key), None)
            if match is None:
                raise ValueError(f"Entity not found: {key}")
            related = mem._graph.get_related(key)
            return json.dumps({
                "name": match["name"],
                "type": match["type"],
                "key": match["key"],
                "attributes": match.get("attributes", {}),
                "related": related,
            }, indent=2)

        elif uri_str.startswith("memwright://memory/"):
            memory_id = uri_str[len("memwright://memory/"):]
            memory = mem.get(memory_id)
            if memory is None:
                raise ValueError(f"Memory not found: {memory_id}")
            return json.dumps({
                "id": memory.id,
                "content": memory.content,
                "category": memory.category,
                "entity": memory.entity,
                "tags": memory.tags,
                "status": memory.status,
                "event_date": memory.event_date,
                "created_at": memory.created_at,
            }, indent=2)

        else:
            raise ValueError(f"Unknown resource URI: {uri_str}")

    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="memwright://entity/{name}",
                name="Entity",
                description="Look up an entity by name",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="memwright://memory/{id}",
                name="Memory",
                description="Look up a memory by ID",
                mimeType="application/json",
            ),
        ]

    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="recall",
                description="Search memories for relevant context",
                arguments=[
                    PromptArgument(
                        name="query",
                        description="Natural language query to search memories",
                        required=True,
                    ),
                ],
            ),
            Prompt(
                name="timeline",
                description="Get chronological history of an entity",
                arguments=[
                    PromptArgument(
                        name="entity",
                        description="Entity name to get timeline for",
                        required=True,
                    ),
                ],
            ),
        ]

    async def get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        args = arguments or {}

        if name == "recall":
            query = args.get("query", "")
            results = mem.recall(query)
            lines = []
            for r in results:
                lines.append(
                    f"[{r.match_source}] (score: {r.score:.2f}) {r.memory.content}"
                )
            text = "\n".join(lines) if lines else "No memories found."
            return GetPromptResult(
                description=f"Recall results for: {query}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=text),
                    ),
                ],
            )

        elif name == "timeline":
            entity = args.get("entity", "")
            memories = mem.timeline(entity)
            lines = []
            for m in memories:
                date = m.event_date or m.created_at
                lines.append(f"[{date}] {m.content}")
            text = "\n".join(lines) if lines else f"No timeline for {entity}."
            return GetPromptResult(
                description=f"Timeline for: {entity}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=text),
                    ),
                ],
            )

        else:
            raise ValueError(f"Unknown prompt: {name}")

    return {
        "list_resources": list_resources,
        "read_resource": read_resource,
        "list_resource_templates": list_resource_templates,
        "list_prompts": list_prompts,
        "get_prompt": get_prompt,
    }


def create_server(memory_path: str):
    """Create an MCP server exposing AgentMemory tools, resources, and prompts.

    MCP is a core dependency — included with poetry add memwright.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        print(
            "MCP package not found. Run: poetry install",
            file=sys.stderr,
        )
        sys.exit(1)

    mem = AgentMemory(memory_path)
    server = Server("agent-memory")

    # -- Tools --

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
                        "namespace": {
                            "type": "string",
                            "description": "Namespace for multi-agent isolation (e.g., 'user:john', 'project:acme')",
                            "default": "default",
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
                        "namespace": {
                            "type": "string",
                            "description": "Filter by namespace (e.g., 'user:john')",
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
                        "namespace": {"type": "string", "description": "Filter by namespace"},
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
                        "namespace": {"type": "string", "description": "Filter by namespace"},
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
                    "Reports status of SQLite, ChromaDB vector store, "
                    "NetworkX graph, and retrieval pipeline. "
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

    # -- Resources and Prompts --

    handlers = _build_handlers(mem)

    @server.list_resources()
    async def _list_resources():
        return await handlers["list_resources"]()

    @server.read_resource()
    async def _read_resource(uri):
        return await handlers["read_resource"](uri)

    @server.list_resource_templates()
    async def _list_resource_templates():
        return await handlers["list_resource_templates"]()

    @server.list_prompts()
    async def _list_prompts():
        return await handlers["list_prompts"]()

    @server.get_prompt()
    async def _get_prompt(name, arguments):
        return await handlers["get_prompt"](name, arguments)

    return server


def _handle_tool(mem: AgentMemory, name: str, args: dict) -> dict:
    if name == "memory_add":
        memory = mem.add(
            content=args["content"],
            tags=args.get("tags", []),
            category=args.get("category", "general"),
            entity=args.get("entity"),
            namespace=args.get("namespace", "default"),
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
        results = mem.recall(
            args["query"],
            budget=args.get("budget", 2000),
            namespace=args.get("namespace"),
        )
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
            namespace=args.get("namespace"),
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
        memories = mem.timeline(args["entity"], namespace=args.get("namespace"))
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
