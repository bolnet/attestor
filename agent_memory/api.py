"""Starlette ASGI app — HTTP API for Memwright over ArangoDB."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger("agent_memory.api")

# Lazy singleton — initialized on first request
_mem = None


def _get_mem():
    global _mem
    if _mem is None:
        from agent_memory.core import AgentMemory

        data_dir = os.environ.get("MEMWRIGHT_DATA_DIR", "/tmp/memwright")
        config: Dict[str, Any] = {"backends": ["arangodb"]}

        # ArangoDB connection from env
        arango_url = os.environ.get("ARANGO_URL", "http://localhost:8529")
        arango_password = os.environ.get("ARANGO_PASSWORD", "")
        arango_database = os.environ.get("ARANGO_DATABASE", "memwright")

        config["arangodb"] = {
            "url": arango_url,
            "database": arango_database,
            "auth": {"username": "root", "password": arango_password},
            "tls": {"verify": os.environ.get("ARANGO_TLS_VERIFY", "false").lower() == "true"},
        }

        _mem = AgentMemory(data_dir, config=config)
    return _mem


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


async def health(request: Request) -> JSONResponse:
    mem = _get_mem()
    return _ok(mem.health())


async def stats(request: Request) -> JSONResponse:
    mem = _get_mem()
    return _ok(mem.stats())


async def add_memory(request: Request) -> JSONResponse:
    body = await request.json()
    content = body.get("content")
    if not content:
        return _err("content is required")
    mem = _get_mem()
    m = mem.add(
        content=content,
        tags=body.get("tags", []),
        category=body.get("category", "general"),
        entity=body.get("entity"),
        event_date=body.get("event_date"),
        confidence=body.get("confidence", 1.0),
        metadata=body.get("metadata", {}),
    )
    return _ok(m.to_dict())


async def recall(request: Request) -> JSONResponse:
    body = await request.json()
    query = body.get("query")
    if not query:
        return _err("query is required")
    mem = _get_mem()
    results = mem.recall(query, budget=body.get("budget"))
    return _ok([{"content": r.memory.content, "score": r.score,
                 "source": r.match_source, "id": r.memory.id,
                 "memory": r.memory.to_dict()} for r in results])


async def search(request: Request) -> JSONResponse:
    body = await request.json()
    mem = _get_mem()
    memories = mem.search(
        query=body.get("query"),
        category=body.get("category"),
        entity=body.get("entity"),
        status=body.get("status", "active"),
        after=body.get("after"),
        before=body.get("before"),
        limit=body.get("limit", 10),
    )
    return _ok([m.to_dict() for m in memories])


async def timeline(request: Request) -> JSONResponse:
    body = await request.json()
    entity = body.get("entity")
    if not entity:
        return _err("entity is required")
    mem = _get_mem()
    memories = mem.timeline(entity)
    return _ok([m.to_dict() for m in memories])


async def forget(request: Request) -> JSONResponse:
    body = await request.json()
    memory_id = body.get("memory_id")
    if not memory_id:
        return _err("memory_id is required")
    mem = _get_mem()
    ok = mem.forget(memory_id)
    return _ok({"forgotten": ok})


async def get_memory(request: Request) -> JSONResponse:
    memory_id = request.path_params["memory_id"]
    mem = _get_mem()
    m = mem.get(memory_id)
    if m is None:
        return _err("not found", status=404)
    return _ok(m.to_dict())


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/stats", stats, methods=["GET"]),
    Route("/add", add_memory, methods=["POST"]),
    Route("/recall", recall, methods=["POST"]),
    Route("/search", search, methods=["POST"]),
    Route("/timeline", timeline, methods=["POST"]),
    Route("/forget", forget, methods=["POST"]),
    Route("/memory/{memory_id}", get_memory, methods=["GET"]),
]

app = Starlette(routes=routes)
