"""Starlette ASGI app — HTTP API for Attestor over ArangoDB."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger("attestor.api")

# Lazy singleton — initialized on first request
_mem = None


def _build_config() -> Optional[Dict[str, Any]]:
    """Build backend config. Returns None when no backend can be resolved.

    Resolution order (top wins):
        1. Explicit env vars: POSTGRES_URL [+ NEO4J_URI] / NEO4J_URI alone /
           ARANGO_URL — preserved for cloud deploys that bind secrets via env.
        2. ``configs/attestor.yaml`` via ``attestor.config.get_stack()`` —
           when no env vars are set we honor the YAML SoT. Pass
           ATTESTOR_CONFIG=/path/to/yaml to override the default file.
        3. None — caller bubbles a clear error.
    """
    postgres_url = os.environ.get("POSTGRES_URL")
    neo4j_uri = os.environ.get("NEO4J_URI")

    cfg: Dict[str, Any] = {}
    backends: list = []

    if postgres_url:
        backends.append("postgres")
        cfg["postgres"] = {
            "mode": "cloud",
            "url": postgres_url,
            "database": os.environ.get("POSTGRES_DATABASE", "attestor"),
            "auth": {
                "username": os.environ.get("POSTGRES_USERNAME", "postgres"),
                "password": os.environ.get("POSTGRES_PASSWORD", ""),
            },
            "sslmode": os.environ.get("POSTGRES_SSLMODE"),
        }

    if neo4j_uri:
        backends.append("neo4j")
        cfg["neo4j"] = {
            "mode": "cloud",
            "url": neo4j_uri,
            "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
            "auth": {
                "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                "password": os.environ.get("NEO4J_PASSWORD", ""),
            },
        }

    if backends:
        cfg["backends"] = backends
        return cfg

    arango_url = os.environ.get("ARANGO_URL")
    if arango_url:
        return {
            "backends": ["arangodb"],
            "arangodb": {
                "mode": "cloud",
                "url": arango_url,
                "database": os.environ.get("ARANGO_DATABASE", "attestor"),
                "auth": {
                    "username": os.environ.get("ARANGO_USERNAME", "root"),
                    "password": os.environ.get("ARANGO_PASSWORD", ""),
                },
                "tls": {
                    "verify": os.environ.get("ARANGO_TLS_VERIFY", "false").lower() == "true",
                },
            },
        }

    # No env override — defer to the YAML SoT.
    try:
        from attestor.config import build_backend_config, get_stack
        return build_backend_config(get_stack())
    except SystemExit:
        # YAML missing required env (e.g. NEO4J_PASSWORD) — return None so
        # the api surfaces a clear "configure backends" error rather than
        # the loader's exit message.
        return None


def _get_mem():
    global _mem
    if _mem is None:
        from attestor.core import AgentMemory
        from attestor._paths import resolve_data_dir

        data_dir = resolve_data_dir()
        config = _build_config()
        if config is not None:
            _mem = AgentMemory(data_dir, config=config)
        else:
            _mem = AgentMemory(data_dir)
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
        namespace=body.get("namespace", "default"),
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
    results = mem.recall(
        query, budget=body.get("budget"), namespace=body.get("namespace")
    )
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
        namespace=body.get("namespace"),
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
    memories = mem.timeline(entity, namespace=body.get("namespace"))
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

# Attach the read-only UI at /ui/* (routes have absolute paths)
try:
    from attestor.ui.app import ui_routes

    routes.extend(ui_routes())
except Exception:  # pragma: no cover - UI is optional
    pass


def _build_middleware() -> list:
    """Return the Starlette middleware stack for the resolved mode.

    SOLO mode (default) returns no middleware — the api is a single-user
    local service and bearer tokens would just be noise.

    HOSTED / SHARED modes attach :class:`JWTAuthMiddleware`. In those
    modes ``ATTESTOR_JWT_PUBLIC_KEY`` (PEM or JWKS JSON) MUST be set;
    if it isn't, every request returns 500 ``"auth not configured"`` so
    we fail visibly rather than serving anonymous traffic in a
    multi-tenant deployment.

    Configurable via env (only used when mode != SOLO):

      ATTESTOR_JWT_PUBLIC_KEY     PEM string or path to .pem; required.
      ATTESTOR_JWT_AUDIENCE       Optional ``aud`` claim to enforce.
      ATTESTOR_JWT_ISSUER         Optional ``iss`` claim to enforce.
      ATTESTOR_JWT_ALGORITHMS     Comma-separated; default ``RS256``.
      ATTESTOR_JWT_LEEWAY_SECONDS Default ``0``.
    """
    from starlette.middleware import Middleware

    from attestor.auth import JWTAuthMiddleware
    from attestor.mode import AttestorMode, detect_mode

    mode = detect_mode()
    if mode is AttestorMode.SOLO:
        return []

    public_key = os.environ.get("ATTESTOR_JWT_PUBLIC_KEY")
    if public_key and os.path.isfile(public_key):
        # Allow pointing at a .pem on disk for cloud deploys that
        # mount keys via a volume rather than env-encoding them.
        try:
            with open(public_key, "r", encoding="utf-8") as fh:
                public_key = fh.read()
        except OSError as e:
            logger.error("could not read ATTESTOR_JWT_PUBLIC_KEY=%r: %s",
                         public_key, e)
            public_key = None

    algorithms_raw = os.environ.get("ATTESTOR_JWT_ALGORITHMS", "RS256")
    algorithms = [a.strip() for a in algorithms_raw.split(",") if a.strip()]

    leeway = 0
    try:
        leeway = int(os.environ.get("ATTESTOR_JWT_LEEWAY_SECONDS", "0"))
    except ValueError:
        pass

    return [
        Middleware(
            JWTAuthMiddleware,
            mode=mode,
            public_key=public_key,
            audience=os.environ.get("ATTESTOR_JWT_AUDIENCE"),
            issuer=os.environ.get("ATTESTOR_JWT_ISSUER"),
            algorithms=algorithms,
            leeway_seconds=leeway,
        ),
    ]


app = Starlette(routes=routes, middleware=_build_middleware())
