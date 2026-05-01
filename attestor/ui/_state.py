"""Shared UI helpers — memory access, HTMX detection, common template context."""

from __future__ import annotations

import os
from typing import Any

from starlette.requests import Request


def get_mem(request: Request) -> Any:
    """Lazy-load the AgentMemory singleton keyed on the app state."""
    app = request.app
    mem = getattr(app.state, "memory", None)
    if mem is not None:
        return mem

    from attestor._paths import resolve_store_path
    from attestor.core import AgentMemory

    data_dir = resolve_store_path()
    app.state.memory = AgentMemory(data_dir)
    return app.state.memory


def is_htmx(request: Request) -> bool:
    """Return True if the request was issued by HTMX."""
    return request.headers.get("HX-Request") == "true"


def common_context(request: Request, mem: Any) -> dict[str, Any]:
    """Build the shared template context (stats, namespace default, store path)."""
    stats: dict[str, Any] = {}
    try:
        stats = mem.stats()
    except Exception:
        stats = {}
    from attestor import _branding as brand
    from attestor._paths import resolve_store_path

    return {
        "stats": stats,
        "namespace_default": os.environ.get(brand.ENV_NAMESPACE, "default"),
        "store_path": resolve_store_path(),
    }
