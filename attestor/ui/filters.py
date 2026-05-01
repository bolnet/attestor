"""Jinja2 filter registration plus query-param filter helpers used by routes.

Two related concepts share this module:

1. ``register_filters`` — placeholder for Jinja2 custom filters (``env.filters``).
   None are registered today; keep the hook so adding one is a one-line change.

2. ``parse_filters`` / ``search_with_filters`` / ``filters_display`` /
   ``filter_query_string`` / ``memory_to_dict`` — helpers that shape the
   query-parameter filter set (``q``, ``namespace``, ``category``, ``entity``,
   ``status``) used by the memories list and export endpoints.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment
from starlette.requests import Request


def register_filters(env: Environment) -> None:
    """Register Jinja2 custom filters on the given environment.

    No filters are registered today — this exists so adding one is a single
    line in this function rather than a refactor of ``app.py``.
    """
    return None


def memory_to_dict(m: Any) -> dict[str, Any]:
    """Flatten a Memory dataclass for template use."""
    created = getattr(m, "created_at", "") or ""
    return {
        "id": m.id,
        "short_id": m.id[:8],
        "content": m.content,
        "excerpt": (m.content[:220] + "…") if len(m.content) > 220 else m.content,
        "tags": list(m.tags or []),
        "category": getattr(m, "category", "general"),
        "entity": getattr(m, "entity", None),
        "namespace": getattr(m, "namespace", "default"),
        "created_at": created,
        "created_date": created[:10] if created else "",
        "created_time": created[11:16] if len(created) >= 16 else "",
        "event_date": getattr(m, "event_date", None),
        "valid_from": getattr(m, "valid_from", None),
        "valid_until": getattr(m, "valid_until", None),
        "superseded_by": getattr(m, "superseded_by", None),
        "confidence": float(getattr(m, "confidence", 1.0) or 0.0),
        "confidence_pct": round(float(getattr(m, "confidence", 1.0) or 0.0) * 100),
        "status": getattr(m, "status", "active"),
        "access_count": int(getattr(m, "access_count", 0) or 0),
        "last_accessed": getattr(m, "last_accessed", None),
        "content_hash": getattr(m, "content_hash", None),
        "metadata": getattr(m, "metadata", {}) or {},
    }


def parse_filters(request: Request) -> dict[str, str | None]:
    """Extract filter params from request, returning a frozen dict."""
    return {
        "q": request.query_params.get("q") or None,
        "namespace": request.query_params.get("namespace") or None,
        "category": request.query_params.get("category") or None,
        "entity": request.query_params.get("entity") or None,
        "status": request.query_params.get("status") or "active",
    }


def search_with_filters(
    mem: Any, filters: dict[str, str | None], limit: int,
) -> list[Any]:
    """Run ``mem.search`` with the given filters dict."""
    try:
        return mem.search(
            query=filters["q"],
            category=filters["category"],
            entity=filters["entity"],
            namespace=filters["namespace"],
            status=filters["status"] or "active",
            limit=limit,
        )
    except Exception:
        return []


def filters_display(filters: dict[str, str | None]) -> dict[str, str]:
    """Return a template-friendly copy with empty strings instead of None."""
    return {k: (v or "") for k, v in filters.items()}


def filter_query_string(filters: dict[str, str | None]) -> str:
    """Build a URL query string fragment from active filters (no leading &)."""
    parts: list[str] = []
    for k, v in filters.items():
        if v:
            parts.append(f"{k}={v}")
    return "&".join(parts)
