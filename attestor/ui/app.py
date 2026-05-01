"""Read-only web UI for Attestor — Starlette sub-app wiring.

Renders Jinja2 templates served with a "Forensic Archive" aesthetic.
All routes are GET-only (one POST for recall debug). The UI talks to
``AgentMemory`` directly; it never mutates the store.

This module is the thin wiring layer:
- Page handlers and JSON-data handlers live in ``attestor.ui.routes``.
- Export endpoints live in ``attestor.ui.export``.
- Cross-module helpers (``get_mem``, ``is_htmx``, ``common_context``)
  live in ``attestor.ui._state``.
- Jinja2 filter registration lives in ``attestor.ui.filters``.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from attestor.ui.export import (
    build_export_routes,
    memories_export_csv,
    memories_export_json,
)
from attestor.ui.filters import register_filters
from attestor.ui.routes import build_routes

_UI_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_UI_DIR / "templates"))
register_filters(_TEMPLATES.env)


def ui_routes() -> list[Route | Mount]:
    """Return absolute UI routes — can be appended to any Starlette app.

    Route ordering preserves the original behaviour: export endpoints are
    registered before the dynamic ``/ui/memories/{memory_id}`` route so they
    are matched first.
    """
    page_routes = build_routes(_TEMPLATES)
    export_routes = build_export_routes()

    # Find insertion point: just after "/ui/memories" and before
    # "/ui/memories/{memory_id}". Keeps URL precedence identical to the
    # pre-split implementation.
    ordered: list[Route | Mount] = []
    for route in page_routes:
        ordered.append(route)
        if isinstance(route, Route) and route.path == "/ui/memories":
            ordered.extend(export_routes)

    ordered.append(
        Mount(
            "/ui/static",
            StaticFiles(directory=str(_UI_DIR / "static")),
            name="static",
        ),
    )
    return ordered


def create_ui_app() -> Starlette:
    """Standalone Starlette app for the UI. Use when running `attestor ui`."""
    return Starlette(routes=ui_routes())


# Re-export the export handlers so any external import path is preserved.
__all__ = [
    "app",
    "create_ui_app",
    "memories_export_csv",
    "memories_export_json",
    "ui_routes",
]


app = create_ui_app()
