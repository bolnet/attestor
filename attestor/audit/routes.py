# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Starlette route handlers for the /audit/* surface.

Kept separate from ``attestor.audit.explorer`` so the explorer stays
import-light (no Starlette dependency for callers that only want the
in-process query API).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from attestor.audit.explorer import (
    TraceExplorer,
    default_log_path,
)

logger = logging.getLogger("attestor.audit")


# ──────────────────────────────────────────────────────────────────────
# Toggle — endpoints OFF by default unless ATTESTOR_AUDIT_ENABLED=1
# (or audit.dashboard.enabled=true in YAML).
# ──────────────────────────────────────────────────────────────────────


def is_enabled() -> bool:
    if os.environ.get("ATTESTOR_AUDIT_ENABLED", "").lower() in ("1", "true", "yes"):
        return True
    # Best-effort: read configs/attestor.yaml without forcing a hard import.
    cfg = _load_audit_block()
    return bool(cfg.get("dashboard", {}).get("enabled", False))


def _load_audit_block() -> dict[str, Any]:
    """Load just the ``audit:`` block from configs/attestor.yaml.

    Done with a tolerant try/except so a missing PyYAML or missing config
    file degrades to ``{}`` rather than crashing the API at import time.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    cfg_path = os.environ.get(
        "ATTESTOR_CONFIG",
        str(Path(__file__).resolve().parents[2] / "configs" / "attestor.yaml"),
    )
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}
    block = data.get("audit") or {}
    if not isinstance(block, dict):
        return {}
    return block


# ──────────────────────────────────────────────────────────────────────
# Lazy explorer — re-resolves on each request so the path can change at
# runtime (e.g. tests monkeypatching ATTESTOR_TRACE_FILE).
# ──────────────────────────────────────────────────────────────────────


def _get_explorer() -> TraceExplorer:
    """Build/return a TraceExplorer pinned to the configured log path.

    Resolution order:
        1. ``ATTESTOR_TRACE_FILE`` env (matches the writer in attestor.trace).
        2. ``audit.trace_log_path`` in configs/attestor.yaml.
        3. ``~/.attestor/trace.jsonl`` default.
    """
    env = os.environ.get("ATTESTOR_TRACE_FILE")
    if env:
        return TraceExplorer(Path(env).expanduser())

    block = _load_audit_block()
    configured = block.get("trace_log_path")
    if configured:
        return TraceExplorer(Path(configured).expanduser())

    return TraceExplorer(default_log_path())


# ──────────────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────────────


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


async def list_recalls_handler(request: Request) -> JSONResponse:
    qp = request.query_params
    try:
        limit = int(qp.get("limit", "100"))
    except ValueError:
        return _err("limit must be an integer")
    limit = max(1, min(limit, 1000))
    explorer = _get_explorer()
    summaries = explorer.list_recalls(
        user_id=qp.get("user_id") or None,
        namespace=qp.get("namespace") or None,
        since=qp.get("since") or None,
        limit=limit,
    )
    return _ok([s.to_dict() for s in summaries])


async def recall_detail_handler(request: Request) -> JSONResponse:
    rid = request.path_params["recall_id"]
    explorer = _get_explorer()
    detail = explorer.get_recall(rid)
    if detail is None:
        return _err("recall not found", status=404)
    return _ok(detail.to_dict())


async def memory_access_handler(request: Request) -> JSONResponse:
    mid = request.path_params["memory_id"]
    explorer = _get_explorer()
    return _ok(explorer.memory_access_timeline(mid))


async def user_activity_handler(request: Request) -> JSONResponse:
    uid = request.path_params["user_id"]
    since = request.query_params.get("since") or None
    explorer = _get_explorer()
    return _ok(explorer.user_activity(uid, since=since))


# ──────────────────────────────────────────────────────────────────────
# Static dashboard handler — Starlette's StaticFiles would mount a
# directory, but we want a single canonical path /audit.html that maps
# to the bundled file. A FileResponse keeps the routing explicit.
# ──────────────────────────────────────────────────────────────────────


_DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "ui" / "static" / "audit.html"


async def dashboard_html_handler(request: Request) -> FileResponse | JSONResponse:
    if not _DASHBOARD_PATH.exists():
        return _err("dashboard asset missing", status=500)
    return FileResponse(
        _DASHBOARD_PATH,
        media_type="text/html; charset=utf-8",
        headers={"cache-control": "no-store"},
    )


def audit_routes() -> list[Route]:
    """Return the audit + dashboard routes for mounting on the API app."""
    return [
        Route("/audit/recalls", list_recalls_handler, methods=["GET"]),
        Route("/audit/recall/{recall_id}", recall_detail_handler, methods=["GET"]),
        Route("/audit/memory/{memory_id}/access",
              memory_access_handler, methods=["GET"]),
        Route("/audit/user/{user_id}/activity",
              user_activity_handler, methods=["GET"]),
        Route("/audit.html", dashboard_html_handler, methods=["GET"]),
    ]


__all__ = ["audit_routes", "is_enabled"]
