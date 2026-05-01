"""Export endpoints — JSON and CSV downloads of the memory list."""

from __future__ import annotations

import csv
import io

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from attestor.ui._state import get_mem
from attestor.ui.filters import (
    memory_to_dict,
    parse_filters,
    search_with_filters,
)

_EXPORT_LIMIT = 5000

_CSV_COLUMNS: list[str] = [
    "id", "content", "category", "entity", "namespace",
    "created_at", "status", "confidence", "tags",
    "event_date", "valid_from", "valid_until",
    "superseded_by", "access_count", "content_hash",
]


async def memories_export_json(request: Request) -> JSONResponse:
    """Export matching memories as a JSON array (up to 5000)."""
    mem = get_mem(request)
    filters = parse_filters(request)
    results = search_with_filters(mem, filters, limit=_EXPORT_LIMIT)
    data = [memory_to_dict(m) for m in results]
    return JSONResponse(
        data,
        headers={
            "Content-Disposition": 'attachment; filename="attestor-export.json"',
        },
    )


async def memories_export_csv(request: Request) -> Response:
    """Export matching memories as CSV (up to 5000)."""
    mem = get_mem(request)
    filters = parse_filters(request)
    results = search_with_filters(mem, filters, limit=_EXPORT_LIMIT)
    data = [memory_to_dict(m) for m in results]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        # Flatten tags list to semicolon-separated string
        row_copy = {**row, "tags": ";".join(row.get("tags") or [])}
        writer.writerow(row_copy)

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="attestor-export.csv"',
        },
    )


def build_export_routes() -> list[Route]:
    """Return the export endpoint routes."""
    return [
        Route("/ui/memories/export.json", memories_export_json),
        Route("/ui/memories/export.csv", memories_export_csv),
    ]
