"""Read-only web UI for Memwright — Starlette sub-app.

Renders Jinja2 templates served with a "Forensic Archive" aesthetic.
All routes are GET-only. The UI talks to AgentMemory directly; it never
mutates the store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

_UI_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_UI_DIR / "templates"))


def _get_mem(request: Request):
    """Lazy-load the AgentMemory singleton keyed on the app state."""
    app = request.app
    mem = getattr(app.state, "memory", None)
    if mem is not None:
        return mem

    from agent_memory.core import AgentMemory

    data_dir = os.environ.get(
        "MEMWRIGHT_PATH",
        os.path.expanduser("~/.memwright"),
    )
    app.state.memory = AgentMemory(data_dir)
    return app.state.memory


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _memory_to_dict(m) -> Dict[str, Any]:
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


def _common_context(request: Request, mem) -> Dict[str, Any]:
    stats = {}
    try:
        stats = mem.stats()
    except Exception:
        stats = {}
    return {
        "stats": stats,
        "namespace_default": os.environ.get("MEMWRIGHT_NAMESPACE", "default"),
        "store_path": os.environ.get(
            "MEMWRIGHT_PATH", os.path.expanduser("~/.memwright")
        ),
    }


async def index(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/ui/memories", status_code=302)


async def memories_list(request: Request) -> HTMLResponse:
    mem = _get_mem(request)

    q = request.query_params.get("q") or None
    namespace = request.query_params.get("namespace") or None
    category = request.query_params.get("category") or None
    entity = request.query_params.get("entity") or None
    status = request.query_params.get("status") or "active"
    limit = int(request.query_params.get("limit") or 60)

    try:
        results = mem.search(
            query=q,
            category=category,
            entity=entity,
            namespace=namespace,
            status=status,
            limit=limit,
        )
    except Exception:
        results = []

    memories = [_memory_to_dict(m) for m in results]

    # Assign a subtle, deterministic rotation per card so the page feels pinned.
    for i, m in enumerate(memories):
        m["tilt"] = ((hash(m["id"]) % 7) - 3) * 0.22

    ctx = {
        "request": request,
        "memories": memories,
        "total": len(memories),
        "filters": {
            "q": q or "",
            "namespace": namespace or "",
            "category": category or "",
            "entity": entity or "",
            "status": status,
        },
        **_common_context(request, mem),
    }

    template = "memories/_grid.html" if _is_htmx(request) else "memories/list.html"
    return _TEMPLATES.TemplateResponse(request, template, ctx)


async def memory_detail(request: Request) -> HTMLResponse:
    mem = _get_mem(request)
    memory_id = request.path_params["memory_id"]

    m = mem.get(memory_id)
    if m is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    data = _memory_to_dict(m)
    tab = request.query_params.get("tab", "content")

    # Supersession chain
    chain = []
    cursor = m
    while cursor is not None and getattr(cursor, "superseded_by", None):
        nxt_id = cursor.superseded_by
        nxt = mem.get(nxt_id)
        if nxt is None or nxt.id in {c["id"] for c in chain}:
            break
        chain.append(_memory_to_dict(nxt))
        cursor = nxt

    # Predecessors (who did this supersede?) — query store directly
    predecessors: List[Dict[str, Any]] = []
    try:
        # naive: scan memories superseded_by this id
        for other in mem.search(namespace=m.namespace, limit=500, status="superseded"):
            if getattr(other, "superseded_by", None) == m.id:
                predecessors.append(_memory_to_dict(other))
    except Exception:
        pass

    # Nearest vector neighbors
    neighbors: List[Dict[str, Any]] = []
    try:
        vs = getattr(mem, "_vector_store", None)
        if vs is not None:
            near = vs.search(m.content[:512], limit=6, namespace=m.namespace)
            for r in near or []:
                if r.get("memory_id") == m.id:
                    continue
                other = mem.get(r["memory_id"])
                if other is None:
                    continue
                d = _memory_to_dict(other)
                d["score"] = round(float(r.get("score", 0.0)), 4)
                neighbors.append(d)
                if len(neighbors) >= 5:
                    break
    except Exception:
        pass

    # PageRank
    pagerank = 0.0
    try:
        pr = mem.pagerank() or {}
        if m.entity and m.entity in pr:
            pagerank = round(float(pr[m.entity]), 5)
    except Exception:
        pass

    # Graph neighborhood (1-hop edges for the selected entity)
    graph_edges: List[Dict[str, Any]] = []
    try:
        g = getattr(mem, "_graph", None)
        if g is not None and m.entity:
            nx_graph = getattr(g, "_graph", None) or getattr(g, "graph", None)
            if nx_graph is not None and m.entity in nx_graph:
                for u, v, d in nx_graph.edges(m.entity, data=True):
                    graph_edges.append(
                        {
                            "source": u,
                            "target": v,
                            "kind": d.get("kind") or d.get("type") or "related",
                        }
                    )
                for u, v, d in nx_graph.in_edges(m.entity, data=True):
                    graph_edges.append(
                        {
                            "source": u,
                            "target": v,
                            "kind": d.get("kind") or d.get("type") or "related",
                        }
                    )
    except Exception:
        pass

    ctx = {
        "request": request,
        "m": data,
        "tab": tab,
        "chain": chain,
        "predecessors": predecessors,
        "neighbors": neighbors,
        "pagerank": pagerank,
        "graph_edges": graph_edges[:30],
        **_common_context(request, mem),
    }

    return _TEMPLATES.TemplateResponse(request, "memories/detail.html", ctx)


async def graph_page(request: Request) -> HTMLResponse:
    """Full knowledge graph visualization via Cytoscape.js."""
    mem = _get_mem(request)

    graph_stats: Dict[str, Any] = {}
    try:
        g = getattr(mem, "_graph", None)
        if g is not None:
            graph_stats = g.graph_stats()
    except Exception:
        pass

    ctx = {
        "request": request,
        "graph_stats": graph_stats,
        **_common_context(request, mem),
    }
    return _TEMPLATES.TemplateResponse(request, "memories/graph.html", ctx)


async def graph_json(request: Request) -> JSONResponse:
    """Return full graph data for Cytoscape.js consumption."""
    mem = _get_mem(request)

    g = getattr(mem, "_graph", None)
    if g is None:
        return JSONResponse({"nodes": [], "edges": [], "stats": {}, "pagerank": {}})

    # Optional entity-type filter
    entity_type = request.query_params.get("type") or None

    entities = g.get_entities(entity_type=entity_type)
    pr = {}
    try:
        pr = mem.pagerank() or {}
    except Exception:
        pass

    stats = {}
    try:
        stats = g.graph_stats()
    except Exception:
        pass

    nodes = []
    node_keys = {e["key"] for e in entities}
    for e in entities:
        nodes.append({
            "id": e["key"],
            "label": e["name"],
            "type": e.get("type", "general"),
            "pagerank": round(pr.get(e["key"], 0.0), 6),
        })

    edges = []
    try:
        nx_graph = getattr(g, "_graph", None)
        if nx_graph is not None:
            for u, v, _key, data in nx_graph.edges(keys=True, data=True):
                if u in node_keys and v in node_keys:
                    edges.append({
                        "source": u,
                        "target": v,
                        "type": data.get("relation_type", "RELATED_TO"),
                    })
    except Exception:
        pass

    return JSONResponse({
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
        "pagerank": pr,
    })


async def recall_page(request: Request) -> HTMLResponse:
    """Recall pipeline replay — interactive query with layer-by-layer trace."""
    mem = _get_mem(request)
    ctx = {
        "request": request,
        **_common_context(request, mem),
    }
    return _TEMPLATES.TemplateResponse(request, "memories/recall.html", ctx)


async def recall_json(request: Request) -> JSONResponse:
    """Execute recall with debug trace and return per-layer results."""
    mem = _get_mem(request)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    namespace = body.get("namespace") or None
    budget = int(body.get("budget", 2000))

    retrieval = getattr(mem, "_retrieval", None)
    if retrieval is None:
        return JSONResponse({"error": "No retrieval orchestrator"}, status_code=500)

    try:
        trace = retrieval.recall_debug(query, token_budget=budget, namespace=namespace)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse(trace)


async def timeline_page(request: Request) -> HTMLResponse:
    """Temporal timeline — visualise memory validity, supersession, and as-of replay."""
    mem = _get_mem(request)
    ctx = {"request": request, **_common_context(request, mem)}
    return _TEMPLATES.TemplateResponse(request, "memories/timeline.html", ctx)


async def timeline_json(request: Request) -> JSONResponse:
    """Return memories sorted by created_at for timeline rendering."""
    mem = _get_mem(request)

    namespace = request.query_params.get("namespace") or None
    entity = request.query_params.get("entity") or None
    status = request.query_params.get("status") or None
    date_from = request.query_params.get("from") or None
    date_to = request.query_params.get("to") or None
    as_of = request.query_params.get("as_of") or None

    try:
        search_status = status if status and status != "all" else None
        results = mem.search(
            query=None, namespace=namespace, status=search_status, limit=500,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    memories = []
    for m in results:
        if entity and (getattr(m, "entity", None) or "").lower() != entity.lower():
            continue
        created = getattr(m, "created_at", "") or ""
        if date_from and created[:10] < date_from:
            continue
        if date_to and created[:10] > date_to:
            continue
        if as_of:
            vf = getattr(m, "valid_from", "") or ""
            vu = getattr(m, "valid_until", None)
            if vf and as_of < vf:
                continue
            if vu and as_of > vu:
                continue
        memories.append(_memory_to_dict(m))

    memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return JSONResponse({"memories": memories})


async def agents_page(request: Request) -> HTMLResponse:
    """Agent Registry — namespace-level view of agent activity."""
    mem = _get_mem(request)
    ctx = {"request": request, **_common_context(request, mem)}
    return _TEMPLATES.TemplateResponse(request, "memories/agents.html", ctx)


async def agents_json(request: Request) -> JSONResponse:
    """Return namespace-level agent data for the agents UI."""
    mem = _get_mem(request)
    detail_ns = request.query_params.get("namespace") or None
    is_detail = request.query_params.get("detail") == "1"
    page = int(request.query_params.get("page") or 1)
    per_page = 20

    if is_detail and detail_ns:
        try:
            results = mem.search(namespace=detail_ns, limit=500)
        except Exception:
            results = []
        memories = [_memory_to_dict(m) for m in results]
        memories.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        entity_counts: Dict[str, int] = {}
        cat_counts: Dict[str, int] = {}
        for m in memories:
            ent = m.get("entity")
            if ent:
                entity_counts[ent] = entity_counts.get(ent, 0) + 1
            cat = m.get("category") or "general"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        entity_freq = sorted(
            [{"name": k, "count": v} for k, v in entity_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:30]
        categories = sorted(
            [{"name": k, "count": v} for k, v in cat_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )
        start = (page - 1) * per_page
        return JSONResponse({
            "namespace_detail": {
                "namespace": detail_ns,
                "memory_count": len(memories),
                "page": page,
                "latest_date": memories[0].get("created_at") if memories else None,
                "entity_freq": entity_freq,
                "categories": categories,
                "memories": memories[start:start + per_page],
            }
        })

    # Overview: all namespaces
    try:
        all_memories = mem.search(limit=2000)
    except Exception:
        all_memories = []

    ns_data: Dict[str, List[Dict[str, Any]]] = {}
    for m in all_memories:
        d = _memory_to_dict(m)
        ns = d.get("namespace") or "default"
        ns_data.setdefault(ns, []).append(d)

    from datetime import datetime, timedelta

    today = datetime.utcnow().date()
    namespaces = []
    for ns_name, mems in sorted(ns_data.items()):
        mems.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        cat_counts: Dict[str, int] = {}
        entity_counts: Dict[str, int] = {}
        for m in mems:
            cat_counts[m.get("category") or "general"] = cat_counts.get(m.get("category") or "general", 0) + 1
            if m.get("entity"):
                entity_counts[m["entity"]] = entity_counts.get(m["entity"], 0) + 1
        categories = sorted(
            [{"name": k, "count": v} for k, v in cat_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )
        top_entities = sorted(
            [{"name": k, "count": v} for k, v in entity_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5]
        activity = []
        for days_ago in range(13, -1, -1):
            d = today - timedelta(days=days_ago)
            ds = d.isoformat()
            activity.append({"date": ds, "count": sum(1 for m in mems if (m.get("created_at") or "")[:10] == ds)})
        namespaces.append({
            "namespace": ns_name,
            "memory_count": len(mems),
            "latest_date": mems[0].get("created_at") if mems else None,
            "categories": categories,
            "top_entities": top_entities,
            "activity": activity,
        })
    return JSONResponse({"namespaces": namespaces})


async def health_json(request: Request) -> JSONResponse:
    mem = _get_mem(request)
    try:
        return JSONResponse(mem.health())
    except Exception as e:
        return JSONResponse({"healthy": False, "error": str(e)}, status_code=500)


def ui_routes() -> list:
    """Return absolute UI routes — can be appended to any Starlette app."""
    return [
        Route("/", index),
        Route("/ui", index),
        Route("/ui/", index),
        Route("/ui/memories", memories_list),
        Route("/ui/memories/{memory_id}", memory_detail),
        Route("/ui/graph", graph_page),
        Route("/ui/graph.json", graph_json),
        Route("/ui/recall", recall_page),
        Route("/ui/recall.json", recall_json, methods=["POST"]),
        Route("/ui/timeline", timeline_page),
        Route("/ui/timeline.json", timeline_json),
        Route("/ui/agents", agents_page),
        Route("/ui/agents.json", agents_json),
        Route("/ui/health.json", health_json),
        Mount(
            "/ui/static",
            StaticFiles(directory=str(_UI_DIR / "static")),
            name="static",
        ),
    ]


def create_ui_app() -> Starlette:
    """Standalone Starlette app for the UI. Use when running `memwright ui`."""
    return Starlette(routes=ui_routes())


app = create_ui_app()
