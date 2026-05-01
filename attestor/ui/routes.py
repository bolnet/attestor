"""HTTP route handlers for the Attestor read-only UI.

Handlers are bound to the shared ``Jinja2Templates`` instance via
``build_routes(templates)`` — the function returns a list of
``Route`` objects ready to be appended to a Starlette router.

All handlers are GET except ``/ui/recall.json`` which is POST.
None of them mutate the underlying store.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from attestor.ui._state import common_context, get_mem, is_htmx
from attestor.ui.filters import (
    filter_query_string,
    filters_display,
    memory_to_dict,
    parse_filters,
    search_with_filters,
)


async def index(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/ui/memories", status_code=302)


def build_routes(templates: Jinja2Templates) -> list[Route]:
    """Build the page + JSON-data routes bound to the shared templates instance."""

    async def memories_list(request: Request) -> HTMLResponse:
        mem = get_mem(request)
        per_page = 60
        page = max(1, int(request.query_params.get("page") or 1))

        filters = parse_filters(request)

        # Fetch one extra to detect a next page
        results = search_with_filters(mem, filters, limit=(page * per_page) + 1)

        # Slice for the current page
        offset = (page - 1) * per_page
        page_results = results[offset : offset + per_page]
        has_next = len(results) > offset + per_page

        memories = [memory_to_dict(m) for m in page_results]

        # Assign a subtle, deterministic rotation per card so the page feels pinned.
        for i, m in enumerate(memories):
            m["tilt"] = ((hash(m["id"]) % 7) - 3) * 0.22

        ctx = {
            "request": request,
            "memories": memories,
            "total": len(memories),
            "page": page,
            "per_page": per_page,
            "has_next": has_next,
            "has_prev": page > 1,
            "filters": filters_display(filters),
            "filter_qs": filter_query_string(filters),
            **common_context(request, mem),
        }

        template = "memories/_grid.html" if is_htmx(request) else "memories/list.html"
        return templates.TemplateResponse(request, template, ctx)

    async def memory_detail(request: Request) -> HTMLResponse:
        mem = get_mem(request)
        memory_id = request.path_params["memory_id"]

        m = mem.get(memory_id)
        if m is None:
            return HTMLResponse("<h1>Not found</h1>", status_code=404)

        data = memory_to_dict(m)
        tab = request.query_params.get("tab", "content")

        # Supersession chain
        chain: list[dict[str, Any]] = []
        cursor = m
        while cursor is not None and getattr(cursor, "superseded_by", None):
            nxt_id = cursor.superseded_by
            nxt = mem.get(nxt_id)
            if nxt is None or nxt.id in {c["id"] for c in chain}:
                break
            chain.append(memory_to_dict(nxt))
            cursor = nxt

        # Predecessors (who did this supersede?) — query store directly
        predecessors: list[dict[str, Any]] = []
        try:
            # naive: scan memories superseded_by this id
            for other in mem.search(namespace=m.namespace, limit=500, status="superseded"):
                if getattr(other, "superseded_by", None) == m.id:
                    predecessors.append(memory_to_dict(other))
        except Exception:
            pass

        # Nearest vector neighbors
        neighbors: list[dict[str, Any]] = []
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
                    d = memory_to_dict(other)
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
        graph_edges: list[dict[str, Any]] = []
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
            **common_context(request, mem),
        }

        return templates.TemplateResponse(request, "memories/detail.html", ctx)

    async def graph_page(request: Request) -> HTMLResponse:
        """Full knowledge graph visualization via Cytoscape.js."""
        mem = get_mem(request)

        graph_stats: dict[str, Any] = {}
        try:
            g = getattr(mem, "_graph", None)
            if g is not None:
                graph_stats = g.graph_stats()
        except Exception:
            pass

        ctx = {
            "request": request,
            "graph_stats": graph_stats,
            **common_context(request, mem),
        }
        return templates.TemplateResponse(request, "memories/graph.html", ctx)

    async def graph_json(request: Request) -> JSONResponse:
        """Return full graph data for Cytoscape.js consumption."""
        mem = get_mem(request)

        g = getattr(mem, "_graph", None)
        if g is None:
            return JSONResponse({"nodes": [], "edges": [], "stats": {}, "pagerank": {}})

        # Optional entity-type filter
        entity_type = request.query_params.get("type") or None

        entities = g.get_entities(entity_type=entity_type)
        pr: dict[str, float] = {}
        try:
            pr = mem.pagerank() or {}
        except Exception:
            pass

        stats: dict[str, Any] = {}
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
        mem = get_mem(request)
        ctx = {
            "request": request,
            **common_context(request, mem),
        }
        return templates.TemplateResponse(request, "memories/recall.html", ctx)

    async def recall_json(request: Request) -> JSONResponse:
        """Execute recall with debug trace and return per-layer results."""
        mem = get_mem(request)

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
        mem = get_mem(request)
        ctx = {"request": request, **common_context(request, mem)}
        return templates.TemplateResponse(request, "memories/timeline.html", ctx)

    async def timeline_json(request: Request) -> JSONResponse:
        """Return memories sorted by created_at for timeline rendering."""
        mem = get_mem(request)

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
            memories.append(memory_to_dict(m))

        memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return JSONResponse({"memories": memories})

    async def agents_page(request: Request) -> HTMLResponse:
        """Agent Registry — namespace-level view of agent activity."""
        mem = get_mem(request)
        ctx = {"request": request, **common_context(request, mem)}
        return templates.TemplateResponse(request, "memories/agents.html", ctx)

    async def agents_json(request: Request) -> JSONResponse:
        """Return namespace-level agent data for the agents UI."""
        mem = get_mem(request)
        detail_ns = request.query_params.get("namespace") or None
        is_detail = request.query_params.get("detail") == "1"
        page = int(request.query_params.get("page") or 1)
        per_page = 20

        if is_detail and detail_ns:
            try:
                results = mem.search(namespace=detail_ns, limit=500)
            except Exception:
                results = []
            memories = [memory_to_dict(m) for m in results]
            memories.sort(key=lambda m: m.get("created_at") or "", reverse=True)
            entity_counts: dict[str, int] = {}
            cat_counts: dict[str, int] = {}
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

        ns_data: dict[str, list[dict[str, Any]]] = {}
        for m in all_memories:
            d = memory_to_dict(m)
            ns = d.get("namespace") or "default"
            ns_data.setdefault(ns, []).append(d)

        today = datetime.utcnow().date()
        namespaces = []
        for ns_name, mems in sorted(ns_data.items()):
            mems.sort(key=lambda m: m.get("created_at") or "", reverse=True)
            cat_counts = {}
            entity_counts = {}
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

    async def health_page(request: Request) -> HTMLResponse:
        """System Health dashboard — component diagnostics with auto-refresh."""
        mem = get_mem(request)
        ctx = {"request": request, **common_context(request, mem)}
        return templates.TemplateResponse(request, "memories/health.html", ctx)

    async def health_json(request: Request) -> JSONResponse:
        mem = get_mem(request)
        try:
            return JSONResponse(mem.health())
        except Exception as e:
            return JSONResponse({"healthy": False, "error": str(e)}, status_code=500)

    async def config_page(request: Request) -> HTMLResponse:
        """Configuration viewer — system parameters, backends, retrieval tuning."""
        mem = get_mem(request)
        ctx = {"request": request, **common_context(request, mem)}
        return templates.TemplateResponse(request, "memories/config.html", ctx)

    async def config_json(request: Request) -> JSONResponse:
        """Return full configuration snapshot as JSON."""
        mem = get_mem(request)

        config = mem.config
        retrieval = getattr(mem, "_retrieval", None)  # noqa: F841 (kept for parity)

        from attestor.store.registry import resolve_backends
        backends = getattr(config, "backends", None) or ["postgres", "neo4j"]
        try:
            role_assignments = resolve_backends(backends)
        except Exception:
            role_assignments = {}

        store_paths: dict[str, Any] = {}
        doc_store = getattr(mem, "_store", None)
        if doc_store and hasattr(doc_store, "db_path"):
            store_paths["db_path"] = str(doc_store.db_path)
            try:
                store_paths["db_size_bytes"] = doc_store.db_path.stat().st_size
            except Exception:
                pass
        graph = getattr(mem, "_graph", None)
        if graph and hasattr(graph, "graph_path"):
            store_paths["graph_path"] = str(graph.graph_path)

        embedding: dict[str, Any] = {}
        vector_store = getattr(mem, "_vector_store", None)
        if vector_store:
            if hasattr(vector_store, "provider"):
                embedding["provider"] = vector_store.provider
            if hasattr(vector_store, "count"):
                try:
                    embedding["vector_count"] = vector_store.count()
                except Exception:
                    pass

        retrieval_data: dict[str, Any] = {
            "fusion_mode": getattr(config, "fusion_mode", "rrf"),
            "enable_mmr": getattr(config, "enable_mmr", True),
            "mmr_lambda": getattr(config, "mmr_lambda", 0.7),
            "min_results": getattr(config, "min_results", 3),
            "default_token_budget": getattr(config, "default_token_budget", 16000),
            "confidence_gate": getattr(config, "confidence_gate", 0.0),
            "confidence_decay_rate": getattr(config, "confidence_decay_rate", 0.001),
            "confidence_boost_rate": getattr(config, "confidence_boost_rate", 0.03),
            "rrf_k": 60,
            "graph_bfs_depth": 2,
        }

        health_data: dict[str, Any] = {}
        try:
            health_data = mem.health()
        except Exception:
            health_data = {"healthy": False, "checks": []}

        for check in health_data.get("checks", []):
            if check.get("name") == "Retrieval Pipeline":
                retrieval_data["active_layers"] = check.get("layers", [])
                retrieval_data["max_layers"] = check.get("max_layers", 3)

        stats: dict[str, Any] = {}
        try:
            stats = mem.stats()
        except Exception:
            pass

        graph_stats: dict[str, Any] = {}
        if graph and hasattr(graph, "graph_stats"):
            try:
                graph_stats = graph.graph_stats()
            except Exception:
                pass

        result: dict[str, Any] = {
            "store_path": str(mem.path),
            "backends": backends,
            "role_assignments": role_assignments,
            "store_paths": store_paths,
            "embedding": embedding,
            "retrieval": retrieval_data,
            "stats": stats,
            "graph_stats": graph_stats,
            "backend_configs": getattr(config, "backend_configs", {}),
            "health": health_data,
        }

        return JSONResponse(result)

    async def ops_page(request: Request) -> HTMLResponse:
        """Operations Log — flight recorder of recent add/recall/health calls."""
        mem = get_mem(request)
        ctx = {"request": request, **common_context(request, mem)}
        return templates.TemplateResponse(request, "memories/ops.html", ctx)

    async def ops_json(request: Request) -> JSONResponse:
        """Return the ops ring buffer as JSON (most recent last)."""
        mem = get_mem(request)
        return JSONResponse({"ops": mem.ops_log})

    async def budget_explore_json(request: Request) -> JSONResponse:
        """Run the same query at multiple budgets, return latency + count."""
        import time as _time

        mem = get_mem(request)
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse({"error": "q is required"}, status_code=400)

        namespace = request.query_params.get("namespace") or None
        retrieval = getattr(mem, "_retrieval", None)
        if retrieval is None:
            return JSONResponse({"error": "No retrieval orchestrator"}, status_code=500)

        budgets = [500, 1000, 2000, 5000, 10000]
        results = []
        for b in budgets:
            t0 = _time.monotonic()
            trace = retrieval.recall_debug(query, token_budget=b, namespace=namespace)
            ms = round((_time.monotonic() - t0) * 1000, 2)
            results.append({
                "budget": b,
                "latency_ms": ms,
                "result_count": trace["final_count"],
                "layers": [
                    {"name": l["name"], "count": l["count"], "latency_ms": l.get("latency_ms", 0)}
                    for l in trace.get("layers", [])
                ],
            })
        return JSONResponse({"query": query, "budgets": results})

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
        Route("/ui/health", health_page),
        Route("/ui/health.json", health_json),
        Route("/ui/config", config_page),
        Route("/ui/config.json", config_json),
        Route("/ui/ops", ops_page),
        Route("/ui/ops.json", ops_json),
        Route("/ui/recall/budget-explore.json", budget_explore_json),
    ]
