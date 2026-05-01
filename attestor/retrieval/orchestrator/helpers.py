"""Pipeline helpers for :class:`RetrievalOrchestrator`.

Split from the original ``orchestrator.py`` (1099-line monolith) into a
mixin so the helper methods can stay co-located with the orchestrator's
state (``self.store``, ``self.vector_store``, ``self.graph``,
``self.config``) without growing the public class file.

All bodies are byte-identical to the pre-split version; only the
enclosing class name (``_OrchestratorHelpersMixin``) is new.
"""

from __future__ import annotations

from typing import Any

from attestor.retrieval.bm25 import BM25Hit
from attestor.retrieval.orchestrator.config import GRAPH_MAX_DEPTH
from attestor.retrieval.tag_matcher import extract_tags


class _OrchestratorHelpersMixin:
    """Pipeline-step helpers: question entities, graph affinity, graph
    triples, score blending, Step 0 temporal pre-filter, Step 1 vector
    lane, Step 1b BM25 lane.

    These methods reference ``self`` state (store, vector_store, graph,
    config, tuning attrs) so they remain methods rather than free
    functions.
    """

    # ── Helpers ──

    def _question_entities(self, query: str) -> list[str]:
        """Proper-noun-ish tokens from the question.

        Detects capitalized tokens in the ORIGINAL query (before extract_tags
        lowercases), then intersects with the stopword-filtered tag list so we
        keep meaningful entities (Caroline, LGBTQ) and drop sentence-initial
        function words ('When', 'What', 'The').
        """
        import re as _re
        capitals = {
            m.group(0)
            for m in _re.finditer(r"[A-Z][A-Za-z0-9']*", query)
        }
        tag_set = set(extract_tags(query))  # lowercased, stopword-filtered
        entities: list[str] = []
        seen = set()
        for cap in capitals:
            bare = cap.rstrip("'s").rstrip("'")  # strip possessive
            key = bare.lower()
            if key in tag_set and key not in seen:
                entities.append(bare)
                seen.add(key)
        return entities

    def _graph_affinity_map(
        self,
        question_entities: list[str],
        namespace: str | None = None,
    ) -> dict[str, int]:
        """Map lowercased candidate-entity → min hop distance to any question entity.

        Empty map when no graph is available or the question has no entities.
        ``namespace`` scopes the BFS — without it, a recall in tenant A could
        pull a graph affinity bonus from tenant B's entities. Backends that
        don't accept ``namespace`` (e.g. v3 graph stores) silently ignore it
        via the TypeError fallback below.
        """
        affinity: dict[str, int] = {}
        if not self.graph or not question_entities:
            return affinity

        def _related(entity: str, depth: int) -> list[str]:
            # Try namespace-aware signature first; fall back for backends
            # that haven't adopted the kwarg yet.
            try:
                return self.graph.get_related(
                    entity, depth=depth, namespace=namespace,
                )
            except TypeError:
                return self.graph.get_related(entity, depth=depth)

        for q_ent in question_entities:
            q_key = q_ent.lower()
            # Question entity is hop-distance 0 from itself. The earlier
            # ``min(affinity.get(q_key, 0), 0)`` formulation was a verbose
            # equivalent — no negative values are ever stored.
            affinity[q_key] = 0

            try:
                d1 = _related(q_ent, 1)
            except Exception:
                d1 = []
            for name in d1:
                k = (name or "").lower()
                if k and affinity.get(k, 99) > 1:
                    affinity[k] = 1

            try:
                d2 = _related(q_ent, GRAPH_MAX_DEPTH)
            except Exception:
                d2 = []
            for name in d2:
                k = (name or "").lower()
                if k and affinity.get(k, 99) > GRAPH_MAX_DEPTH:
                    affinity[k] = GRAPH_MAX_DEPTH
        return affinity

    def _graph_context_triples(
        self,
        question_entities: list[str],
        namespace: str | None = None,
    ) -> list[str]:
        """Render typed edges incident to question entities as inline strings.

        Filters edges by ``namespace`` so synthetic-triple injection can't
        leak relationships from other tenants into the recall context.
        """
        triples: list[str] = []
        if not self.graph or not hasattr(self.graph, "get_edges"):
            return triples
        seen = set()
        for ent in question_entities:
            try:
                try:
                    edges = self.graph.get_edges(ent, namespace=namespace)
                except TypeError:
                    edges = self.graph.get_edges(ent)
            except Exception:
                continue
            for edge in edges[:30]:
                subj = edge.get("subject", "")
                pred = edge.get("predicate", "related_to")
                obj = edge.get("object", "")
                date = edge.get("event_date", "")
                quote = edge.get("source_quote", "")
                key = (subj.lower(), pred, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                date_str = f" ({date})" if date else ""
                quote_str = f' — "{quote}"' if quote else ""
                triples.append(f"{subj} {pred} {obj}{date_str}{quote_str}")
        return triples

    def _blend_score(
        self, vector_sim: float, hop: int | None
    ) -> tuple[float, float]:
        """Blend normalized vector similarity with graph affinity bonus.

        Returns (final_score, graph_bonus) for tracing.
        """
        cfg = self.config
        vec_norm = max(0.0, min(1.0, vector_sim))
        if hop is None:
            bonus = cfg.graph_unreachable_penalty
        else:
            bonus = cfg.graph_affinity_bonus.get(hop, 0.0)
        final = (
            cfg.vector_weight * vec_norm
            + cfg.graph_weight * max(0.0, bonus)
        )
        if hop is None:
            final += cfg.graph_unreachable_penalty
        return final, bonus

    # ──────────────────────────────────────────────────────────────────
    # Step 0 helper — temporal pre-filter (RC4).
    # ──────────────────────────────────────────────────────────────────

    def _step0_temporal_prefilter(
        self, query: str, time_window: Any | None,
    ) -> Any | None:
        """When enabled and the question contains a relative time phrase,
        tighten the event-time window passed to subsequent lanes. Caller-
        supplied ``time_window`` takes precedence — never override an
        explicit bound. Falls through silently when no phrase matches.
        """
        from attestor import trace as _tr
        tp_cfg = getattr(self, "temporal_prefilter_cfg", None)
        if (
            tp_cfg is not None
            and getattr(tp_cfg, "enabled", False)
            and time_window is None
        ):
            from attestor.retrieval.temporal_prefilter import detect_window
            detected = detect_window(
                query, tolerance_days=int(tp_cfg.tolerance_days),
            )
            if detected is not None:
                time_window = detected.window
                if _tr.is_enabled():
                    _tr.event(
                        "recall.stage.temporal_prefilter",
                        matched_phrase=detected.matched_text,
                        target=detected.target.isoformat(),
                        window_start=(
                            detected.window.start.isoformat()
                            if detected.window.start else None
                        ),
                        window_end=(
                            detected.window.end.isoformat()
                            if detected.window.end else None
                        ),
                    )
        return time_window

    # ──────────────────────────────────────────────────────────────────
    # Step 1 helper — vector lane (single | hyde | multi_query).
    # ──────────────────────────────────────────────────────────────────

    def _compute_lane_vector(
        self,
        query: str,
        namespace: str | None,
        as_of: Any | None,
        time_window: Any | None,
    ) -> tuple[list[dict[str, Any]], list[str] | None, str]:
        """Returns ``(vector_hits_raw, mq_used, path)``.

        Path priority: multi_query > hyde > single — mutually exclusive.
        If both flags are on, prefers multi_query (longer track record)
        and logs a warning.
        """
        from attestor import trace as _tr
        vector_hits_raw: list[dict] = []
        mq_used: list[str] | None = None

        def _single_vector_search(q: str) -> list[dict]:
            """One vector_store.search call with v3/v4 signature
            fallback. Returns [] on any exception so a degraded
            backend can't crash the whole recall."""
            if not self.vector_store:
                return []
            try:
                try:
                    return list(self.vector_store.search(
                        q, limit=self.vector_top_k, namespace=namespace,
                        as_of=as_of, time_window=time_window,
                    ))
                except TypeError:
                    return list(self.vector_store.search(
                        q, limit=self.vector_top_k, namespace=namespace,
                    ))
            except Exception:
                return []

        path = "single"
        if self.vector_store:
            mq_cfg = getattr(self, "multi_query_cfg", None)
            hyde_cfg = getattr(self, "hyde_cfg", None)
            mq_enabled = bool(getattr(mq_cfg, "enabled", False))
            hyde_enabled = bool(getattr(hyde_cfg, "enabled", False))

            if mq_enabled and hyde_enabled:
                import logging as _log
                _log.getLogger("attestor.retrieval.orchestrator").warning(
                    "retrieval.multi_query and retrieval.hyde both enabled; "
                    "preferring multi_query (HyDE skipped this run).",
                )
                hyde_enabled = False

            if mq_enabled:
                from attestor.retrieval.multi_query import multi_query_search

                mq_used, vector_hits_raw = multi_query_search(
                    query,
                    vector_search=_single_vector_search,
                    n=int(getattr(mq_cfg, "n", 3)),
                    merge=str(getattr(mq_cfg, "merge", "rrf")),
                    rewriter_model=getattr(mq_cfg, "rewriter_model", None),
                )
                path = "multi_query"
            elif hyde_enabled:
                from attestor.retrieval.hyde import hyde_search

                mq_used, vector_hits_raw = hyde_search(
                    query,
                    vector_search=_single_vector_search,
                    model=getattr(hyde_cfg, "generator_model", None),
                    merge=str(getattr(hyde_cfg, "merge", "rrf")),
                )
                path = "hyde"
            else:
                vector_hits_raw = _single_vector_search(query)

        if _tr.is_enabled():
            _tr.event("recall.stage.vector",
                      top_k=self.vector_top_k, hit_count=len(vector_hits_raw),
                      multi_query=bool(mq_used),
                      n_queries=(len(mq_used) if mq_used else 1),
                      path=path,
                      hits=[{"id": h.get("memory_id"),
                             "distance": round(float(h.get("distance", 1.0)), 4)}
                            for h in vector_hits_raw[:10]])

        return vector_hits_raw, mq_used, path

    # ──────────────────────────────────────────────────────────────────
    # Step 1b helper — BM25 lane.
    # ──────────────────────────────────────────────────────────────────

    def _compute_lane_bm25(
        self,
        query: str,
        as_of: Any | None,
        time_window: Any | None,
    ) -> list[BM25Hit]:
        """BM25 lane (Postgres FTS or in-memory rank_bm25 in
        experiments). Runs alongside the vector lane; fused via RRF
        in step 2b. No-op when the lane isn't configured."""
        from attestor import trace as _tr
        bm25_hits_raw: list[BM25Hit] = []
        if self.bm25_lane is not None:
            try:
                bm25_hits_raw = self.bm25_lane.search(
                    query, limit=self.bm25_top_k,
                    min_rank=self.bm25_min_rank,
                    as_of=as_of, time_window=time_window,
                    # When replaying past belief OR slicing an event-time
                    # window, superseded rows are valid answers; drop the
                    # current-state filter.
                    active_only=(as_of is None and time_window is None),
                )
            except Exception:
                bm25_hits_raw = []
        if _tr.is_enabled():
            _tr.event("recall.stage.bm25",
                      enabled=self.bm25_lane is not None,
                      hit_count=len(bm25_hits_raw),
                      hits=[{"id": h.memory_id,
                             "rank": getattr(h, "rank", None)}
                            for h in bm25_hits_raw[:10]])
        return bm25_hits_raw
