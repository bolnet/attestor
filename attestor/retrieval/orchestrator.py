"""Semantic-first retrieval orchestrator.

Pipeline (permanent as of 2026-04-19):
    1. Vector semantic search (top-K=50)
    2. Graph narrow — soft boost by hop-distance from each hit's entity
       to the question entities (BFS depth ≤ 2 via Neo4j)
    3. Inject typed-edge triples as synthetic memories (for LLM reasoning)
    4. MMR diversity rerank (if enabled)
    5. Confidence decay + temporal boost (if enabled)
    6. Fit to token budget

Every call writes a JSONL trace to ``logs/attestor_trace.jsonl``
(disable via ``ATTESTOR_TRACE=0``).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.scorer import (
    confidence_decay_boost,
    deduplicate,
    entity_boost,
    fit_to_budget,
    mmr_rerank,
    temporal_boost,
)
from attestor.retrieval.tag_matcher import extract_tags
from attestor.retrieval.trace import write as trace_write
from attestor.store.base import DocumentStore


VECTOR_TOP_K = 50
GRAPH_MAX_DEPTH = 2
# Affinity bonus added to the normalized vector score per hop distance.
GRAPH_AFFINITY_BONUS = {0: 0.30, 1: 0.20, 2: 0.10}
GRAPH_UNREACHABLE_PENALTY = -0.05
VECTOR_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3


class RetrievalOrchestrator:
    """Semantic-first retrieval with graph narrowing."""

    def __init__(
        self,
        store: DocumentStore,
        min_results: int = 3,
        vector_store=None,
        graph=None,
        enable_temporal_boost: bool = True,
        bm25_lane=None,
    ):
        self.store = store
        self.min_results = min_results
        self.vector_store = vector_store
        self.graph = graph
        self.enable_temporal_boost = enable_temporal_boost
        self.bm25_lane = bm25_lane            # Phase 4.3 — keyword/FTS lane
        self.bm25_top_k: int = 30             # how many BM25 hits to feed to RRF
        self.bm25_min_rank: float = 0.0       # drop hits below this rank
        self.confidence_gate: float = 0.0
        self.confidence_decay_rate: float = 0.001
        self.confidence_boost_rate: float = 0.03
        self.enable_mmr: bool = True
        self.mmr_lambda: float = 0.7
        # Retained for callers; has no effect in semantic-first pipeline.
        self.fusion_mode: str = "semantic_first"

        # Vector-lane top-K + MMR cap. Defaults match the legacy
        # constants below; AgentMemory wires the resolved YAML values
        # post-construction so callers can also tune via Python.
        self.vector_top_k: int = VECTOR_TOP_K
        self.mmr_top_n: Optional[int] = None  # None = no cap

        # Multi-query lane (Phase 3 PR-C). Defaults to None which means
        # "single-query path" — the legacy behavior. AgentMemory wires
        # the resolved MultiQueryCfg from configs/attestor.yaml after
        # construction. When ``multi_query_cfg.enabled`` is True, Step 1
        # runs N+1 vector searches (original + N rewrites) and merges
        # them via RRF before the rest of the cascade.
        self.multi_query_cfg = None  # type: ignore[assignment]

        # Temporal pre-filter (Phase 3 RC4). Defaults to None — no
        # behavior change. AgentMemory wires the resolved
        # TemporalPrefilterCfg post-construction. When
        # ``temporal_prefilter_cfg.enabled`` is True and the question
        # contains a relative time phrase, recall() builds a
        # ``TimeWindow`` and passes it through the existing
        # ``time_window`` kwarg before Step 1 runs.
        self.temporal_prefilter_cfg = None  # type: ignore[assignment]

        # HyDE retrieval (Phase 3 PR-D). Sibling of multi_query.
        # Defaults to None — no behavior change. AgentMemory wires
        # the resolved HydeCfg post-construction. Mutually exclusive
        # with multi_query in this PR — if both are enabled, the
        # orchestrator prefers multi_query (it shipped first + has
        # the longer track record) and logs a warning.
        self.hyde_cfg = None  # type: ignore[assignment]

    # ── Helpers ──

    def _question_entities(self, query: str) -> List[str]:
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
        entities: List[str] = []
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
        question_entities: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, int]:
        """Map lowercased candidate-entity → min hop distance to any question entity.

        Empty map when no graph is available or the question has no entities.
        ``namespace`` scopes the BFS — without it, a recall in tenant A could
        pull a graph affinity bonus from tenant B's entities. Backends that
        don't accept ``namespace`` (e.g. v3 graph stores) silently ignore it
        via the TypeError fallback below.
        """
        affinity: Dict[str, int] = {}
        if not self.graph or not question_entities:
            return affinity

        def _related(entity: str, depth: int) -> List[str]:
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
            affinity[q_key] = min(affinity.get(q_key, 0), 0)

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
        question_entities: List[str],
        namespace: Optional[str] = None,
    ) -> List[str]:
        """Render typed edges incident to question entities as inline strings.

        Filters edges by ``namespace`` so synthetic-triple injection can't
        leak relationships from other tenants into the recall context.
        """
        triples: List[str] = []
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
        self, vector_sim: float, hop: Optional[int]
    ) -> tuple[float, float]:
        """Blend normalized vector similarity with graph affinity bonus.

        Returns (final_score, graph_bonus) for tracing.
        """
        vec_norm = max(0.0, min(1.0, vector_sim))
        if hop is None:
            bonus = GRAPH_UNREACHABLE_PENALTY
        else:
            bonus = GRAPH_AFFINITY_BONUS.get(hop, 0.0)
        final = VECTOR_WEIGHT * vec_norm + GRAPH_WEIGHT * max(0.0, bonus)
        if hop is None:
            final += GRAPH_UNREACHABLE_PENALTY
        return final, bonus

    # ── Public API ──

    def recall(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
        as_of: Optional[Any] = None,           # datetime; Phase 5.3
        time_window: Optional[Any] = None,     # TimeWindow; Phase 5.3
    ) -> List[RetrievalResult]:
        """Semantic-first recall with graph narrowing.

        Phase 5.3 — bi-temporal:
          as_of       — replay past belief: see only memories the system
                        BELIEVED were true at that timestamp
          time_window — pre-filter by EVENT-time overlap (when the fact
                        was true in the world)

        Both kwargs are passed through to the vector + BM25 lanes; lanes
        that don't support them ignore them silently. Behavior unchanged
        when both are None.
        """
        t_total = time.monotonic()
        from attestor import trace as _tr
        question_entities = self._question_entities(query)
        if _tr.is_enabled():
            _tr.event("recall.start", query=query[:200], namespace=namespace,
                      token_budget=token_budget,
                      question_entities=question_entities,
                      as_of=str(as_of) if as_of else None,
                      time_window=str(time_window) if time_window else None)

        # ── Step 0: Temporal pre-filter (RC4) ──
        # When enabled and the question contains a relative time phrase,
        # tighten the event-time window passed to subsequent lanes. Caller-
        # supplied time_window takes precedence — never override an explicit
        # bound. Falls through silently when no phrase matches.
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

        # ── Step 1: Vector top-K ──
        # When multi_query is enabled, this step fans out: rewrite the
        # question into N paraphrases via a small LLM call, run each
        # through ``vector_store.search`` independently, and RRF-merge
        # the N+1 ranked lists. The merged list flows into the rest
        # of the cascade unchanged. Disabled (default) → single-query
        # legacy path; behavior identical to before this change.
        vector_hits_raw: List[Dict] = []
        results: List[RetrievalResult] = []
        mq_used: Optional[List[str]] = None  # populated when multi_query fired

        def _single_vector_search(q: str) -> List[Dict]:
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

        # Path priority: multi_query > hyde > single. They are
        # mutually exclusive in this PR — combining them is a future
        # follow-up. If both flags are on, prefer multi_query (longer
        # track record) and warn loudly so the operator notices.
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

        # ── Step 1b: BM25 lane (Phase 4.3) ──
        # Runs alongside the vector lane; fused via RRF below. When the
        # bm25_lane isn't configured (v3 deployments, custom backends),
        # this is a no-op and recall behaves exactly as before.
        bm25_hits_raw: List = []
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

        # ── Step 2: Materialise vector candidates with preliminary vector_sim
        # status='active' is current-state filtering. When as_of is set,
        # the lane SQL has already restricted by t_created/t_expired so a
        # row that's now 'superseded' but was active at as_of must pass.
        # When time_window is set, the lane filtered by event-time overlap;
        # a fact that was true during the window may now be superseded too.
        require_active = (as_of is None and time_window is None)
        candidates: List[Dict] = []
        seen_ids: set = set()
        _drop = {"missing": 0, "inactive": 0, "namespace": 0, "kept": 0,
                 "first_drop_status": None, "first_drop_namespace": None}
        # When multi-query merged the lanes via RRF, the hit order
        # encodes consensus signal that the per-hit ``distance`` field
        # (carried over from one lane only) does NOT. Convert merged
        # rank → similarity so a consensus hit at rank-3 outranks a
        # lone-top hit at rank-1 even though its lane-0 distance was
        # worse. Inactive when no rrf_score is present (legacy path).
        _merged_via_rrf = (
            bool(vector_hits_raw)
            and "rrf_score" in vector_hits_raw[0]
        )
        _total_merged = max(1, len(vector_hits_raw))
        for _idx, vr in enumerate(vector_hits_raw):
            mid = vr["memory_id"]
            if mid in seen_ids:
                continue
            memory = self.store.get(mid)
            if not memory:
                _drop["missing"] += 1
                continue
            if require_active and memory.status != "active":
                _drop["inactive"] += 1
                if _drop["first_drop_status"] is None:
                    _drop["first_drop_status"] = memory.status
                continue
            if namespace and memory.namespace != namespace:
                _drop["namespace"] += 1
                if _drop["first_drop_namespace"] is None:
                    _drop["first_drop_namespace"] = memory.namespace
                continue
            distance = float(vr.get("distance", 1.0))
            distance_sim = max(0.0, 1.0 - distance)
            if _merged_via_rrf:
                # Rank-derived sim from RRF order. Item at idx=0 →
                # 1.0; last → near 0. Use max() so a consensus item
                # also gets credit for any genuinely-good distance.
                rank_sim = 1.0 - (_idx / _total_merged)
                vector_sim = max(distance_sim, rank_sim)
            else:
                vector_sim = distance_sim
            candidates.append(
                {"memory": memory, "distance": distance, "vector_sim": vector_sim}
            )
            seen_ids.add(mid)
            _drop["kept"] += 1
        if _tr.is_enabled():
            _tr.event("recall.stage.candidates",
                      vector_in=len(vector_hits_raw),
                      kept=_drop["kept"],
                      dropped_missing=_drop["missing"],
                      dropped_inactive=_drop["inactive"],
                      dropped_namespace=_drop["namespace"],
                      sample_inactive_status=_drop["first_drop_status"],
                      sample_other_namespace=_drop["first_drop_namespace"],
                      require_active=require_active,
                      filter_namespace=namespace)

        # Pull BM25-only hits into the candidate set (vector_sim=0 for those).
        # The RRF-style boost below rewards them based on their BM25 rank.
        for hit in bm25_hits_raw:
            if hit.memory_id in seen_ids:
                continue
            memory = self.store.get(hit.memory_id)
            if not memory:
                continue
            if require_active and memory.status != "active":
                continue
            if namespace and memory.namespace != namespace:
                continue
            candidates.append({
                "memory": memory, "distance": 1.0, "vector_sim": 0.0,
            })
            seen_ids.add(hit.memory_id)

        # ── Step 2b: RRF-blend vector + BM25 rank into a unified score ──
        if bm25_hits_raw:
            from attestor.retrieval.bm25 import reciprocal_rank_fusion
            vector_ranked = [vr["memory_id"] for vr in vector_hits_raw]
            bm25_ranked = [h.memory_id for h in bm25_hits_raw]
            fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
            # Map id → fused position (0-based)
            fused_rank = {mid: i for i, mid in enumerate(fused)}
            # Convert fused position into a normalized [0, 1] bonus that we
            # add to vector_sim. Top of fused list ≈ 1.0, tail decays slowly.
            n = max(1, len(fused))
            for c in candidates:
                pos = fused_rank.get(c["memory"].id)
                if pos is not None:
                    c["vector_sim"] = max(
                        c["vector_sim"], 1.0 - (pos / n),
                    )
            if _tr.is_enabled():
                _tr.event("recall.stage.rrf",
                          fused_count=len(fused),
                          top_fused=[{"id": mid, "rank": i}
                                     for i, mid in enumerate(fused[:10])])

        # ── Step 2: Graph narrow ──
        # Pass namespace through so the BFS can't pull affinity from
        # entities written by another tenant. Pre-namespace nodes still
        # recall under namespace="default" via coalesce in the backend.
        affinity_map = self._graph_affinity_map(
            question_entities, namespace=namespace,
        )
        if _tr.is_enabled():
            _tr.event("recall.stage.graph",
                      query_entities=question_entities,
                      reachable_entity_count=len(affinity_map),
                      affinity_sample={k: v for k, v in
                                       list(affinity_map.items())[:10]})
        trace_hits = []
        for c in candidates:
            mem = c["memory"]
            ent_key = (mem.entity or "").lower()
            hop = affinity_map.get(ent_key)
            final_score, bonus = self._blend_score(c["vector_sim"], hop)
            results.append(
                RetrievalResult(
                    memory=mem, score=final_score, match_source="vector",
                )
            )
            trace_hits.append({
                "memory_id": mem.id,
                "entity": mem.entity,
                "namespace": mem.namespace,
                "category": mem.category,
                "distance": round(c["distance"], 4),
                "vector_sim": round(c["vector_sim"], 4),
                "graph_hop": hop if hop is not None else -1,
                "graph_bonus": round(bonus, 4),
                "final_score": round(final_score, 4),
                "content_preview": mem.content[:160],
            })

        # Sort by blended score
        results.sort(key=lambda r: r.score, reverse=True)
        ranked_preview = [
            {"id": r.memory.id, "score": round(r.score, 4),
             "entity": r.memory.entity, "namespace": r.memory.namespace}
            for r in results[:30]
        ]

        # ── Step 3: Inject synthetic triple memories ──
        triple_strs = self._graph_context_triples(
            question_entities, namespace=namespace,
        )
        for triple_str in triple_strs[:20]:
            results.append(
                RetrievalResult(
                    memory=Memory(
                        content=triple_str,
                        category="graph_relation",
                        tags=[],
                    ),
                    score=0.6,
                    match_source="graph",
                )
            )

        results = deduplicate(results)
        results = temporal_boost(results, enabled=self.enable_temporal_boost)
        results = entity_boost(results, question_entities or None)

        # ── Step 4: MMR diversity ──
        if self.enable_mmr:
            _pre_mmr_count = len(results)
            results = mmr_rerank(results, lambda_param=self.mmr_lambda)
            # Apply post-MMR cap if configured. None = legacy behavior
            # (no cap; let mmr_rerank's own diversity trim decide).
            if self.mmr_top_n is not None and len(results) > self.mmr_top_n:
                results = results[: self.mmr_top_n]
            if _tr.is_enabled():
                _tr.event("recall.stage.mmr",
                          lambda_=self.mmr_lambda,
                          mmr_top_n=self.mmr_top_n,
                          in_count=_pre_mmr_count, out_count=len(results))

        # ── Step 5: Confidence decay ──
        results = confidence_decay_boost(
            results,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )

        # ── Step 6: Fit to budget ──
        _pre_pack_count = len(results)
        final = fit_to_budget(results, token_budget)
        if _tr.is_enabled():
            _tr.event("recall.stage.pack",
                      token_budget=token_budget,
                      in_count=_pre_pack_count, out_count=len(final))
            _tr.event("recall.done",
                      query=query[:120],
                      final_count=len(final),
                      latency_ms=round((time.monotonic() - t_total) * 1000, 2),
                      final_ids=[
                          {"id": r.memory.id, "score": round(r.score, 4),
                           "source": r.match_source,
                           "preview": r.memory.content[:80]}
                          for r in final[:20]
                      ])

        trace_write({
            "kind": "recall",
            "query": query,
            "namespace": namespace,
            "token_budget": token_budget,
            "question_entities": question_entities,
            "vector_top_k": self.vector_top_k,
            "vector_hits": trace_hits,
            "graph_triples_injected": len(triple_strs[:20]),
            "ranked_after_blend": ranked_preview,
            "final_count": len(final),
            "final_ids": [
                {"id": r.memory.id, "score": round(r.score, 4),
                 "source": r.match_source, "entity": r.memory.entity}
                for r in final
            ],
            "latency_ms": round((time.monotonic() - t_total) * 1000, 2),
        })

        return final

    def recall_debug(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
    ) -> Dict:
        """Per-step trace for UI / ad-hoc inspection (mirrors the JSONL trace)."""
        t_total = time.monotonic()
        question_entities = self._question_entities(query)
        layers: List[Dict] = []

        # Vector
        t = time.monotonic()
        vector_hits_raw: List[Dict] = []
        candidates: List[Dict] = []
        vector_error: Optional[str] = None
        if self.vector_store:
            try:
                vector_hits_raw = self.vector_store.search(
                    query, limit=self.vector_top_k, namespace=namespace
                )
            except Exception as e:
                vector_hits_raw = []
                vector_error = f"{type(e).__name__}: {e}"
        else:
            vector_error = "no vector_store configured"
        for vr in vector_hits_raw:
            mid = vr["memory_id"]
            memory = self.store.get(mid)
            if not memory or memory.status != "active":
                continue
            if namespace and memory.namespace != namespace:
                continue
            distance = float(vr.get("distance", 1.0))
            candidates.append({
                "memory": memory, "distance": distance,
                "vector_sim": max(0.0, 1.0 - distance),
            })
        vector_layer: Dict = {
            "name": "Vector Top-K",
            "description": f"Top {self.vector_top_k} by cosine similarity",
            "count": len(candidates),
            "latency_ms": round((time.monotonic() - t) * 1000, 2),
            "results": [
                {
                    "id": c["memory"].id,
                    "score": round(c["vector_sim"], 4),
                    "distance": round(c["distance"], 4),
                    "entity": c["memory"].entity,
                    "source": "vector",
                    "content": c["memory"].content[:200],
                    "category": c["memory"].category,
                }
                for c in candidates
            ],
        }
        if vector_error:
            vector_layer["error"] = vector_error
        layers.append(vector_layer)

        # Graph narrow
        t = time.monotonic()
        affinity_map = self._graph_affinity_map(
            question_entities, namespace=namespace,
        )
        results: List[RetrievalResult] = []
        narrowed: List[Dict] = []
        for c in candidates:
            mem = c["memory"]
            hop = affinity_map.get((mem.entity or "").lower())
            final_score, bonus = self._blend_score(c["vector_sim"], hop)
            results.append(
                RetrievalResult(memory=mem, score=final_score, match_source="vector")
            )
            narrowed.append({
                "id": mem.id,
                "graph_hop": hop if hop is not None else -1,
                "graph_bonus": round(bonus, 4),
                "score": round(final_score, 4),
                "entity": mem.entity,
                "source": "vector",
                "content": mem.content[:200],
                "category": mem.category,
            })
        results.sort(key=lambda r: r.score, reverse=True)
        narrowed.sort(key=lambda r: r["score"], reverse=True)
        layers.append({
            "name": "Graph Narrow",
            "description": (
                f"Blend {VECTOR_WEIGHT}·vector + {GRAPH_WEIGHT}·graph; "
                f"hops 0→+0.3, 1→+0.2, 2→+0.1, unreachable→{GRAPH_UNREACHABLE_PENALTY}"
            ),
            "question_entities": question_entities,
            "affinity_map_size": len(affinity_map),
            "count": len(narrowed),
            "latency_ms": round((time.monotonic() - t) * 1000, 2),
            "results": narrowed,
        })

        # Triples + boosts + MMR + budget
        t = time.monotonic()
        triple_strs = self._graph_context_triples(
            question_entities, namespace=namespace,
        )
        for triple_str in triple_strs[:20]:
            results.append(
                RetrievalResult(
                    memory=Memory(
                        content=triple_str, category="graph_relation", tags=[],
                    ),
                    score=0.6, match_source="graph",
                )
            )
        results = deduplicate(results)
        results = temporal_boost(results, enabled=self.enable_temporal_boost)
        results = entity_boost(results, question_entities or None)
        if self.enable_mmr:
            results = mmr_rerank(results, lambda_param=self.mmr_lambda)
        results = confidence_decay_boost(
            results,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )
        final = fit_to_budget(results, token_budget)
        layers.append({
            "name": "Triples + Diversity + Fit",
            "description": (
                f"Inject {len(triple_strs[:20])} triples, temporal+entity boost, "
                f"MMR λ={self.mmr_lambda}, budget={token_budget}"
            ),
            "count": len(final),
            "latency_ms": round((time.monotonic() - t) * 1000, 2),
            "results": [
                {
                    "id": r.memory.id,
                    "score": round(r.score, 4),
                    "source": r.match_source,
                    "entity": r.memory.entity,
                    "content": r.memory.content[:200],
                    "category": r.memory.category,
                }
                for r in final
            ],
        })

        warnings: List[str] = []
        if vector_error:
            warnings.append(
                "Vector lane failed — query did not produce an embedding. "
                "Likely the embedder (Ollama / OpenAI) is unreachable. "
                f"Underlying error: {vector_error}"
            )

        return {
            "query": query,
            "namespace": namespace,
            "token_budget": token_budget,
            "total_latency_ms": round((time.monotonic() - t_total) * 1000, 2),
            "layers": layers,
            "final_count": len(final),
            "warnings": warnings,
            "config": {
                "pipeline": "semantic_first",
                "vector_top_k": self.vector_top_k,
                "vector_weight": VECTOR_WEIGHT,
                "graph_weight": GRAPH_WEIGHT,
                "mmr_lambda": self.mmr_lambda,
                "confidence_gate": self.confidence_gate,
            },
        }

    def recall_as_context(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
    ) -> str:
        results = self.recall(query, token_budget, namespace=namespace)
        if not results:
            return ""
        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return "\n".join(lines)
