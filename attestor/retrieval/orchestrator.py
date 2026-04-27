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
        self, question_entities: List[str]
    ) -> Dict[str, int]:
        """Map lowercased candidate-entity → min hop distance to any question entity.

        Empty map when no graph is available or the question has no entities.
        """
        affinity: Dict[str, int] = {}
        if not self.graph or not question_entities:
            return affinity

        for q_ent in question_entities:
            q_key = q_ent.lower()
            affinity[q_key] = min(affinity.get(q_key, 0), 0)

            try:
                d1 = self.graph.get_related(q_ent, depth=1)
            except Exception:
                d1 = []
            for name in d1:
                k = (name or "").lower()
                if k and affinity.get(k, 99) > 1:
                    affinity[k] = 1

            try:
                d2 = self.graph.get_related(q_ent, depth=GRAPH_MAX_DEPTH)
            except Exception:
                d2 = []
            for name in d2:
                k = (name or "").lower()
                if k and affinity.get(k, 99) > GRAPH_MAX_DEPTH:
                    affinity[k] = GRAPH_MAX_DEPTH
        return affinity

    def _graph_context_triples(
        self, question_entities: List[str]
    ) -> List[str]:
        """Render typed edges incident to question entities as inline strings."""
        triples: List[str] = []
        if not self.graph or not hasattr(self.graph, "get_edges"):
            return triples
        seen = set()
        for ent in question_entities:
            try:
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
        question_entities = self._question_entities(query)

        # ── Step 1: Vector top-K ──
        vector_hits_raw: List[Dict] = []
        results: List[RetrievalResult] = []
        if self.vector_store:
            try:
                # Newer v4 backends accept as_of/time_window; v3 backends
                # don't — fall back to the legacy signature on TypeError.
                try:
                    vector_hits_raw = self.vector_store.search(
                        query, limit=VECTOR_TOP_K, namespace=namespace,
                        as_of=as_of, time_window=time_window,
                    )
                except TypeError:
                    vector_hits_raw = self.vector_store.search(
                        query, limit=VECTOR_TOP_K, namespace=namespace,
                    )
            except Exception:
                vector_hits_raw = []

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

        # ── Step 2: Materialise vector candidates with preliminary vector_sim
        # status='active' is current-state filtering. When as_of is set,
        # the lane SQL has already restricted by t_created/t_expired so a
        # row that's now 'superseded' but was active at as_of must pass.
        # When time_window is set, the lane filtered by event-time overlap;
        # a fact that was true during the window may now be superseded too.
        require_active = (as_of is None and time_window is None)
        candidates: List[Dict] = []
        seen_ids: set = set()
        for vr in vector_hits_raw:
            mid = vr["memory_id"]
            if mid in seen_ids:
                continue
            memory = self.store.get(mid)
            if not memory:
                continue
            if require_active and memory.status != "active":
                continue
            if namespace and memory.namespace != namespace:
                continue
            distance = float(vr.get("distance", 1.0))
            vector_sim = max(0.0, 1.0 - distance)
            candidates.append(
                {"memory": memory, "distance": distance, "vector_sim": vector_sim}
            )
            seen_ids.add(mid)

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

        # ── Step 2: Graph narrow ──
        affinity_map = self._graph_affinity_map(question_entities)
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
        triple_strs = self._graph_context_triples(question_entities)
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
            results = mmr_rerank(results, lambda_param=self.mmr_lambda)

        # ── Step 5: Confidence decay ──
        results = confidence_decay_boost(
            results,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )

        # ── Step 6: Fit to budget ──
        final = fit_to_budget(results, token_budget)

        trace_write({
            "kind": "recall",
            "query": query,
            "namespace": namespace,
            "token_budget": token_budget,
            "question_entities": question_entities,
            "vector_top_k": VECTOR_TOP_K,
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
                    query, limit=VECTOR_TOP_K, namespace=namespace
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
            "description": f"Top {VECTOR_TOP_K} by cosine similarity",
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
        affinity_map = self._graph_affinity_map(question_entities)
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
        triple_strs = self._graph_context_triples(question_entities)
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
                "vector_top_k": VECTOR_TOP_K,
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
