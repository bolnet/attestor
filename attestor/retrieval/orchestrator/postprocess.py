"""Post-processing mixin for :class:`RetrievalOrchestrator`.

Houses ``_post_process_candidates`` — Steps 2-6 of the recall pipeline
shared by both the sync ``recall()`` and the async ``recall_async()``.
Pure CPU-bound work after the two I/O lanes (vector + BM25) return:
candidate materialization, RRF blend, graph narrow, triple injection,
MMR diversity, confidence decay, budget fit.

Bodies are byte-identical to the pre-split ``orchestrator.py``; only the
enclosing class name (``_OrchestratorPostProcessMixin``) is new.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.scorer import (
    confidence_decay_boost,
    deduplicate,
    entity_boost,
    fit_to_budget,
    mmr_rerank,
    temporal_boost,
)
from attestor.retrieval.trace import write as trace_write


class _OrchestratorPostProcessMixin:
    """Steps 2-6 shared between sync recall and async recall."""

    def _post_process_candidates(
        self,
        *,
        query: str,
        namespace: str | None,
        as_of: Any | None,
        time_window: Any | None,
        question_entities: list[str],
        vector_hits_raw: list[dict],
        bm25_hits_raw: list,
        mq_used: list[str] | None,
        path: str,
        token_budget: int,
        t_total: float,
    ) -> list[RetrievalResult]:
        from attestor import trace as _tr
        results: list[RetrievalResult] = []

        # ── Step 2: Materialise vector candidates with preliminary vector_sim
        require_active = (as_of is None and time_window is None)
        candidates: list[dict] = []
        seen_ids: set = set()
        _drop: Counter[str] = Counter()
        # First-drop diagnostic samples (string-valued, not counters).
        _first_drop_status: str | None = None
        _first_drop_namespace: str | None = None
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
                if _first_drop_status is None:
                    _first_drop_status = memory.status
                continue
            if namespace and memory.namespace != namespace:
                _drop["namespace"] += 1
                if _first_drop_namespace is None:
                    _first_drop_namespace = memory.namespace
                continue
            distance = float(vr.get("distance", 1.0))
            distance_sim = max(0.0, 1.0 - distance)
            if _merged_via_rrf:
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
                      sample_inactive_status=_first_drop_status,
                      sample_other_namespace=_first_drop_namespace,
                      require_active=require_active,
                      filter_namespace=namespace)

        # Pull BM25-only hits.
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
            fused_rank = {mid: i for i, mid in enumerate(fused)}
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
