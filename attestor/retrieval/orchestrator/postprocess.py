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
    layer_boost,
    mmr_rerank,
    temporal_boost,
)
from attestor.retrieval.trace import write as trace_write
from attestor.utils.tokens import estimate_tokens

# Default cap for long-context mode. Sized for 1M-context answerers
# (Claude Sonnet 4.6 / Opus 4.x / Gemini 2 Pro) — 200_000 tokens
# leaves ~800k for the system prompt + question + downstream reasoning.
_DEFAULT_LONG_CONTEXT_MAX_TOKENS = 200_000


def _long_context_pack(
    candidates: list[RetrievalResult],
    *,
    max_tokens: int,
) -> list[RetrievalResult]:
    """Pack the top-scored candidates verbatim, capped by ``max_tokens``.

    Sorts by score desc, accumulates until adding the next memory would
    exceed the cap, drops it whole (never partial). Reuses the same
    ``estimate_tokens`` helper that ``fit_to_budget`` uses, so cap-vs-
    budget semantics stay aligned.

    Designed for downstream answerers running on 1M-context models —
    no MMR diversity penalty (near-duplicates that are genuinely relevant
    survive). Pure CPU-bound; no LLM call.
    """
    if not candidates:
        return []
    sorted_results = sorted(candidates, key=lambda r: r.score, reverse=True)
    selected: list[RetrievalResult] = []
    tokens_used = 0
    for r in sorted_results:
        t = estimate_tokens(r.memory.content)
        if tokens_used + t > max_tokens:
            continue
        selected.append(r)
        tokens_used += t
    return selected


def _step6_pack(
    candidates: list[RetrievalResult],
    *,
    token_budget: int,
    long_context_mode: bool = False,
    long_context_max_tokens: int = _DEFAULT_LONG_CONTEXT_MAX_TOKENS,
) -> list[RetrievalResult]:
    """Step 6 of the recall cascade — token-fit packing.

    Two strategies:

      Standard (``long_context_mode=False``)
        Greedy fit-to-budget — optimized for short-context answerers
        (gpt-4o, claude-haiku) where every token has measurable cost.

      Long-context (``long_context_mode=True``)
        Pack the top-scored candidates verbatim up to
        ``long_context_max_tokens``. Designed for 1M-context answerers
        where the diversity penalty actively HURTS (cuts genuinely-
        relevant near-duplicates).

    Default behavior (``long_context_mode=False``) is byte-identical
    to the pre-feature pipeline.
    """
    if long_context_mode:
        return _long_context_pack(
            candidates, max_tokens=long_context_max_tokens,
        )
    return fit_to_budget(candidates, token_budget)


class _OrchestratorPostProcessMixin:
    """Steps 2-6 shared between sync recall and async recall."""

    def _step3p5_rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Cross-encoder rerank between RRF and graph BFS.

        Wraps each candidate dict's ``memory`` in a ``RetrievalResult``
        seeded with its current ``vector_sim``, calls ``rerank()``, then
        rebuilds the dict list with the rerank score promoted into
        ``vector_sim``. Skipped (identity) when reranker_cfg is None,
        disabled, or when no candidates were produced.
        """
        from attestor import trace as _tr

        cfg = getattr(self, "reranker_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False) or not candidates:
            return candidates

        # Lazy import to avoid forcing torch/sentence-transformers on
        # every recall — the module itself is light, but follow the
        # codebase's convention of deferring optional-feature imports
        # so error paths in unrelated tests stay clean.
        from attestor.models import RetrievalResult
        from attestor.retrieval.reranker import rerank as _rerank_fn

        # Materialize a typed result list for the reranker; the
        # rerank() helper never mutates its input.
        wrapped: list[RetrievalResult] = []
        for c in candidates:
            wrapped.append(
                RetrievalResult(
                    memory=c["memory"],
                    score=float(c["vector_sim"]),
                    match_source="vector",
                )
            )

        # Cached provider singleton on the orchestrator — tests can
        # inject ``orch.reranker = stub`` directly to bypass the
        # build_reranker() factory and skip model loading.
        provider = getattr(self, "reranker", None)
        reranked = _rerank_fn(query, wrapped, cfg=cfg, provider=provider)
        if reranked is wrapped:
            return candidates  # rerank() short-circuited (disabled / no-op)

        # Rebuild dicts, joining on memory.id so original distance/etc.
        # is preserved.
        by_id: dict[str, dict] = {c["memory"].id: c for c in candidates}
        out: list[dict] = []
        for r in reranked:
            existing = by_id.get(r.memory.id)
            if existing is None:
                continue
            out.append({
                "memory": r.memory,
                "distance": existing["distance"],
                "vector_sim": float(r.score),
            })

        if _tr.is_enabled():
            _tr.event(
                "recall.stage.rerank",
                provider=getattr(cfg, "provider", "?"),
                top_k=getattr(cfg, "top_k", 0),
                top_n_input=getattr(cfg, "top_n_input", 0),
                in_count=len(candidates),
                out_count=len(out),
            )
        return out

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
        long_context: bool = False,
        long_context_max_tokens: int = _DEFAULT_LONG_CONTEXT_MAX_TOKENS,
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

        # ── Step 3.5: Cross-encoder rerank (opt-in via YAML) ──
        # Reranks candidate dicts by their `vector_sim` field — the
        # rerank score replaces vector_sim so the downstream graph
        # narrow + score blend pick up the new signal. Skipped when
        # reranker_cfg is None / disabled (default behavior unchanged).
        candidates = self._step3p5_rerank(query, candidates)

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
        # Layer-aware tiebreaker (~0.05 semantic > episodic). Tunable via
        # ``self.layer_weights`` (None = scorer default, empty dict = no-op).
        results = layer_boost(
            results, weights=getattr(self, "layer_weights", None),
        )

        # ── Step 4: MMR diversity ──
        # Skipped when long_context=True — the downstream 1M-context
        # answerer benefits more from raw top-K than from a diversity-
        # trimmed subset (near-duplicates that are genuinely relevant
        # survive). MMR + long-context are mutually exclusive by design.
        if self.enable_mmr and not long_context:
            _pre_mmr_count = len(results)
            results = mmr_rerank(results, lambda_param=self.mmr_lambda)
            if self.mmr_top_n is not None and len(results) > self.mmr_top_n:
                results = results[: self.mmr_top_n]
            if _tr.is_enabled():
                _tr.event("recall.stage.mmr",
                          lambda_=self.mmr_lambda,
                          mmr_top_n=self.mmr_top_n,
                          in_count=_pre_mmr_count, out_count=len(results))
        elif long_context and _tr.is_enabled():
            _tr.event("recall.stage.mmr_skipped",
                      reason="long_context_mode")

        # ── Step 5: Confidence decay ──
        results = confidence_decay_boost(
            results,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )

        # ── Step 6: Fit to budget (or long-context pack) ──
        _pre_pack_count = len(results)
        final = _step6_pack(
            results,
            token_budget=token_budget,
            long_context_mode=long_context,
            long_context_max_tokens=long_context_max_tokens,
        )
        if _tr.is_enabled():
            _tr.event("recall.stage.pack",
                      mode="long_context" if long_context else "greedy_fit",
                      token_budget=token_budget,
                      long_context_max_tokens=(
                          long_context_max_tokens if long_context else None
                      ),
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
