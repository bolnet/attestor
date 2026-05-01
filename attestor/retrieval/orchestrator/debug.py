"""Debug-tracing variant of recall — kept on a mixin so it stays close
to the orchestrator's state without inflating ``core.py``.

``recall_debug`` mirrors the JSONL trace as a returnable Python dict,
intended for ad-hoc UI inspection. Behavior is byte-identical to the
pre-split monolith.
"""

from __future__ import annotations

import time

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.scorer import (
    confidence_decay_boost,
    deduplicate,
    entity_boost,
    fit_to_budget,
    mmr_rerank,
    temporal_boost,
)


class _OrchestratorDebugMixin:
    """Per-step trace variant for debugging."""

    def recall_debug(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: str | None = None,
    ) -> dict:
        """Per-step trace for UI / ad-hoc inspection (mirrors the JSONL trace)."""
        t_total = time.monotonic()
        question_entities = self._question_entities(query)
        layers: list[dict] = []

        # Vector
        t = time.monotonic()
        vector_hits_raw: list[dict] = []
        candidates: list[dict] = []
        vector_error: str | None = None
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
        vector_layer: dict = {
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
        results: list[RetrievalResult] = []
        narrowed: list[dict] = []
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
        _gab = self.config.graph_affinity_bonus
        layers.append({
            "name": "Graph Narrow",
            "description": (
                f"Blend {self.config.vector_weight}·vector + "
                f"{self.config.graph_weight}·graph; "
                f"hops 0→+{_gab.get(0, 0.0)}, "
                f"1→+{_gab.get(1, 0.0)}, "
                f"2→+{_gab.get(2, 0.0)}, "
                f"unreachable→{self.config.graph_unreachable_penalty}"
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

        warnings: list[str] = []
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
                "vector_weight": self.config.vector_weight,
                "graph_weight": self.config.graph_weight,
                "mmr_lambda": self.mmr_lambda,
                "confidence_gate": self.confidence_gate,
            },
        }
