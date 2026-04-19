"""3-layer retrieval cascade orchestrator."""

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
    pagerank_boost,
    temporal_boost,
)
from attestor.retrieval.tag_matcher import extract_tags
from attestor.store.base import DocumentStore


class RetrievalOrchestrator:
    """Orchestrates the 3-layer retrieval cascade."""

    def __init__(
        self,
        store: DocumentStore,
        min_results: int = 3,
        vector_store=None,
        graph=None,
        enable_temporal_boost: bool = True,
    ):
        self.store = store
        self.min_results = min_results
        self.vector_store = vector_store
        self.graph = graph
        self.enable_temporal_boost = enable_temporal_boost
        self.confidence_gate: float = 0.0
        self.confidence_decay_rate: float = 0.001
        self.confidence_boost_rate: float = 0.03
        self.enable_mmr: bool = True
        self.mmr_lambda: float = 0.7
        self.fusion_mode: str = "rrf"  # "rrf" or "graph_blend"
        self.graph_blend_vector_weight: float = 0.7
        self.graph_blend_graph_weight: float = 0.3

    def recall(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Execute the retrieval cascade: Tags → Graph → Vectors."""
        results: List[RetrievalResult] = []

        # Layer 0: Graph expansion (find related entities via graph traversal)
        expanded_queries = []
        graph_context_triples: List[str] = []
        if self.graph:
            try:
                tags = extract_tags(query)
                for tag in tags:
                    # Get related entities at depth=2 for multi-hop
                    related = self.graph.get_related(tag, depth=2)
                    expanded_queries.extend(related)

                    # Get typed edges for relationship context
                    if hasattr(self.graph, "get_edges"):
                        edges = self.graph.get_edges(tag)
                        for edge in edges:
                            subj = edge.get("subject", "")
                            pred = edge.get("predicate", "related_to")
                            obj = edge.get("object", "")
                            event_date = edge.get("event_date", "")
                            date_str = f" ({event_date})" if event_date else ""
                            graph_context_triples.append(
                                f"{subj} {pred} {obj}{date_str}"
                            )
            except Exception:
                pass

        # Layer 1: Tag match
        tags = extract_tags(query)
        if tags:
            tag_memories = self.store.tag_search(tags, namespace=namespace)
            for mem in tag_memories:
                results.append(
                    RetrievalResult(memory=mem, score=1.0, match_source="tag")
                )

        # Layer 2: Entity-field search for graph-connected entities
        if self.graph and expanded_queries:
            seen_ids = {r.memory.id for r in results}
            for entity_name in expanded_queries[:8]:
                entity_memories = self.store.list_memories(
                    entity=entity_name, status="active",
                    namespace=namespace, limit=5,
                )
                for mem in entity_memories:
                    if mem.id not in seen_ids:
                        results.append(
                            RetrievalResult(
                                memory=mem, score=0.5, match_source="graph",
                            )
                        )
                        seen_ids.add(mem.id)

        # Layer 3: Vector similarity (pgvector / Arango / other VectorStore)
        if self.vector_store:
            try:
                vec_results = self.vector_store.search(
                    query, limit=20, namespace=namespace
                )
                seen_ids = {r.memory.id for r in results}
                for vr in vec_results:
                    mid = vr["memory_id"]
                    if mid in seen_ids:
                        continue
                    memory = self.store.get(mid)
                    if not memory or memory.status != "active":
                        continue
                    if namespace and memory.namespace != namespace:
                        continue
                    distance = vr.get("distance", 1.0)
                    score = max(0.0, 1.0 - distance)
                    results.append(
                        RetrievalResult(
                            memory=memory, score=score, match_source="vector",
                        )
                    )
                    seen_ids.add(mid)
            except Exception:
                pass

        # Layer 4: Inject graph relationship triples as synthetic memories
        # These give the LLM structured relationship data for multi-hop reasoning
        if graph_context_triples:
            for triple_str in graph_context_triples[:20]:
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

        # Score, fuse, and assemble
        if self.fusion_mode == "graph_blend" and self.graph and len(results) >= 3:
            results = self._graph_blend(results)
        else:
            results = self._reciprocal_rank_fusion(results)
        results = temporal_boost(results, enabled=self.enable_temporal_boost)

        # Extract entities from query for entity boost
        # Include proper nouns and any multi-char meaningful words
        query_entities = [t for t in tags if t[0].isupper()] if tags else None
        results = entity_boost(results, query_entities)

        # MMR diversity reranking
        if self.enable_mmr:
            results = mmr_rerank(results, lambda_param=self.mmr_lambda)

        # Confidence decay/boost
        results = confidence_decay_boost(
            results,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )

        return fit_to_budget(results, token_budget)

    def _reciprocal_rank_fusion(
        self, results: List[RetrievalResult], k: int = 60
    ) -> List[RetrievalResult]:
        """Combine multi-source results using Reciprocal Rank Fusion.

        RRF score = sum over sources of: 1 / (k + rank_in_source)

        Memories found by multiple retrieval channels (vector + graph)
        get boosted scores. Falls back to simple deduplication when only one
        source is present.
        """
        if not results:
            return results

        # Group results by source
        by_source: Dict[str, List[RetrievalResult]] = {}
        for r in results:
            by_source.setdefault(r.match_source, []).append(r)

        # If only one source, just deduplicate
        if len(by_source) == 1:
            return deduplicate(results)

        # Sort each source by score descending to assign ranks
        for source in by_source:
            by_source[source].sort(key=lambda r: r.score, reverse=True)

        # Compute RRF score per memory
        rrf_scores: Dict[str, float] = {}
        best_result: Dict[str, RetrievalResult] = {}

        for source, source_results in by_source.items():
            for rank, r in enumerate(source_results):
                mid = r.memory.id
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
                if mid not in best_result or r.score > best_result[mid].score:
                    best_result[mid] = r

        # Build final results with RRF scores
        fused = []
        for mid, rrf_score in rrf_scores.items():
            r = best_result[mid]
            fused.append(
                RetrievalResult(
                    memory=r.memory,
                    score=rrf_score,
                    match_source=r.match_source,
                )
            )

        return fused

    def _graph_blend(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Weighted blend of vector similarity and PageRank scores.

        Alternative to RRF that preserves score magnitudes. Falls back to RRF
        when result set < 3 (normalization unstable with too few points).
        """
        results = deduplicate(results)
        if not results:
            return results

        # Get PageRank scores
        pr_scores: Dict[str, float] = {}
        if self.graph and hasattr(self.graph, "pagerank"):
            try:
                pr_scores = self.graph.pagerank()
            except Exception:
                pass

        if not pr_scores:
            return self._reciprocal_rank_fusion(results)

        # Min-max normalize vector scores
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0

        # Collect PR values for normalization
        pr_values = []
        for r in results:
            entity_key = r.memory.entity.lower() if r.memory.entity else ""
            pr_values.append(pr_scores.get(entity_key, 0.0))
        min_pr, max_pr = min(pr_values), max(pr_values)
        pr_range = max_pr - min_pr if max_pr > min_pr else 1.0

        blended = []
        for r, pr_val in zip(results, pr_values):
            norm_score = (r.score - min_s) / score_range
            norm_pr = (pr_val - min_pr) / pr_range
            final = (
                self.graph_blend_vector_weight * norm_score
                + self.graph_blend_graph_weight * norm_pr
            )
            blended.append(
                RetrievalResult(
                    memory=r.memory, score=final, match_source=r.match_source,
                )
            )

        blended.sort(key=lambda r: r.score, reverse=True)
        return blended

    def recall_debug(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
    ) -> Dict:
        """Execute recall and return per-layer debug trace.

        Returns dict with keys: query, layers (list of layer dicts),
        final (list of result dicts), config.
        """
        layers = []

        def _snap(results: List[RetrievalResult]) -> List[Dict]:
            return [
                {
                    "id": r.memory.id,
                    "content": r.memory.content[:200],
                    "score": round(r.score, 4),
                    "source": r.match_source,
                    "entity": r.memory.entity,
                    "category": r.memory.category,
                }
                for r in results
            ]

        all_results: List[RetrievalResult] = []
        t_total = time.monotonic()

        # Layer 1: Tag match
        t_layer = time.monotonic()
        tags = extract_tags(query)
        tag_results: List[RetrievalResult] = []
        if tags:
            tag_memories = self.store.tag_search(tags, namespace=namespace)
            for mem in tag_memories:
                tag_results.append(
                    RetrievalResult(memory=mem, score=1.0, match_source="tag")
                )
        all_results.extend(tag_results)
        layers.append({
            "name": "Tag Match",
            "description": "FTS on extracted tags",
            "tags": tags,
            "count": len(tag_results),
            "latency_ms": round((time.monotonic() - t_layer) * 1000, 2),
            "results": _snap(tag_results),
        })

        # Layer 2: Graph expansion
        t_layer = time.monotonic()
        graph_results: List[RetrievalResult] = []
        expanded_queries = []
        graph_context_triples: List[str] = []
        if self.graph:
            try:
                for tag in tags:
                    related = self.graph.get_related(tag, depth=2)
                    expanded_queries.extend(related)
                    if hasattr(self.graph, "get_edges"):
                        edges = self.graph.get_edges(tag)
                        for edge in edges:
                            subj = edge.get("subject", "")
                            pred = edge.get("predicate", "related_to")
                            obj = edge.get("object", "")
                            event_date = edge.get("event_date", "")
                            date_str = f" ({event_date})" if event_date else ""
                            graph_context_triples.append(
                                f"{subj} {pred} {obj}{date_str}"
                            )
            except Exception:
                pass

            seen_ids = {r.memory.id for r in all_results}
            for entity_name in expanded_queries[:8]:
                entity_memories = self.store.list_memories(
                    entity=entity_name, status="active",
                    namespace=namespace, limit=5,
                )
                for mem in entity_memories:
                    if mem.id not in seen_ids:
                        r = RetrievalResult(
                            memory=mem, score=0.5, match_source="graph",
                        )
                        graph_results.append(r)
                        all_results.append(r)
                        seen_ids.add(mem.id)

            # Graph triples as synthetic memories
            for triple_str in graph_context_triples[:20]:
                r = RetrievalResult(
                    memory=Memory(
                        content=triple_str,
                        category="graph_relation",
                        tags=[],
                    ),
                    score=0.6,
                    match_source="graph",
                )
                graph_results.append(r)
                all_results.append(r)

        layers.append({
            "name": "Graph Expansion",
            "description": "BFS depth=2, entity lookup, relation triples",
            "expanded_entities": expanded_queries[:8],
            "triples": len(graph_context_triples),
            "count": len(graph_results),
            "latency_ms": round((time.monotonic() - t_layer) * 1000, 2),
            "results": _snap(graph_results),
        })

        # Layer 3: Vector search
        t_layer = time.monotonic()
        vector_results: List[RetrievalResult] = []
        if self.vector_store:
            try:
                vec_results = self.vector_store.search(
                    query, limit=20, namespace=namespace
                )
                seen_ids = {r.memory.id for r in all_results}
                for vr in vec_results:
                    mid = vr["memory_id"]
                    if mid in seen_ids:
                        continue
                    memory = self.store.get(mid)
                    if not memory or memory.status != "active":
                        continue
                    if namespace and memory.namespace != namespace:
                        continue
                    distance = vr.get("distance", 1.0)
                    score = max(0.0, 1.0 - distance)
                    r = RetrievalResult(
                        memory=memory, score=score, match_source="vector",
                    )
                    vector_results.append(r)
                    all_results.append(r)
                    seen_ids.add(mid)
            except Exception:
                pass

        layers.append({
            "name": "Vector Search",
            "description": "Cosine similarity via vector backend",
            "count": len(vector_results),
            "latency_ms": round((time.monotonic() - t_layer) * 1000, 2),
            "results": _snap(vector_results),
        })

        # Layer 4: Fusion + Rank (RRF or graph blend)
        t_layer = time.monotonic()
        if self.fusion_mode == "graph_blend" and self.graph and len(all_results) >= 3:
            fused = self._graph_blend(all_results)
        else:
            fused = self._reciprocal_rank_fusion(all_results)
        fused = temporal_boost(fused, enabled=self.enable_temporal_boost)
        query_entities = [t for t in tags if t[0].isupper()] if tags else None
        fused = entity_boost(fused, query_entities)

        layers.append({
            "name": "Fusion + Rank",
            "description": f"RRF k=60, temporal boost, entity boost" if self.fusion_mode == "rrf" else "Graph blend + temporal + entity boost",
            "fusion_mode": self.fusion_mode,
            "count": len(fused),
            "latency_ms": round((time.monotonic() - t_layer) * 1000, 2),
            "results": _snap(fused),
        })

        # Layer 5: Diversity + Fit (MMR + token budget)
        t_layer = time.monotonic()
        if self.enable_mmr:
            fused = mmr_rerank(fused, lambda_param=self.mmr_lambda)
        fused = confidence_decay_boost(
            fused,
            decay_rate=self.confidence_decay_rate,
            boost_rate=self.confidence_boost_rate,
            gate=self.confidence_gate,
        )
        final = fit_to_budget(fused, token_budget)

        layers.append({
            "name": "Diversity + Fit",
            "description": f"MMR λ={self.mmr_lambda}, confidence decay, token budget={token_budget}",
            "mmr_enabled": self.enable_mmr,
            "budget": token_budget,
            "count": len(final),
            "latency_ms": round((time.monotonic() - t_layer) * 1000, 2),
            "results": _snap(final),
        })

        total_ms = round((time.monotonic() - t_total) * 1000, 2)

        return {
            "query": query,
            "namespace": namespace,
            "token_budget": token_budget,
            "total_latency_ms": total_ms,
            "layers": layers,
            "final_count": len(final),
            "config": {
                "fusion_mode": self.fusion_mode,
                "mmr_lambda": self.mmr_lambda,
                "confidence_gate": self.confidence_gate,
                "confidence_decay_rate": self.confidence_decay_rate,
            },
        }

    def recall_as_context(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: Optional[str] = None,
    ) -> str:
        """Recall and format as a context string for prompt injection."""
        results = self.recall(query, token_budget, namespace=namespace)
        if not results:
            return ""

        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return "\n".join(lines)
