"""3-layer retrieval cascade orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional

from agent_memory.models import Memory, RetrievalResult
from agent_memory.retrieval.scorer import (
    deduplicate,
    entity_boost,
    fit_to_budget,
    temporal_boost,
)
from agent_memory.retrieval.tag_matcher import extract_tags
from agent_memory.store.sqlite_store import SQLiteStore


class RetrievalOrchestrator:
    """Orchestrates the 3-layer retrieval cascade."""

    def __init__(
        self,
        store: SQLiteStore,
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

    def recall(
        self,
        query: str,
        token_budget: int = 2000,
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
            tag_memories = self.store.tag_search(tags)
            for mem in tag_memories:
                results.append(
                    RetrievalResult(memory=mem, score=1.0, match_source="tag")
                )

        # Layer 2: Entity-field search for graph-connected entities
        if self.graph and expanded_queries:
            seen_ids = {r.memory.id for r in results}
            for entity_name in expanded_queries[:8]:
                entity_memories = self.store.list_memories(
                    entity=entity_name, status="active", limit=5,
                )
                for mem in entity_memories:
                    if mem.id not in seen_ids:
                        results.append(
                            RetrievalResult(
                                memory=mem, score=0.5, match_source="graph",
                            )
                        )
                        seen_ids.add(mem.id)

        # TODO: rewire in Plan 03 -- vector similarity layer (ChromaDB)

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
        results = self._reciprocal_rank_fusion(results)
        results = temporal_boost(results, enabled=self.enable_temporal_boost)

        # Extract entities from query for entity boost
        # Include proper nouns and any multi-char meaningful words
        query_entities = [t for t in tags if t[0].isupper()] if tags else None
        results = entity_boost(results, query_entities)

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

    def recall_as_context(self, query: str, token_budget: int = 2000) -> str:
        """Recall and format as a context string for prompt injection."""
        results = self.recall(query, token_budget)
        if not results:
            return ""

        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return "\n".join(lines)
