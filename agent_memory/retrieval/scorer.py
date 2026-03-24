"""Scoring, deduplication, temporal boost, entity boost, token budget assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from agent_memory.models import RetrievalResult
from agent_memory.utils.tokens import estimate_tokens


def deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Remove duplicate memories, keeping the highest-scored entry."""
    seen: dict[str, RetrievalResult] = {}
    for r in results:
        mid = r.memory.id
        if mid not in seen or r.score > seen[mid].score:
            seen[mid] = r
    return list(seen.values())


def confidence_decay_boost(
    results: List[RetrievalResult],
    decay_rate: float = 0.001,
    boost_rate: float = 0.03,
    gate: float = 0.0,
) -> List[RetrievalResult]:
    """Apply time-based decay and access-based boost to scores.

    - decay_rate: confidence lost per hour since last access (default 0.001 = floor in ~41 days)
    - boost_rate: confidence gained per access (default 0.03)
    - gate: minimum confidence to keep in results (0.0 = no filtering)

    Modifies the RetrievalResult.score, not the stored Memory.confidence.
    """
    now = datetime.now(timezone.utc)
    out: List[RetrievalResult] = []
    for r in results:
        # Determine reference time for decay
        ref_time_str = r.memory.last_accessed or r.memory.created_at
        try:
            ref_time = datetime.fromisoformat(ref_time_str)
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)
            hours_since = max(0.0, (now - ref_time).total_seconds() / 3600.0)
        except (ValueError, TypeError):
            hours_since = 0.0

        access_count = r.memory.access_count or 0

        # Compute adjusted confidence
        conf = r.memory.confidence
        conf -= decay_rate * hours_since
        conf += boost_rate * access_count
        conf = max(0.1, min(1.0, conf))  # clamp to [0.1, 1.0]

        if conf < gate:
            continue

        # Apply confidence as a multiplier on the existing score
        out.append(
            RetrievalResult(
                memory=r.memory,
                score=r.score * conf,
                match_source=r.match_source,
            )
        )
    return out


def temporal_boost(
    results: List[RetrievalResult],
    decay_days: int = 90,
    enabled: bool = True,
) -> List[RetrievalResult]:
    """Boost scores for more recent memories.

    Can be disabled for benchmark data spanning long time periods.
    """
    if not enabled:
        return results
    now = datetime.now(timezone.utc)
    for r in results:
        try:
            created = datetime.fromisoformat(r.memory.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).days
            # Linear decay: full boost at 0 days, no boost after decay_days
            boost = max(0.0, 1.0 - (age_days / decay_days)) * 0.2
            r.score += boost
        except (ValueError, TypeError):
            pass
    return results


def entity_boost(
    results: List[RetrievalResult], query_entities: Optional[List[str]] = None
) -> List[RetrievalResult]:
    """Boost score if memory entity or content matches query entities.

    Checks both the entity field (strong match) and content substring (weaker match).
    """
    if not query_entities:
        return results
    query_entities_lower = {e.lower() for e in query_entities}
    for r in results:
        # Strong boost: entity field exact match
        if r.memory.entity and r.memory.entity.lower() in query_entities_lower:
            r.score += 0.3
        # Weaker boost: entity name appears in content
        else:
            content_lower = r.memory.content.lower()
            for ent in query_entities_lower:
                if len(ent) >= 3 and ent in content_lower:
                    r.score += 0.15
                    break
    return results


def pagerank_boost(
    results: List[RetrievalResult],
    pagerank_scores: Dict[str, float],
    weight: float = 0.3,
) -> List[RetrievalResult]:
    """Boost scores based on PageRank centrality of memory entities.

    Memories about high-PageRank entities are more structurally important.
    """
    if not pagerank_scores:
        return results
    out: List[RetrievalResult] = []
    for r in results:
        pr = 0.0
        if r.memory.entity:
            pr = pagerank_scores.get(r.memory.entity.lower(), 0.0)
        out.append(
            RetrievalResult(
                memory=r.memory,
                score=r.score + weight * pr,
                match_source=r.match_source,
            )
        )
    return out


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def mmr_rerank(
    results: List[RetrievalResult],
    lambda_param: float = 0.7,
    max_results: int = 10,
) -> List[RetrievalResult]:
    """Maximal Marginal Relevance reranking for diverse retrieval.

    Balances relevance (lambda) vs diversity (1 - lambda).
    Uses Jaccard word-overlap similarity to penalize near-duplicates.
    """
    if len(results) <= 1:
        return results

    # Sort by score descending as candidates
    candidates = sorted(results, key=lambda r: r.score, reverse=True)
    selected: List[RetrievalResult] = [candidates[0]]
    remaining = candidates[1:]

    while remaining and len(selected) < max_results:
        best_mmr = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Max similarity to any already-selected result
            max_sim = max(
                _jaccard_similarity(candidate.memory.content, s.memory.content)
                for s in selected
            )
            mmr = lambda_param * candidate.score - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def fit_to_budget(
    results: List[RetrievalResult], token_budget: int
) -> List[RetrievalResult]:
    """Select highest-scored results that fit within token budget."""
    # Sort by score descending
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    selected = []
    tokens_used = 0
    for r in sorted_results:
        t = estimate_tokens(r.memory.content)
        if tokens_used + t <= token_budget:
            selected.append(r)
            tokens_used += t
        elif not selected:
            # Always include at least one result
            selected.append(r)
            break
    return selected
