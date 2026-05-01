"""Scoring, deduplication, temporal boost, entity boost, token budget assembly.

All boost functions are PURE: each returns a new list of new RetrievalResult
instances. The input list and its members are never mutated. This keeps
chains of boosters reasoning-friendly — no hidden side effects.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone

from attestor.models import RetrievalResult
from attestor.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Remove duplicate memories, keeping the highest-scored entry."""
    seen: dict[str, RetrievalResult] = {}
    for r in results:
        mid = r.memory.id
        if mid not in seen or r.score > seen[mid].score:
            seen[mid] = r
    return list(seen.values())


def confidence_decay_boost(
    results: list[RetrievalResult],
    decay_rate: float = 0.001,
    boost_rate: float = 0.03,
    gate: float = 0.0,
) -> list[RetrievalResult]:
    """Apply time-based decay and access-based boost to scores.

    - decay_rate: confidence lost per hour since last access (default 0.001 = floor in ~41 days)
    - boost_rate: confidence gained per access (default 0.03)
    - gate: minimum confidence to keep in results (0.0 = no filtering)

    Returns a NEW list of NEW RetrievalResult instances; never raises on
    malformed timestamps (logged at DEBUG level instead).
    """
    now = datetime.now(timezone.utc)
    out: list[RetrievalResult] = []
    for r in results:
        # Determine reference time for decay
        ref_time_str = r.memory.last_accessed or r.memory.created_at
        try:
            ref_time = datetime.fromisoformat(ref_time_str)
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)
            hours_since = max(0.0, (now - ref_time).total_seconds() / 3600.0)
        except (ValueError, TypeError) as e:
            logger.debug(
                "confidence_decay: malformed ref_time on memory %s: %s",
                r.memory.id,
                e,
            )
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
        out.append(replace(r, score=r.score * conf))
    return out


def temporal_boost(
    results: list[RetrievalResult],
    decay_days: int = 90,
    enabled: bool = True,
) -> list[RetrievalResult]:
    """Boost scores for more recent memories.

    Can be disabled for benchmark data spanning long time periods. Returns a
    NEW list of NEW RetrievalResult instances (or the input list unchanged
    when disabled — callers should still treat it as immutable).
    """
    if not enabled:
        return list(results)
    now = datetime.now(timezone.utc)
    out: list[RetrievalResult] = []
    for r in results:
        boost = 0.0
        try:
            created = datetime.fromisoformat(r.memory.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).days
            # Linear decay: full boost at 0 days, no boost after decay_days
            boost = max(0.0, 1.0 - (age_days / decay_days)) * 0.2
        except (ValueError, TypeError) as e:
            logger.debug(
                "temporal_boost: malformed created_at on memory %s: %s",
                r.memory.id,
                e,
            )
        out.append(replace(r, score=r.score + boost))
    return out


def entity_boost(
    results: list[RetrievalResult],
    query_entities: list[str] | None = None,
) -> list[RetrievalResult]:
    """Boost score if memory entity or content matches query entities.

    Checks both the entity field (strong match) and content substring (weaker
    match). Returns a NEW list of NEW RetrievalResult instances.
    """
    if not query_entities:
        return list(results)
    query_entities_lower = {e.lower() for e in query_entities}
    out: list[RetrievalResult] = []
    for r in results:
        delta = 0.0
        # Strong boost: entity field exact match
        if r.memory.entity and r.memory.entity.lower() in query_entities_lower:
            delta = 0.3
        else:
            # Weaker boost: entity name appears in content
            content_lower = r.memory.content.lower()
            for ent in query_entities_lower:
                if len(ent) >= 3 and ent in content_lower:
                    delta = 0.15
                    break
        out.append(replace(r, score=r.score + delta))
    return out


def pagerank_boost(
    results: list[RetrievalResult],
    pagerank_scores: dict[str, float],
    weight: float = 0.3,
) -> list[RetrievalResult]:
    """Boost scores based on PageRank centrality of memory entities.

    Memories about high-PageRank entities are more structurally important.
    Returns a NEW list of NEW RetrievalResult instances.
    """
    if not pagerank_scores:
        return list(results)
    out: list[RetrievalResult] = []
    for r in results:
        pr = 0.0
        if r.memory.entity:
            pr = pagerank_scores.get(r.memory.entity.lower(), 0.0)
        out.append(replace(r, score=r.score + weight * pr))
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
    results: list[RetrievalResult],
    lambda_param: float = 0.7,
    max_results: int = 10,
) -> list[RetrievalResult]:
    """Maximal Marginal Relevance reranking for diverse retrieval.

    Balances relevance (lambda) vs diversity (1 - lambda). Uses Jaccard
    word-overlap similarity to penalize near-duplicates. Returns a new list
    (the existing RetrievalResult instances are reused — no score mutation,
    so the original list's order is the only thing that changes).
    """
    if len(results) <= 1:
        return list(results)

    # Sort by score descending as candidates
    candidates = sorted(results, key=lambda r: r.score, reverse=True)
    selected: list[RetrievalResult] = [candidates[0]]
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
    results: list[RetrievalResult], token_budget: int
) -> list[RetrievalResult]:
    """Select highest-scored results that fit within token budget."""
    # Sort by score descending
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    selected: list[RetrievalResult] = []
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
