"""Multi-hop / global-query lane (GraphRAG-style).

Bypasses the deterministic 6-step recall cascade for queries that ask
for a synthesis spanning many memories ("summarize my interactions
with X over 6 months", "what are my recurring concerns?", "what trips
have I taken across 8 months?"). Microsoft GraphRAG, HippoRAG, and
Anthropic's contextual retrieval all converge on the same playbook:

    classify → community-detect → cluster-summarize → return summaries

This module implements the playbook on top of Attestor's existing
Postgres + Pinecone + Neo4j stack:

    1. Classify the question as ``local`` or ``global``. Cheap regex
       heuristic first; optional LLM tiebreaker only when ambiguous.
    2. Pull candidate entities for the question from the user's Neo4j
       subgraph (tenancy-scoped via ``namespace`` + ``user_id``).
    3. Run Leiden community detection on the subgraph (via
       ``gds.leiden.stream``) — clusters are dense local groups of
       entities related by typed edges.
    4. For each top community, pull the underlying memories from the
       document store and ask the summarizer LLM for a cluster-level
       summary grounded in real facts.
    5. Return the summaries as ``RetrievalResult`` objects with
       ``category="global_summary"`` and ``_cluster_id`` /
       ``_source_entities`` metadata so callers can tell them apart
       from local hits.

Design constraints:
  - Disabled by default (``GlobalQueryCfg.enabled=False``) so the
    default codepath stays byte-identical to the 6-step cascade.
  - Failure-isolated. Graph store down OR LLM error → return empty
    result + log warning, never crash recall.
  - Bounded LLM calls. 1 classifier (optional) + N ≤ ``max_clusters``
    summaries. Cost surfaced via
    ``GlobalQueryResult.cost_estimate_usd``.
  - Reuses ``Neo4jBackend`` and ``PostgresBackend`` interfaces — no
    raw Cypher leaks into the orchestrator or AgentMemory layer.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from attestor.models import Memory, RetrievalResult

logger = logging.getLogger("attestor.retrieval.global_query")


# ── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class GlobalQueryResult:
    """Outcome of one ``run_global_query`` call.

    Frozen so the caller can pass it down recall paths without worrying
    about mutation. ``summaries`` and ``cluster_entities`` are tuples
    (not lists) for the same reason — frozen dataclasses must hold
    immutable values to be safely shareable across threads.
    """

    summaries: tuple[str, ...]
    cluster_entities: tuple[tuple[str, ...], ...]
    cluster_count: int
    elapsed_ms: float
    llm_model: str
    cost_estimate_usd: float


# ── Classifier ────────────────────────────────────────────────────────


# Cheap regex cues that almost always indicate a global query. Keeping
# the list small + curated avoids false positives ("over the moon" /
# "across the street" / "patterns of behavior in chess"). Unlike the
# answerer-side prompts, this fires on every recall call so it must be
# fast. ~200ns per match on the regex; no LLM unless ambiguous.
_GLOBAL_HEURISTIC_PATTERNS = (
    r"\bsummari[sz]e\b",
    r"\bsummary\b",
    r"\bover (the )?(?:last |past )?\d+\s*(?:month|week|day|year|quarter)s?\b",
    r"\bover time\b",
    r"\bacross (?:the )?(?:last |past )?\d+\s*(?:month|week|day|year|quarter)s?\b",
    r"\bacross (?:all|my|the)\b",
    r"\brecurring\b",
    r"\bthemes?\b",
    r"\bpatterns?\b",
    r"\boverall\b",
    r"\boverview\b",
    r"\bwhat trips\b",
    r"\bwhat (?:are|were) my\b.*\b(?:concerns?|topics?|interests?)\b",
    r"\bthis (?:quarter|year|month)\b",
)
_GLOBAL_HEURISTIC_RE = re.compile(
    "|".join(_GLOBAL_HEURISTIC_PATTERNS), re.IGNORECASE,
)


# Local cues — when present, force False even if a global cue matched.
# Specific factoid forms ("what is my pre-approval amount") that look
# global because of the noun "amount" / "balance" but resolve to a
# single fact.
_LOCAL_OVERRIDE_PATTERNS = (
    r"\bwhat is my\b.*\bamount\b",
    r"\bwhat is my\b.*\bbalance\b",
    r"\bwhat is my\b.*\b(?:limit|cap|threshold)\b",
    r"\bhow much\b.*\bowe\b",
    r"\bpre[- ]approval\b",
    r"\bphone number\b",
    r"\baddress\b",
)
_LOCAL_OVERRIDE_RE = re.compile(
    "|".join(_LOCAL_OVERRIDE_PATTERNS), re.IGNORECASE,
)


def _classify_via_llm(query: str, *, model: str) -> str:
    """Single-shot LLM classifier for ambiguous queries.

    Returns ``"global"`` or ``"local"``. Defaults to ``"local"`` on any
    error so the lane never inflates calls under failure. Tests
    monkey-patch this symbol; production wires it through the standard
    ``llm_trace.get_client_for_model`` path.
    """
    try:
        from attestor.llm_trace import get_client_for_model, traced_create
    except Exception as e:  # noqa: BLE001
        logger.debug("global_query classifier LLM import failed: %s", e)
        return "local"
    try:
        client, clean_model = get_client_for_model(model)
        prompt = (
            "Classify the following user question as exactly one of "
            "two labels:\n"
            "  global — needs synthesis across many memories "
            "(e.g. summaries, themes, recurring patterns, time-spanning "
            "overviews)\n"
            "  local  — a specific factoid answerable from one or two "
            "memories\n\n"
            f"Question: {query}\n\n"
            "Reply with exactly one word: global OR local."
        )
        resp = traced_create(
            client,
            role="planner",
            model=clean_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        text = resp.choices[0].message.content.strip().lower()
        return "global" if text.startswith("global") else "local"
    except Exception as e:  # noqa: BLE001
        logger.debug("global_query classifier LLM call failed: %s", e)
        return "local"


def is_global_query(query: str, *, model: str | None = None) -> bool:
    """Return True iff ``query`` should route to the global lane.

    Cheap regex heuristic first — fires when one of the global cue
    phrases matches AND no local-override phrase matches. If neither
    side fires AND a ``model`` was supplied, defer to a single LLM
    classifier call. When ``model`` is None (cheapest mode), an
    unmatched query stays on the local pipeline.
    """
    if not query or not query.strip():
        return False
    if _LOCAL_OVERRIDE_RE.search(query):
        return False
    if _GLOBAL_HEURISTIC_RE.search(query):
        return True
    if model is None:
        return False
    verdict = _classify_via_llm(query, model=model)
    return verdict == "global"


# ── Summarizer protocol ────────────────────────────────────────────────


class _Summarizer(Protocol):
    def __call__(
        self, query: str, *, cluster_id: int,
        memories: list[Memory], model: str,
    ) -> str: ...


def _default_summarizer(
    query: str, *, cluster_id: int, memories: list[Memory], model: str,
) -> str:
    """Production summarizer — calls the configured LLM with the
    cluster's facts laid out as bullets.

    Failure-isolated: any exception returns a degraded ``""`` summary
    so the lane keeps producing non-fatal output. The caller filters
    empty summaries before returning ``RetrievalResult`` objects.
    """
    try:
        from attestor.llm_trace import get_client_for_model, traced_create
    except Exception as e:  # noqa: BLE001
        logger.warning("global_query summarizer LLM import failed: %s", e)
        return ""
    try:
        client, clean_model = get_client_for_model(model)
        bullets = "\n".join(f"- {m.content}" for m in memories[:50])
        prompt = (
            "You are summarizing one cluster of related memories for a "
            "user. Write a concise 1-3 sentence summary that directly "
            "addresses the user's question, grounded ONLY in the bullet "
            "points below. If the bullets do not contain enough "
            "information, write a single sentence saying so.\n\n"
            f"User question: {query}\n\n"
            f"Cluster {cluster_id} memories:\n{bullets}\n\n"
            "Summary:"
        )
        resp = traced_create(
            client,
            role="benchmark_default",
            model=clean_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "global_query summarizer LLM call failed for cluster %d: %s",
            cluster_id, e,
        )
        return ""


# ── Helpers ────────────────────────────────────────────────────────────


def _safe_call(
    fn_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any,
) -> Any:
    """Invoke ``fn`` and convert any exception into a sentinel.

    Returns the function's result on success or ``None`` on error,
    after logging at WARNING. Used for graph-store calls that we never
    want to surface up as a recall crash.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "global_query: %s failed (%s: %s)",
            fn_name, type(e).__name__, e,
        )
        return None


def _candidate_entities(
    graph_store: Any, query: str, *,
    namespace: str | None, user_id: str | None,
) -> list[str]:
    """Pull candidate entities matching the query from the graph store.

    The Neo4jBackend exposes ``get_entities_for_query`` (this PR adds
    it). When a backend doesn't implement that method we fall back to
    a tag/word-heuristic via ``get_entities`` so the lane still
    produces *something* in unit tests against minimal fakes.
    """
    if hasattr(graph_store, "get_entities_for_query"):
        out = _safe_call(
            "get_entities_for_query",
            graph_store.get_entities_for_query,
            query, namespace=namespace, user_id=user_id, limit=50,
        )
        return list(out or [])
    # Fallback path (backends without the new method).
    if hasattr(graph_store, "get_entities"):
        try:
            ents = graph_store.get_entities()
            return [e["name"] for e in ents if e.get("name")]
        except Exception:  # noqa: BLE001
            return []
    return []


def _leiden_communities(
    graph_store: Any, node_keys: list[str], *,
    namespace: str | None, user_id: str | None,
) -> dict[int, list[str]]:
    """Wrap ``leiden_communities``; return ``{}`` on error / missing."""
    if not node_keys:
        return {}
    if not hasattr(graph_store, "leiden_communities"):
        # Synthesize one community holding every entity so the lane
        # still produces a summary on backends without GDS Leiden.
        return {0: list(node_keys)}
    out = _safe_call(
        "leiden_communities",
        graph_store.leiden_communities,
        node_keys, namespace=namespace, user_id=user_id,
    )
    return out or {}


def _memories_for_entities(
    doc_store: Any, entities: list[str], *,
    namespace: str | None, user_id: str | None, limit: int,
) -> list[Memory]:
    """Fetch the underlying memories for a cluster's entities.

    Uses ``list_memories_for_entities`` when present; falls back to
    repeated ``list_memories`` calls otherwise so the lane works
    against bare PostgresBackend instances too.
    """
    if hasattr(doc_store, "list_memories_for_entities"):
        out = _safe_call(
            "list_memories_for_entities",
            doc_store.list_memories_for_entities,
            entities, namespace=namespace, user_id=user_id, limit=limit,
        )
        return list(out or [])
    if not hasattr(doc_store, "list_memories"):
        return []
    found: list[Memory] = []
    for ent in entities:
        try:
            mems = doc_store.list_memories(
                entity=ent, namespace=namespace, status="active",
                limit=limit,
            )
            found.extend(mems)
            if len(found) >= limit:
                break
        except Exception:  # noqa: BLE001
            continue
    return found[:limit]


# ── Public entry point ────────────────────────────────────────────────


def run_global_query(
    query: str,
    *,
    user_id: str | None,
    namespace: str | None = None,
    graph_store: Any,
    doc_store: Any,
    llm_model: str | None = None,
    max_clusters: int = 8,
    max_entities_per_cluster: int = 20,
    subgraph_depth: int = 2,
    cost_per_summary_usd: float = 0.005,
    summarizer: _Summarizer | None = None,
) -> GlobalQueryResult:
    """Run the global-query lane and return cluster-level summaries.

    Always returns a ``GlobalQueryResult`` — never raises. On failure
    the result has ``cluster_count=0`` and empty ``summaries`` so the
    caller can fall through to the local pipeline if it wants to.
    """
    t0 = time.monotonic()
    model = llm_model or "anthropic/claude-haiku-4.5"
    summary_fn: _Summarizer = summarizer or _default_summarizer

    def _result(
        summaries: tuple[str, ...],
        cluster_entities: tuple[tuple[str, ...], ...],
        cost: float,
    ) -> GlobalQueryResult:
        return GlobalQueryResult(
            summaries=summaries,
            cluster_entities=cluster_entities,
            cluster_count=len(summaries),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 2),
            llm_model=model,
            cost_estimate_usd=cost,
        )

    # 1. Candidate entities for the query.
    candidates = _candidate_entities(
        graph_store, query, namespace=namespace, user_id=user_id,
    )
    if not candidates:
        logger.debug(
            "global_query: no candidate entities for query=%r namespace=%r",
            query, namespace,
        )
        return _result((), (), 0.0)

    # 2. Pull the k-hop subgraph for each candidate; collect node keys.
    all_keys: list[str] = []
    seen: set[str] = set()
    name_to_key: dict[str, str] = {}
    key_to_display: dict[str, str] = {}
    for ent in candidates:
        sg = _safe_call(
            "get_subgraph",
            graph_store.get_subgraph,
            ent, depth=subgraph_depth,
            namespace=namespace, user_id=user_id,
        )
        if sg is None:
            # Fatal-ish — graph layer down. Bail with empty result.
            return _result((), (), 0.0)
        for node in sg.get("nodes") or []:
            key = node.get("key") or node.get("name")
            if not key or key in seen:
                continue
            seen.add(key)
            all_keys.append(key)
            display = node.get("name") or key
            name_to_key[display] = key
            key_to_display[key] = display

    if not all_keys:
        return _result((), (), 0.0)

    # 3. Leiden community detection.
    communities = _leiden_communities(
        graph_store, all_keys, namespace=namespace, user_id=user_id,
    )
    if not communities:
        return _result((), (), 0.0)

    # 4. Sort communities by size (largest first), cap at max_clusters.
    ordered = sorted(
        communities.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:max_clusters]

    summaries_acc: list[str] = []
    entities_acc: list[tuple[str, ...]] = []
    cost = 0.0

    for cluster_id, member_keys in ordered:
        # Map cluster keys back to display names; cap per-cluster size.
        display_names = [
            key_to_display.get(k, k) for k in member_keys
        ][:max_entities_per_cluster]
        # 5. Fetch memories for this cluster's entities.
        mems = _memories_for_entities(
            doc_store, display_names,
            namespace=namespace, user_id=user_id,
            limit=max_entities_per_cluster * 5,
        )
        # 6. Summarize via the injected (or default) summarizer.
        try:
            summary = summary_fn(
                query, cluster_id=int(cluster_id),
                memories=mems, model=model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "global_query: summarizer raised on cluster %s (%s: %s)",
                cluster_id, type(e).__name__, e,
            )
            summary = ""
        if not summary:
            continue
        summaries_acc.append(summary)
        entities_acc.append(tuple(display_names))
        cost += float(cost_per_summary_usd)

    return _result(
        tuple(summaries_acc), tuple(entities_acc), round(cost, 6),
    )


def to_retrieval_results(
    result: GlobalQueryResult, *, namespace: str | None = None,
) -> list[RetrievalResult]:
    """Wrap ``result.summaries`` as ``RetrievalResult`` objects.

    Each summary becomes a synthetic ``Memory`` with
    ``category="global_summary"`` and metadata flagging the cluster id
    + source entities so callers can filter / route differently from
    real memories.
    """
    out: list[RetrievalResult] = []
    for cluster_id, (summary, entities) in enumerate(
        zip(result.summaries, result.cluster_entities, strict=True)
    ):
        mem = Memory(
            content=summary,
            category="global_summary",
            namespace=namespace or "default",
            metadata={
                "_cluster_id": cluster_id,
                "_source_entities": list(entities),
                "_global_query": True,
            },
        )
        out.append(
            RetrievalResult(
                memory=mem,
                # Score 1.0 / (rank+1) so larger clusters rank higher.
                score=1.0 / float(cluster_id + 1),
                match_source="global_query",
            )
        )
    return out


__all__ = [
    "GlobalQueryResult",
    "is_global_query",
    "run_global_query",
    "to_retrieval_results",
]
