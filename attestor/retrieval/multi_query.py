"""Multi-query retrieval: rewrite + RRF-merge.

A small LLM rewrites the user question into N paraphrases, each is run
through the vector lane independently, and the N hit lists are merged
via Reciprocal Rank Fusion before entering the rest of the cascade.

Per the RCA roadmap, this is the single biggest accuracy lever (+8%
on LME-S). The reasoning: a single embedding query has poor recall on
multi-hop questions ("what was X before the merger") because the
embedder collapses too many surface forms into one vector. Paraphrases
spread the query across the latent space, raising the chance the
correct memory is in *some* lane's top-K even if it's missing from
the original lane's top-K.

This module is pure — it produces (rewrites: list[str], merged_hits:
list[dict]) but does not call the vector store. The orchestrator wires
the lane into its existing pipeline, which keeps every other stage
(BM25, graph, MMR, budget fit) unchanged.

Configuration lives in ``configs/attestor.yaml`` under
``retrieval.multi_query``:

    retrieval:
      multi_query:
        enabled: false        # default off — flip on per bench run
        n: 3                  # how many rewrites
        rewriter_model: null  # null → models.extraction
        rewriter_reasoning_effort: low
        merge: rrf            # rrf | union

Trace events emitted (when ATTESTOR_TRACE=1):

  - ``recall.multi_query.rewrites`` — the N rewrites produced
  - ``recall.multi_query.merged`` — pre-merge per-lane sizes + final size
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("attestor.retrieval.multi_query")


# ──────────────────────────────────────────────────────────────────────
# Rewriter
# ──────────────────────────────────────────────────────────────────────


_REWRITE_PROMPT = """You are a query-rewriting assistant for a personal-memory \
search system. Given the user's question, produce {n} short paraphrases that \
preserve the original intent but vary the surface form (synonyms, alternate \
phrasings, focus on different keywords).

Rules:
- Each paraphrase must be a single line, ≤ 25 words.
- Do NOT add new entities or facts.
- Do NOT ask clarifying questions.
- Output a JSON array of strings only — no preamble, no commentary.

Question: {question}

JSON:"""


@dataclass(frozen=True)
class RewriteResult:
    """The rewriter's output. ``original`` is always included as the
    first element of ``queries`` so downstream code can iterate one
    list and recover both the original + paraphrases."""

    original: str
    paraphrases: List[str] = field(default_factory=list)

    @property
    def queries(self) -> List[str]:
        # original first, then paraphrases — rank position matters for RRF
        return [self.original, *self.paraphrases]


def _resolve_rewriter_model() -> str:
    """Rewriter model: env override > YAML > extraction default."""
    if env := os.environ.get("MULTI_QUERY_REWRITER_MODEL"):
        return env
    from attestor.config import get_stack
    stack = get_stack()
    multi_query_model = getattr(stack.retrieval.multi_query, "rewriter_model", None) \
        if hasattr(stack.retrieval, "multi_query") else None
    if multi_query_model:
        return multi_query_model
    return stack.models.extraction


def rewrite_query(
    question: str,
    *,
    n: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> RewriteResult:
    """Produce ``n`` paraphrases of ``question`` via a single LLM call.

    On any error (missing key, malformed JSON, timeout), returns the
    original question alone — multi-query degrades gracefully to
    single-query rather than failing the whole recall.
    """
    if n <= 0 or not question.strip():
        return RewriteResult(original=question.strip())

    model = model or _resolve_rewriter_model()

    try:
        from attestor.llm_trace import (
            get_client_for_model,
            _get_pool,
            make_client,
            traced_create,
        )
        if api_key:
            pool = _get_pool()
            head, sep, tail = model.partition("/")
            if sep and head in pool.providers:
                strategy = pool._strategies[head]  # noqa: SLF001
                clean_model = tail
            else:
                strategy = pool.default_strategy()
                clean_model = model
            client = make_client(
                base_url=strategy.base_url,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            client, clean_model = get_client_for_model(model)
        response = traced_create(
            client,
            role="multi_query_rewriter",
            model=clean_model,
            max_tokens=400,
            messages=[
                {"role": "user", "content": _REWRITE_PROMPT.format(
                    n=n, question=question,
                )},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("multi_query.rewrite_query: LLM call failed: %s", e)
        return RewriteResult(original=question.strip())

    paraphrases = _parse_rewrites(text, n=n)

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.multi_query.rewrites",
            original=question[:200],
            n_requested=n,
            n_produced=len(paraphrases),
            paraphrases=paraphrases[:5],
        )

    return RewriteResult(
        original=question.strip(),
        paraphrases=paraphrases,
    )


# ──────────────────────────────────────────────────────────────────────
# Async rewriter (Phase 2 — docs/plans/async-retrieval/PLAN.md).
# Pins temperature=0.0 — same audit invariant A7 as HyDE: same prompt
# + same model = same paraphrases = same RRF order. Async amplifies
# non-determinism risk if T > 0.
# ──────────────────────────────────────────────────────────────────────


async def rewrite_query_async(
    question: str,
    *,
    n: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> RewriteResult:
    """Async sibling of ``rewrite_query``. Same prompt, same parser,
    same fallback semantics. Uses ``openai.AsyncOpenAI`` so the
    rewriter call can run concurrently with vector lanes.
    """
    if n <= 0 or not question.strip():
        return RewriteResult(original=question.strip())

    model = model or _resolve_rewriter_model()

    try:
        from attestor.llm_trace import _get_pool, make_async_client
        pool = _get_pool()
        head, sep, tail = model.partition("/")
        if sep and head in pool.providers:
            strategy = pool._strategies[head]  # noqa: SLF001
            clean_model = tail
        elif sep:
            logger.debug(
                "multi_query.rewrite_query_async: unknown provider %r in "
                "model %r", head, model,
            )
            return RewriteResult(original=question.strip())
        else:
            strategy = pool.default_strategy()
            clean_model = model

        key = api_key or os.environ.get(strategy.api_key_env)
        if not key:
            logger.debug(
                "multi_query.rewrite_query_async: %s not set; returning "
                "original only", strategy.api_key_env,
            )
            return RewriteResult(original=question.strip())

        client = make_async_client(
            base_url=strategy.base_url,
            api_key=key,
            timeout=timeout,
        )
        response = await client.chat.completions.create(
            model=clean_model,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "user", "content": _REWRITE_PROMPT.format(
                    n=n, question=question,
                )},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("multi_query.rewrite_query_async: LLM call failed: %s", e)
        return RewriteResult(original=question.strip())

    paraphrases = _parse_rewrites(text, n=n)

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.multi_query.rewrites.async",
            original=question[:200],
            n_requested=n,
            n_produced=len(paraphrases),
            paraphrases=paraphrases[:5],
        )

    return RewriteResult(
        original=question.strip(),
        paraphrases=paraphrases,
    )


def _parse_rewrites(text: str, *, n: int) -> List[str]:
    """Extract a JSON array of strings from the rewriter response.

    Handles fenced code blocks (```json ... ```) and stray text around
    the array. Returns up to ``n`` non-empty stripped strings.
    """
    if not text:
        return []

    # Strip fenced code blocks.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        text = text.strip()

    # Try direct parse first.
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Find the first array bracket and parse from there.
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []

    if not isinstance(parsed, list):
        return []

    out: List[str] = []
    for item in parsed:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
# RRF merge
# ──────────────────────────────────────────────────────────────────────

# k=60 is the standard RRF constant from Cormack et al. (2009). It
# tempers the contribution of high-rank items so a memory that lands
# at rank 1 in only one lane doesn't crowd out a memory at rank 5 in
# all lanes.
RRF_K = 60


def reciprocal_rank_fusion(
    lanes: Sequence[Sequence[Dict[str, Any]]],
    *,
    key: str = "memory_id",
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Merge N ranked lists into one via reciprocal rank fusion.

    Each ``lane`` is an ordered list of hit dicts (vector_store.search
    output shape). The dict's ``key`` field identifies the same item
    across lanes. Returns a single list ordered by descending RRF
    score, with ``rrf_score`` and ``per_lane_ranks`` annotations
    appended to each hit.

    Stable for empty/single-lane input — degenerate to that lane's
    order.
    """
    if not lanes:
        return []

    # First-occurrence wins for the hit dict itself; we annotate it
    # with the merged score below. This preserves the lane ordering
    # for tie-breaking.
    seen: Dict[Any, Dict[str, Any]] = {}
    score: Dict[Any, float] = {}
    per_lane_ranks: Dict[Any, List[int]] = {}

    for lane_idx, lane in enumerate(lanes):
        for rank, hit in enumerate(lane, start=1):
            mid = hit.get(key)
            if mid is None:
                continue
            score[mid] = score.get(mid, 0.0) + 1.0 / (k + rank)
            per_lane_ranks.setdefault(mid, []).append(rank)
            if mid not in seen:
                # Copy so we don't mutate the caller's list elements.
                seen[mid] = dict(hit)

    merged: List[Dict[str, Any]] = []
    for mid, hit in seen.items():
        hit = dict(hit)
        hit["rrf_score"] = round(score[mid], 6)
        hit["per_lane_ranks"] = per_lane_ranks[mid]
        merged.append(hit)

    merged.sort(key=lambda h: h["rrf_score"], reverse=True)
    return merged


def union_merge(
    lanes: Sequence[Sequence[Dict[str, Any]]],
    *,
    key: str = "memory_id",
) -> List[Dict[str, Any]]:
    """Simple deduplicated union — preserves first-seen order across
    lanes, no rank fusion. Cheaper than RRF but loses cross-lane
    consistency signal. Used as a fallback when ``merge: union`` is
    set in YAML."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for lane in lanes:
        for hit in lane:
            mid = hit.get(key)
            if mid is None or mid in seen:
                continue
            seen.add(mid)
            out.append(dict(hit))
    return out


# ──────────────────────────────────────────────────────────────────────
# End-to-end multi-query lane
# ──────────────────────────────────────────────────────────────────────


def multi_query_search(
    question: str,
    *,
    vector_search: Callable[[str], List[Dict[str, Any]]],
    n: int = 3,
    merge: str = "rrf",
    rewriter_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Run the rewriter, fan out to ``vector_search`` per query,
    merge the lanes, return ``(queries_used, merged_hits)``.

    ``vector_search`` must be a 1-arg callable mapping query → list of
    hit dicts (already namespace/limit-bound). The orchestrator wraps
    its ``vector_store.search`` in a closure to provide this.
    """
    rewrite = rewrite_query(
        question, n=n, model=rewriter_model, api_key=api_key,
    )
    queries = rewrite.queries

    lanes: List[List[Dict[str, Any]]] = []
    for q in queries:
        try:
            hits = list(vector_search(q))
        except Exception as e:  # noqa: BLE001
            logger.debug("multi_query: lane %r failed: %s", q[:60], e)
            hits = []
        lanes.append(hits)

    if merge == "union":
        merged = union_merge(lanes)
    else:
        merged = reciprocal_rank_fusion(lanes)

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.multi_query.merged",
            n_queries=len(queries),
            per_lane_sizes=[len(l) for l in lanes],
            merged_size=len(merged),
            merge_strategy=merge,
        )

    return queries, merged


# ──────────────────────────────────────────────────────────────────────
# Async end-to-end lane (Phase 2 — docs/plans/async-retrieval/PLAN.md)
#
# Two concurrency levers:
#   1. Rewriter LLM call runs concurrently with the original-question
#      vector search (orig lane embeds while rewriter writes paraphrases).
#   2. Once paraphrases are known, the N paraphrase lanes fan out
#      simultaneously via asyncio.gather. Wallclock collapses from
#      (N+1) × per-lane to ~max(per-lane).
#
# Audit invariants preserved:
#   A7 — rewriter pins temperature=0.0 (same as HyDE generator).
#   RRF merge is deterministic given identical lane inputs — gather
#   order does NOT affect rank positions because ranks are assigned
#   per-lane independently before merge (see test_multi_query_async
#   _preserves_RRF_order).
# ──────────────────────────────────────────────────────────────────────


async def multi_query_search_async(
    question: str,
    *,
    vector_search: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    n: int = 3,
    merge: str = "rrf",
    rewriter_model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Async sibling of ``multi_query_search``. ``vector_search`` MUST
    be an async callable.

    Latency story (with both LLM and vector ~200ms each, n=3 → 4 lanes):
        sync:  ~1000ms (rewriter 200 + 4 lanes × 200 sequential)
        async: ~400ms  (rewriter 200 ‖ orig lane 200, then 3 paraphrase
                        lanes gathered ≈ 200) — ~60% reduction

    Per-lane failures are isolated via gather(return_exceptions=True);
    RRF degrades to surviving lanes. On rewriter failure, falls back
    to single-lane (original-question) without raising.
    """
    # Phase A: rewriter + orig-lane in parallel. Orig lane is "free"
    # on the wallclock — it runs while the LLM writes paraphrases.
    rewriter_task = asyncio.create_task(
        rewrite_query_async(
            question, n=n, model=rewriter_model, api_key=api_key,
            timeout=timeout,
        )
    )
    orig_task = asyncio.create_task(_safe_async_call(vector_search, question))

    try:
        rewrite = await asyncio.wait_for(rewriter_task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.debug("multi_query_async: rewriter exceeded timeout=%.2fs", timeout)
        rewrite = RewriteResult(original=question.strip())
    except Exception as e:  # noqa: BLE001
        logger.debug("multi_query_async: rewriter failed: %s", e)
        rewrite = RewriteResult(original=question.strip())

    queries = rewrite.queries

    # Phase B: fan out paraphrase lanes (all known after rewriter
    # returns) gathered with the already-running orig task.
    paraphrase_tasks = [
        asyncio.create_task(_safe_async_call(vector_search, p))
        for p in rewrite.paraphrases
    ]
    all_tasks = [orig_task] + paraphrase_tasks
    lanes_results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Coerce to lanes; exceptions become empty lanes (RRF handles).
    lanes: List[List[Dict[str, Any]]] = []
    for lr in lanes_results:
        if isinstance(lr, Exception):
            lanes.append([])
        else:
            lanes.append(lr or [])

    if merge == "union":
        merged = union_merge(lanes)
    else:
        merged = reciprocal_rank_fusion(lanes)

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.multi_query.merged.async",
            n_queries=len(queries),
            per_lane_sizes=[len(l) for l in lanes],
            merged_size=len(merged),
            merge_strategy=merge,
        )

    return queries, merged


async def _safe_async_call(
    fn: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    arg: str,
) -> List[Dict[str, Any]]:
    """Await ``fn(arg)``; on exception, return ``[]`` so the lane
    contributes empty to RRF rather than raising. Mirrors the helper
    in ``attestor/retrieval/hyde.py`` — kept local to avoid a
    circular import."""
    try:
        return list(await fn(arg))
    except Exception as e:  # noqa: BLE001
        logger.debug("multi_query_async: lane %r failed: %s", arg[:60], e)
        return []
