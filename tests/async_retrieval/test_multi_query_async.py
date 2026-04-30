"""Phase 2 — RED tests for ``multi_query_search_async``.

Target API (does NOT exist yet — these tests will fail until PR-2 lands):

    from attestor.retrieval.multi_query import multi_query_search_async

    queries, merged = await multi_query_search_async(
        question,
        vector_search=async_callable,
        n=3,
        merge="rrf",
    )

The async version must:
  1. Run the rewriter LLM call concurrently with the original-question
     vector embed (same gather pattern as HyDE).
  2. Fan out the N+1 vector searches in parallel — wallclock close to
     max(per-lane), NOT sum.
  3. RRF-merge the lanes in identical order to the sync version.
  4. Survive a partial lane failure (one paraphrase's vector search
     raises) and return the remaining lanes' merged hits.

See ``docs/plans/async-retrieval/PLAN.md`` § 4 — Phase 2 row.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from tests.async_retrieval.conftest import (
    StopwatchAsync,
    make_failing_search,
    make_slow_search,
)


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ──────────────────────────────────────────────────────────────────────
# Concurrency assertions
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_query_async_parallelizes_n_lanes():
    """With n=3 paraphrases (so 4 lanes total: original + 3) and each
    vector search sleeping 200ms, sequential would be 800ms+; parallel
    gather must finish in ~250ms."""
    from attestor.retrieval.multi_query import (  # RED
        multi_query_search_async,
        RewriteResult,
    )

    slow_search = make_slow_search(per_call_ms=200)

    async def fake_rewriter(question, **kwargs):
        return RewriteResult(
            original=question,
            paraphrases=[f"paraphrase {i}" for i in range(3)],
        )

    with patch(
        "attestor.retrieval.multi_query.rewrite_query_async",
        side_effect=fake_rewriter,
    ):
        async with StopwatchAsync() as sw:
            queries, merged = await multi_query_search_async(
                "test question", vector_search=slow_search, n=3,
            )

    # 4 lanes × 200ms sequential = 800ms. Parallel = ~250ms tops.
    assert sw.elapsed_ms < 500, (
        f"multi-query lanes ran sequentially? wallclock={sw.elapsed_ms:.0f}ms, "
        f"expected < 500ms with gather"
    )
    assert len(queries) == 4, "expected original + 3 paraphrases"
    assert len(merged) >= 1


@pytest.mark.asyncio
async def test_multi_query_async_preserves_RRF_order():
    """Async fan-out must produce the same RRF-ranked merged list as
    the sync version given identical inputs. Differences would mean
    the gather order corrupted the per-lane rank-position fed into RRF.
    """
    from attestor.retrieval.multi_query import (  # RED
        multi_query_search,
        multi_query_search_async,
        RewriteResult,
    )

    fixed_paraphrases = ["alt 1", "alt 2"]

    def sync_search(q: str):
        # Deterministic per-query hits — order matters for RRF merge.
        return [
            {"memory_id": f"{q}-a", "session_id": "s", "distance": 0.1},
            {"memory_id": f"{q}-b", "session_id": "s", "distance": 0.2},
        ]

    async def async_search(q: str):
        return sync_search(q)

    sync_rewrite = RewriteResult(original="q", paraphrases=fixed_paraphrases)

    with patch(
        "attestor.retrieval.multi_query.rewrite_query",
        return_value=sync_rewrite,
    ), patch(
        "attestor.retrieval.multi_query.rewrite_query_async",
        return_value=sync_rewrite,
    ):
        sync_queries, sync_merged = multi_query_search(
            "q", vector_search=sync_search, n=2,
        )
        async_queries, async_merged = await multi_query_search_async(
            "q", vector_search=async_search, n=2,
        )

    assert sync_queries == async_queries
    assert [h["memory_id"] for h in sync_merged] == [
        h["memory_id"] for h in async_merged
    ], "RRF order must match — async gather must not corrupt rank positions"


# ──────────────────────────────────────────────────────────────────────
# Partial failure
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_query_async_handles_partial_lane_failure():
    """If the second-lane vector search raises, the original-question
    lane must still produce a merged result — gather(return_exceptions=True)
    contract."""
    from attestor.retrieval.multi_query import (  # RED
        multi_query_search_async,
        RewriteResult,
    )

    failing_search = make_failing_search(when="second_lane")

    async def fake_rewriter(question, **kwargs):
        return RewriteResult(original=question, paraphrases=["alt"])

    with patch(
        "attestor.retrieval.multi_query.rewrite_query_async",
        side_effect=fake_rewriter,
    ):
        # Should NOT raise. RRF merges only the surviving lane.
        queries, merged = await multi_query_search_async(
            "q", vector_search=failing_search, n=1,
        )

    assert len(queries) == 2
    assert len(merged) >= 1, "first lane succeeded → at least one merged hit"
