"""Phase 6 — orchestrator-level async recall tests.

Pins three contracts:

  1. ``recall_async()`` runs the vector lane and BM25 lane concurrently
     via ``asyncio.gather`` — wallclock < sum.

  2. A failing lane does NOT abort the recall — the surviving lane's
     hits flow through to the standard post-processing.

  3. ``recall_async()`` opens a ``recall_started_at_scope_async`` so
     the Postgres ceiling (audit invariant **A2**) is active for the
     entire async recall — no concurrent writes leak in mid-flight.

See ``docs/plans/async-retrieval/PLAN.md`` § 4 — Phase 6.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from attestor.models import Memory


pytestmark = [pytest.mark.unit]


def _make_orchestrator_with_slow_lanes(
    vec_sleep_ms: int, bm25_sleep_ms: int,
):
    """Build a stub orchestrator whose vector lane and BM25 lane each
    sleep for the configured duration, then return a fixed hit list.
    Used to assert the lanes actually overlap in wallclock under
    ``asyncio.gather``."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    # Bypass __init__ — set just the attributes recall_async() needs.
    orch = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
    orch.store = MagicMock()
    orch.store.get = lambda mid: Memory(
        id=mid, content=f"content {mid}", entity="alice", category="role",
        namespace="default", status="active", confidence=1.0,
    )
    orch.vector_top_k = 50
    orch.bm25_top_k = 50
    orch.bm25_min_rank = 0.0
    orch.enable_temporal_boost = False
    orch.enable_mmr = False
    orch.mmr_lambda = 0.7
    orch.mmr_top_n = None
    orch.confidence_decay_rate = 0.0
    orch.confidence_boost_rate = 0.0
    orch.confidence_gate = 0.0
    orch.temporal_prefilter_cfg = None
    orch.multi_query_cfg = None
    orch.hyde_cfg = None

    # vector_store stub: sleep then return one hit per query.
    vec_store = MagicMock()
    def slow_vec_search(q, **kwargs):
        time.sleep(vec_sleep_ms / 1000.0)
        return [{"memory_id": "mem-vec-1", "distance": 0.1}]
    vec_store.search = slow_vec_search
    orch.vector_store = vec_store

    # bm25_lane stub: sleep then return one hit.
    bm25_lane = MagicMock()
    def slow_bm25_search(q, **kwargs):
        time.sleep(bm25_sleep_ms / 1000.0)
        h = MagicMock()
        h.memory_id = "mem-bm25-1"
        h.rank = 0.5
        return [h]
    bm25_lane.search = slow_bm25_search
    orch.bm25_lane = bm25_lane

    # Stub graph helpers used inside _post_process_candidates.
    orch._question_entities = lambda q: []
    orch._graph_affinity_map = lambda ents, namespace=None: {}
    orch._graph_context_triples = lambda ents, namespace=None: []
    orch._blend_score = lambda sim, hop: (sim, 0.0)

    return orch


# ──────────────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_recall_async_runs_vector_and_bm25_concurrently():
    """With both lanes sleeping 200ms, sequential = 400ms;
    parallel via gather + to_thread = ~200ms."""
    orch = _make_orchestrator_with_slow_lanes(
        vec_sleep_ms=200, bm25_sleep_ms=200,
    )

    t0 = time.perf_counter()
    results = await orch.recall_async("test query", token_budget=2000)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Sequential would be ~400ms; parallel ~200ms + a little overhead.
    assert elapsed_ms < 350, (
        f"vector ‖ BM25 ran sequentially? wallclock={elapsed_ms:.0f}ms, "
        f"expected < 350ms with gather"
    )
    # Both lanes contributed candidates → final results non-empty.
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_orchestrator_recall_async_handles_one_lane_failure():
    """If the vector lane raises, the BM25 lane still produces results.
    `gather(return_exceptions=True)` ensures siblings keep running."""
    orch = _make_orchestrator_with_slow_lanes(
        vec_sleep_ms=10, bm25_sleep_ms=10,
    )

    # Make the vector lane explode; BM25 lane still works.
    def vec_explodes(q, **kwargs):
        raise RuntimeError("vector store down")
    orch.vector_store.search = vec_explodes

    # Should NOT raise. BM25-only path produces at least one result.
    results = await orch.recall_async("test query")
    assert isinstance(results, list)
    # BM25 lane returns one hit → one result via the BM25-only path.
    assert any(r.memory.id == "mem-bm25-1" for r in results), (
        "BM25 lane's hit must surface even when vector lane errors"
    )


# ──────────────────────────────────────────────────────────────────────
# Audit invariant A2 — recall_started_at scope active during async
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_recall_async_opens_recall_started_at_scope():
    """A2 — ``recall_async`` must open a ``recall_started_at_scope_async``
    so the Postgres ceiling is active throughout. We verify this by
    reading ``current_recall_started_at()`` from inside the vector_store
    stub — it must be non-None during the recall and None afterwards."""
    from attestor.recall_context import current_recall_started_at

    orch = _make_orchestrator_with_slow_lanes(
        vec_sleep_ms=5, bm25_sleep_ms=5,
    )

    seen: list = []

    def vec_search_with_ceiling_check(q, **kwargs):
        seen.append(current_recall_started_at())
        return [{"memory_id": "x", "distance": 0.1}]
    orch.vector_store.search = vec_search_with_ceiling_check

    assert current_recall_started_at() is None  # outside scope
    await orch.recall_async("test query")
    assert current_recall_started_at() is None  # back to None

    # Inside the lane, the ceiling was active (non-None datetime).
    assert len(seen) == 1
    assert seen[0] is not None, (
        "ceiling must be active inside vector lane during recall_async"
    )


# ──────────────────────────────────────────────────────────────────────
# Behavioral parity — sync recall() and recall_async() produce the
# same final results given identical inputs
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_recall_async_matches_sync_recall_results():
    """Given identical stubs (no sleep), recall() and recall_async()
    must produce the same list of memory IDs in the same order. The
    async version is a latency optimization, not a behavior change."""
    orch_sync = _make_orchestrator_with_slow_lanes(
        vec_sleep_ms=0, bm25_sleep_ms=0,
    )
    orch_async = _make_orchestrator_with_slow_lanes(
        vec_sleep_ms=0, bm25_sleep_ms=0,
    )

    sync_results = orch_sync.recall("test query")
    async_results = await orch_async.recall_async("test query")

    assert [r.memory.id for r in sync_results] == [
        r.memory.id for r in async_results
    ], "sync and async recall must produce identical rankings"
