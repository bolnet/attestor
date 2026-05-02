"""Synthetic latency bench — sync recall() vs async recall_async().

Stubs the vector_store and bm25_lane with controlled-sleep callables so
we can measure parallelism gains independent of any real backend (no
Pinecone tokens consumed, no DB roundtrips, deterministic numbers).

Reports min/median/max wallclock across N runs for both paths.

Usage:
    .venv/bin/python experiments/async_recall_latency.py
    .venv/bin/python experiments/async_recall_latency.py --vec-ms 100 --bm25-ms 80 --runs 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attestor.models import Memory  # noqa: E402
from attestor.retrieval.orchestrator import RetrievalOrchestrator  # noqa: E402


def _build_stub_orchestrator(vec_sleep_ms: int, bm25_sleep_ms: int) -> RetrievalOrchestrator:
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

    vec_store = MagicMock()
    def vec_search(q, **kwargs):
        time.sleep(vec_sleep_ms / 1000.0)
        return [{"memory_id": "mem-vec-1", "distance": 0.1}]
    vec_store.search = vec_search
    orch.vector_store = vec_store

    bm25_lane = MagicMock()
    def bm25_search(q, **kwargs):
        time.sleep(bm25_sleep_ms / 1000.0)
        h = MagicMock()
        h.memory_id = "mem-bm25-1"
        h.rank = 0.5
        return [h]
    bm25_lane.search = bm25_search
    orch.bm25_lane = bm25_lane

    orch._question_entities = lambda q: []
    orch._graph_affinity_map = lambda ents, namespace=None: {}
    orch._graph_context_triples = lambda ents, namespace=None: []
    orch._blend_score = lambda sim, hop: (sim, 0.0)
    return orch


def _time_sync(orch: RetrievalOrchestrator, runs: int) -> list[float]:
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        orch.recall("query")
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


async def _time_async(orch: RetrievalOrchestrator, runs: int) -> list[float]:
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await orch.recall_async("query")
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _summary(label: str, ms_list: list[float]) -> None:
    print(
        f"  {label:<10s}  "
        f"min={min(ms_list):6.1f}ms  "
        f"med={statistics.median(ms_list):6.1f}ms  "
        f"mean={statistics.mean(ms_list):6.1f}ms  "
        f"max={max(ms_list):6.1f}ms"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vec-ms", type=int, default=100)
    p.add_argument("--bm25-ms", type=int, default=100)
    p.add_argument("--runs", type=int, default=10)
    args = p.parse_args()

    orch = _build_stub_orchestrator(args.vec_ms, args.bm25_ms)

    print(
        f"Bench: vector_lane={args.vec_ms}ms, bm25_lane={args.bm25_ms}ms, "
        f"runs={args.runs}"
    )
    print(f"Sequential expected: ~{args.vec_ms + args.bm25_ms}ms")
    print(f"Parallel expected:   ~{max(args.vec_ms, args.bm25_ms)}ms")
    print()

    sync_ms = _time_sync(orch, args.runs)
    async_ms = asyncio.run(_time_async(orch, args.runs))

    _summary("sync", sync_ms)
    _summary("async", async_ms)

    sync_med = statistics.median(sync_ms)
    async_med = statistics.median(async_ms)
    speedup = sync_med / async_med if async_med > 0 else float("inf")
    delta_pct = (1.0 - async_med / sync_med) * 100.0 if sync_med > 0 else 0.0
    print()
    print(f"speedup: {speedup:.2f}x  (latency reduction: {delta_pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
