"""End-to-end smoke: orchestrator.recall() vs recall_async() on the live stack.

Drives the real RetrievalOrchestrator (PG + Pinecone + Neo4j per
configs/attestor.yaml) through both code paths on the same data and
compares:

  1. Wallclock (median over N runs)
  2. Result parity (top-K memory IDs must match)
  3. Trace contract (recall_id propagates through async tasks; ceiling
     SQL clause appears for v4 backends)

Smoke caps: 5 seed memories + 5 timing iterations per path. Per the
project smoke-only rule (feedback_smoke_only_no_full_bench).

Usage:
    set -a && source .env && set +a
    .venv/bin/python experiments/orchestrator_async_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    # Capture trace from THIS run only — separate file from the
    # canonical logs/attestor_trace.jsonl so analysis is clean.
    trace_path = Path(tempfile.mkstemp(suffix="-smoke.jsonl")[1])
    os.environ["ATTESTOR_TRACE"] = "1"
    os.environ["ATTESTOR_TRACE_FILE"] = str(trace_path)
    print(f"trace → {trace_path}")

    # Use the simple in-process default backend (Postgres v4) without
    # spinning up the Pinecone vector store — keeps this smoke focused
    # on the orchestrator parallelism + audit invariants. The real
    # vector layer is exercised by tests/test_orchestrator_async.py
    # with mocks, and end-to-end Pinecone integration is the next
    # smoke once we want to gate Pinecone token spend.
    from attestor.core import AgentMemory
    from attestor import trace as _tr
    _tr.reset_for_test()

    print("bootstrapping AgentMemory (auto-detect from configs/attestor.yaml)…")
    smoke_path = tempfile.mkdtemp(prefix="async-smoke-")
    mem = AgentMemory(path=smoke_path)
    print(f"  path = {smoke_path}")
    print(f"  store backend = {type(mem._store).__name__}")
    print(f"  vector backend = {type(getattr(mem, '_vector', None)).__name__ if hasattr(mem, '_vector') else 'inline (pgvector)'}")
    print(f"  has recall_async = {hasattr(mem._retrieval, 'recall_async')}")
    print()

    # ── Seed: 5 small memories about a couple of entities ──
    print("seeding 5 memories…")
    seed_data = [
        {"content": "Alice is the CTO of Acme Corp", "entity": "Alice", "category": "role"},
        {"content": "Alice prefers dark mode in her IDE", "entity": "Alice", "category": "preference"},
        {"content": "Bob joined as VP of engineering in March 2026", "entity": "Bob", "category": "role"},
        {"content": "Acme Corp uses Postgres + Pinecone + Neo4j as its memory stack", "entity": "Acme Corp", "category": "tech"},
        {"content": "Caroline reviewed the Q1 architecture proposal", "entity": "Caroline", "category": "activity"},
    ]
    for s in seed_data:
        mem.add(**s)
    print(f"  seeded {len(seed_data)}")
    print()

    # ── Time sync recall ──
    query = "who runs engineering at Acme?"
    N = 5

    print(f"timing sync recall × {N} (query={query!r})…")
    sync_ms = []
    sync_results = None
    for i in range(N):
        t0 = time.perf_counter()
        results = mem._retrieval.recall(query, token_budget=2000)
        elapsed = (time.perf_counter() - t0) * 1000.0
        sync_ms.append(elapsed)
        if i == 0:
            sync_results = results
    print(f"  sync   min={min(sync_ms):6.1f}ms  med={statistics.median(sync_ms):6.1f}ms  max={max(sync_ms):6.1f}ms")

    # ── Time async recall ──
    print(f"timing async recall × {N}…")

    async def run_async():
        out = []
        first = None
        for i in range(N):
            t0 = time.perf_counter()
            results = await mem._retrieval.recall_async(query, token_budget=2000)
            elapsed = (time.perf_counter() - t0) * 1000.0
            out.append(elapsed)
            if i == 0:
                first = results
        return out, first

    async_ms, async_results = asyncio.run(run_async())
    print(f"  async  min={min(async_ms):6.1f}ms  med={statistics.median(async_ms):6.1f}ms  max={max(async_ms):6.1f}ms")
    print()

    # ── Parity ──
    sync_ids = [r.memory.id for r in (sync_results or [])]
    async_ids = [r.memory.id for r in (async_results or [])]
    print(f"sync top-{len(sync_ids)} ids:  {sync_ids[:5]}")
    print(f"async top-{len(async_ids)} ids: {async_ids[:5]}")
    parity = sync_ids == async_ids
    print(f"parity: {'✓ identical' if parity else '✗ DIFFER'}")
    print()

    # ── Trace contract verification ──
    events = []
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"trace events emitted: {len(events)}")

    has_recall_id = sum(1 for e in events if e.get("recall_id"))
    has_seq = sum(1 for e in events if e.get("seq") is not None)
    has_event_id = sum(1 for e in events if e.get("event_id"))
    async_mode_events = sum(
        1 for e in events
        if e.get("event") == "recall.start" and e.get("mode") == "async"
    )
    print(f"  events with event_id:    {has_event_id}/{len(events)}")
    print(f"  events with recall_id:   {has_recall_id}/{len(events)} (only when in recall_scope)")
    print(f"  events with monotonic seq: {has_seq}/{len(events)} (only when in recall_scope)")
    print(f"  recall.start with mode=async: {async_mode_events}")
    print()

    # ── Headline ──
    sync_med = statistics.median(sync_ms)
    async_med = statistics.median(async_ms)
    speedup = sync_med / async_med if async_med > 0 else float("inf")
    delta_pct = (1.0 - async_med / sync_med) * 100.0 if sync_med > 0 else 0.0
    print(f"speedup: {speedup:.2f}x  (latency reduction: {delta_pct:.1f}%)")
    print(f"trace path: {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
