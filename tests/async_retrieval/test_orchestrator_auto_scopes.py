"""Audit-completeness tests — both sync ``recall()`` and async
``recall_async()`` must auto-open ``recall_started_at_scope`` and
``trace.recall_scope`` so audit invariants A2 and A5 are enforced
for ALL callers (production smoke runners, the LME pipeline, etc.),
not just async-aware tests that explicitly wrap their recalls.

Closes the gap surfaced in `logs/lme_smoke_20260430_005019.jsonl`
where 1458 sync-recall events emitted ZERO recall_id / seq fields
because no caller opened the scopes manually.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from attestor.models import Memory


pytestmark = [pytest.mark.unit]


def _build_minimal_orchestrator():
    """Minimal orchestrator stub with stubbed-but-non-empty lanes so
    ``_post_process_candidates`` produces at least one event."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    orch = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
    orch.store = MagicMock()
    orch.store.get = lambda mid: Memory(
        id=mid, content="x", entity="alice", category="role",
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
    orch.bm25_lane = None  # disable BM25 to keep events count predictable

    vec_store = MagicMock()
    vec_store.search = lambda q, **kwargs: [
        {"memory_id": "m1", "distance": 0.1},
    ]
    orch.vector_store = vec_store

    orch._question_entities = lambda q: []
    orch._graph_affinity_map = lambda ents, namespace=None: {}
    orch._graph_context_triples = lambda ents, namespace=None: []
    orch._blend_score = lambda sim, hop: (sim, 0.0)
    return orch


def test_sync_recall_auto_opens_scopes(tmp_path, monkeypatch):
    """Sync ``recall()`` must auto-open both audit scopes — every event
    emitted has ``recall_id``, ``seq``, and the same ``recall_id`` is
    shared by all events from a single recall (so they reconstruct as
    a tree)."""
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    import attestor.trace as tr
    tr.reset_for_test()

    orch = _build_minimal_orchestrator()
    orch.recall("query")

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    assert len(events) >= 1, "sync recall must emit at least one trace event"

    # Every event must have a recall_id.
    rids = {e.get("recall_id") for e in events}
    assert None not in rids, (
        f"sync recall emitted events WITHOUT recall_id; "
        f"recall_ids seen: {rids}"
    )
    # All events share the same recall_id (one recall = one tree).
    assert len(rids) == 1, (
        f"sync recall must produce exactly ONE recall_id; got {rids}"
    )
    # seq is monotonic per-recall, starting at 1.
    seqs = [e.get("seq") for e in events]
    assert seqs == sorted(seqs), f"seqs must be monotonic; got {seqs}"
    assert min(seqs) == 1, "seq must start at 1"


def test_sync_recall_auto_activates_recall_started_at_ceiling():
    """Sync ``recall()`` must auto-open ``recall_started_at_scope``
    so ``current_recall_started_at()`` returns a non-None value during
    the recall — verified by reading the contextvar from inside the
    vector_store stub."""
    from attestor.recall_context import current_recall_started_at

    orch = _build_minimal_orchestrator()

    seen_inside: list = []
    def vec_search(q, **kwargs):
        seen_inside.append(current_recall_started_at())
        return [{"memory_id": "m1", "distance": 0.1}]
    orch.vector_store.search = vec_search

    assert current_recall_started_at() is None  # outside scope
    orch.recall("query")
    assert current_recall_started_at() is None  # back to None

    assert len(seen_inside) == 1
    assert seen_inside[0] is not None, (
        "ceiling must be active inside vector lane during sync recall"
    )


@pytest.mark.asyncio
async def test_async_recall_auto_opens_trace_scope(tmp_path, monkeypatch):
    """``recall_async()`` already opened the ceiling scope (P6); this
    test pins that it now ALSO opens ``trace.recall_scope`` so async
    events get recall_id / seq tagging too."""
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    import attestor.trace as tr
    tr.reset_for_test()

    orch = _build_minimal_orchestrator()
    await orch.recall_async("query")

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    assert len(events) >= 1
    rids = {e.get("recall_id") for e in events}
    assert None not in rids, (
        f"async recall emitted events WITHOUT recall_id; recall_ids seen: {rids}"
    )
    assert len(rids) == 1
    # mode=async should appear on the recall.start event.
    starts = [e for e in events if e.get("event") == "recall.start"]
    assert len(starts) == 1
    assert starts[0].get("mode") == "async"


@pytest.mark.asyncio
async def test_concurrent_async_recalls_get_distinct_recall_ids(tmp_path, monkeypatch):
    """Two concurrent ``recall_async`` calls must get DIFFERENT
    recall_ids — proves the contextvar is per-task, not shared."""
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    import attestor.trace as tr
    tr.reset_for_test()

    orch = _build_minimal_orchestrator()
    await asyncio.gather(
        orch.recall_async("q1"),
        orch.recall_async("q2"),
        orch.recall_async("q3"),
    )

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    rids = {e.get("recall_id") for e in events if e.get("recall_id")}
    assert len(rids) == 3, (
        f"3 concurrent recalls must produce 3 distinct recall_ids; got {len(rids)}: {rids}"
    )
