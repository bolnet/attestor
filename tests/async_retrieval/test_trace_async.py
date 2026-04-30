"""Phase 3 — trace schema (recall_id + seq + parent_event_id).

Tests for ``attestor.trace.recall_scope`` / ``event_scope`` and the
auto-tagging behaviour of ``trace.event``. Companion to the cross-cutting
``test_audit_invariants_under_async.test_trace_reconstructable_under_async``
which validates the same machinery under N concurrent recalls.

Audit invariant **A5** — trace events emitted inside a recall_scope
must carry enough metadata to reconstruct the per-recall event tree
from the JSONL log, even when concurrent recalls interleave.
"""

from __future__ import annotations

import asyncio
import json

import pytest


pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_trace_event_includes_recall_id(tmp_path, monkeypatch):
    """Every event emitted inside a ``recall_scope()`` must carry a
    ``recall_id`` field equal to the scope's id."""
    import attestor.trace as tr

    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    tr.reset_for_test()

    with tr.recall_scope() as rid:
        tr.event("recall.alpha", q="hello")
        tr.event("recall.beta")

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    assert len(events) == 2
    for e in events:
        assert e["recall_id"] == rid


@pytest.mark.asyncio
async def test_trace_event_includes_monotonic_seq(tmp_path, monkeypatch):
    """``seq`` is 1-based, monotonic, scoped per recall_id. After a
    second recall_scope opens, seq resets."""
    import attestor.trace as tr

    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    tr.reset_for_test()

    with tr.recall_scope():
        tr.event("recall.a")
        tr.event("recall.b")
        tr.event("recall.c")

    with tr.recall_scope():
        tr.event("recall.d")

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    seqs_first = [e["seq"] for e in events if e["event"] in {"recall.a", "recall.b", "recall.c"}]
    seqs_second = [e["seq"] for e in events if e["event"] == "recall.d"]
    assert seqs_first == [1, 2, 3]
    assert seqs_second == [1]


@pytest.mark.asyncio
async def test_trace_event_parent_id_under_async_gather(tmp_path, monkeypatch):
    """``event_scope`` propagates a ``parent_event_id`` to every event
    emitted inside, INCLUDING events emitted from gathered tasks. This
    is the load-bearing assertion for tree reconstruction under async."""
    import attestor.trace as tr

    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    tr.reset_for_test()

    async def child(label: str):
        # event() inside the task — should pick up the parent's
        # event_id from the contextvar copy that asyncio.create_task
        # makes when spawning.
        tr.event(f"child.{label}")

    with tr.recall_scope():
        tr.event("recall.parent")
        with tr.event_scope() as parent_eid:
            await asyncio.gather(child("a"), child("b"), child("c"))

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    parents = [e for e in events if e["event"] == "recall.parent"]
    children = [e for e in events if e["event"].startswith("child.")]

    assert len(parents) == 1
    assert len(children) == 3
    # Children all share the parent_event_id but the parent itself
    # doesn't (it's the root).
    for c in children:
        assert c.get("parent_event_id") == parent_eid, (
            f"child {c['event']} missing parent_event_id={parent_eid!r}; "
            f"got {c.get('parent_event_id')!r}"
        )


@pytest.mark.asyncio
async def test_trace_backward_compat_pre_async_format(tmp_path, monkeypatch):
    """Events emitted OUTSIDE a recall_scope must still write — they
    just lack ``recall_id`` / ``seq`` / ``parent_event_id``. This is
    the backwards-compat guarantee for pre-Phase-3 trace consumers."""
    import attestor.trace as tr

    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    tr.reset_for_test()

    # NO recall_scope — bare event call, like pre-Phase-3 code.
    tr.event("legacy.event", foo="bar")

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "legacy.event"
    assert e["foo"] == "bar"
    # event_id is always present (Phase 3 emits it unconditionally).
    assert "event_id" in e
    # recall_id / seq / parent_event_id are absent — pre-Phase-3 shape.
    assert "recall_id" not in e
    assert "seq" not in e
    assert "parent_event_id" not in e
