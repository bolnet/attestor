"""Cross-cutting audit invariants — must hold across every async phase.

These tests pin the audit guarantees described in
``docs/plans/async-retrieval/PLAN.md`` § 2. P1-scope tests are RED today
(target async API doesn't exist); P3+ scope tests are TODO and will be
added when those phases ship.

The eight audit invariants:
  A1: recall(as_of=X) reproducible under concurrent writes
  A2: recall_started_at ceiling — no post-recall writes leak in
  A3: provenance fields immutable per memory_id
  A4: supersession serial per memory_id
  A5: trace events form per-recall reconstructable tree
  A6: RLS propagates through async tasks (current_user_id)
  A7: HyDE/multi-query LLM determinism (temperature=0)
  A8: deletion_audit row written even under async
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ──────────────────────────────────────────────────────────────────────
# A7 — HyDE determinism (P1-scope, RED)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hyde_temperature_zero_determinism_across_runs():
    """Same question → same hypothetical (temp=0) → same RRF order.
    Async amplifies non-determinism risk because gathered lanes can
    observe different hypotheticals across runs if T > 0. This test
    pins the contract: the async generator must call the LLM with
    temperature=0.0 every time.

    We patch the async-client factory rather than ``openai.AsyncOpenAI``
    so the test is independent of which provider the YAML routes to
    (openrouter / openai / anthropic) and bypasses the per-process
    client cache in ``attestor.retrieval.hyde``.
    """
    from attestor.retrieval.hyde import generate_hypothetical_answer_async  # RED

    captured_temps: list = []

    class _StubResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "snippet"})()})()]

    async def fake_create(**kwargs):
        captured_temps.append(kwargs.get("temperature"))
        return _StubResp()

    fake_client = type(
        "FakeAsyncClient",
        (),
        {"chat": type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(fake_create)})()})()},
    )()

    with patch(
        "attestor.retrieval.hyde._get_async_client", return_value=fake_client,
    ):
        for _ in range(3):
            await generate_hypothetical_answer_async("same question", api_key="stub")

    assert captured_temps == [0.0, 0.0, 0.0], (
        f"async generator must call with temperature=0.0 on every invocation; "
        f"observed {captured_temps}"
    )


# ──────────────────────────────────────────────────────────────────────
# A1 — as_of replay under concurrent writes (P5-scope; placeholder asserts
# the contract shape so the test file stays compileable + runnable)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_as_of_replay_under_concurrent_writes_contract():
    """A1 — when ``recall_started_at`` is set, the SQL filter pins
    the snapshot. Even if another coroutine writes a new memory
    *during* the recall, that memory's ``t_created`` is greater than
    the ceiling, so it's excluded by the WHERE clause. The test
    verifies the SQL contract — actual Postgres-side correctness is
    a property of the WHERE clause + Postgres MVCC, both of which
    are well-understood."""
    from datetime import datetime, timezone
    from attestor.recall_context import recall_started_at_scope
    from attestor.store.postgres_backend import PostgresBackend

    sqls: list = []
    params_log: list = []
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = True

    def _capture(sql, params=None):
        sqls.append(sql)
        params_log.append(params)
        return []
    backend._execute = _capture
    backend._embed = lambda text: [0.0] * 1024

    ceiling = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)

    async def reader():
        with recall_started_at_scope(started_at=ceiling):
            backend.search("query")

    # 5 concurrent readers — each must see the same ceiling propagated
    # into its SQL via contextvars.
    await asyncio.gather(reader(), reader(), reader(), reader(), reader())

    assert len(sqls) == 5
    for sql in sqls:
        assert "t_created <= %s" in sql, (
            "every concurrent recall must include the ceiling filter"
        )
    for p in params_log:
        assert ceiling in (p or [])


# ──────────────────────────────────────────────────────────────────────
# A4 — supersession serial per memory_id (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supersession_serial_per_memory_id():
    """Contract assertion (full impl in P5): two concurrent
    supersession attempts on the same memory_id must linearize —
    only one wins, the other's `superseded_by` chain is consistent."""
    pytest.skip(
        "P5 scope — write-side serialization invariant. "
        "See docs/plans/async-retrieval/PLAN.md § 2 — A4."
    )


# ──────────────────────────────────────────────────────────────────────
# A5 — trace events reconstructable tree (P3-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trace_reconstructable_under_async(tmp_path, monkeypatch):
    """A5 — under N concurrent recalls, every event must carry a
    ``recall_id`` and a monotonic ``seq`` so the audit dashboard can
    reconstruct each recall's event tree from the JSONL log even when
    events from different recalls interleave in append-time order."""
    import json
    import attestor.trace as tr

    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log_path))
    tr.reset_for_test()

    N_RECALLS = 10
    EVENTS_PER_RECALL = 4

    async def one_recall(idx: int):
        with tr.recall_scope() as rid:
            tr.event("recall.start", q=f"q{idx}")
            await asyncio.sleep(0.001)  # interleave with other tasks
            tr.event("recall.vector")
            await asyncio.sleep(0.001)
            tr.event("recall.graph")
            tr.event("recall.end")
        return rid

    rids = await asyncio.gather(
        *[one_recall(i) for i in range(N_RECALLS)]
    )

    # Read back the JSONL log.
    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]

    # Reconstruct per-recall.
    by_rid: dict = {}
    for e in events:
        by_rid.setdefault(e.get("recall_id"), []).append(e)

    assert set(rids) <= set(by_rid.keys()), "every recall_id must appear in the log"

    for rid in rids:
        rid_events = by_rid[rid]
        assert len(rid_events) == EVENTS_PER_RECALL, (
            f"recall {rid} should have {EVENTS_PER_RECALL} events, "
            f"got {len(rid_events)}"
        )
        seqs = [e["seq"] for e in rid_events]
        assert seqs == [1, 2, 3, 4], (
            f"recall {rid} seqs must be monotonic 1..4, got {seqs}"
        )
        # All events from one recall share the same recall_id but have
        # distinct event_ids.
        eids = [e["event_id"] for e in rid_events]
        assert len(set(eids)) == EVENTS_PER_RECALL, (
            f"event_ids must be unique within a recall; got dup in {eids}"
        )


# ──────────────────────────────────────────────────────────────────────
# A6 — RLS propagation through async tasks (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rls_propagates_through_async_tasks():
    """A6 — per-recall metadata propagates through asyncio task
    boundaries via ``contextvars.ContextVar`` (which copies the active
    context onto every task spawned via ``asyncio.create_task`` /
    ``asyncio.gather``).

    This test pins the LANGUAGE-LEVEL guarantee: setting a contextvar
    in the parent and reading it in a child task returns the parent's
    value. Both ``recall_started_at`` and the trace ``recall_id`` rely
    on this property; if it ever broke, the audit chain would too.

    The full RLS-on-the-wire test (set ``attestor.current_user_id``
    as a Postgres session-local var, ensure async tasks read the
    parent's value when sharing the connection) requires a live
    Postgres connection and is gated to the integration suite."""
    import contextvars
    from attestor.recall_context import (
        current_recall_started_at, recall_started_at_scope,
    )

    seen: list = []
    user_var: contextvars.ContextVar = contextvars.ContextVar(
        "test.user_id", default=None,
    )

    async def child_task():
        # Both ``recall_started_at`` (Phase 5) and a generic user_id
        # contextvar must propagate to children.
        seen.append((current_recall_started_at(), user_var.get()))

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    user_token = user_var.set("user-aarjay")
    try:
        with recall_started_at_scope(started_at=pinned):
            await asyncio.gather(
                child_task(), child_task(), child_task(),
            )
    finally:
        user_var.reset(user_token)

    assert seen == [
        (pinned, "user-aarjay"),
        (pinned, "user-aarjay"),
        (pinned, "user-aarjay"),
    ], f"both contextvars must propagate to all children; got {seen}"


# ──────────────────────────────────────────────────────────────────────
# A8 — deletion_audit under async (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deletion_audit_under_async():
    """Contract assertion: even if forget() runs async, the
    deletion_audit row gets INSERTed BEFORE the actual delete.
    schema.sql line 304 is the load-bearing comment ('NO RLS on
    deletion_audit') — this test guards against an async refactor
    that loses the pre-delete log."""
    pytest.skip(
        "P5 scope — write-side async (which is non-goal for this plan). "
        "See docs/plans/async-retrieval/PLAN.md § 1 — non-goals."
    )
