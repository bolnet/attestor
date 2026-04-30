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
import os
from unittest.mock import AsyncMock, patch

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ──────────────────────────────────────────────────────────────────────
# A7 — HyDE determinism (P1-scope, RED)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
async def test_hyde_temperature_zero_determinism_across_runs():
    """Same question → same hypothetical (temp=0) → same RRF order.
    Async amplifies non-determinism risk because gathered lanes can
    observe different hypotheticals across runs if T > 0. This test
    pins the contract: the async generator must call the LLM with
    temperature=0.0 every time."""
    from attestor.retrieval.hyde import generate_hypothetical_answer_async  # RED

    captured_temps = []

    class _StubResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "snippet"})()})()]

    async def fake_create(**kwargs):
        captured_temps.append(kwargs.get("temperature"))
        return _StubResp()

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "stub"}):
        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value.chat.completions.create = fake_create
            for _ in range(3):
                await generate_hypothetical_answer_async("same question")

    assert captured_temps == [0.0, 0.0, 0.0], (
        f"async generator must call with temperature=0.0 on every invocation; "
        f"observed {captured_temps}"
    )


# ──────────────────────────────────────────────────────────────────────
# A1 — as_of replay under concurrent writes (P5-scope; placeholder asserts
# the contract shape so the test file stays compileable + runnable)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
async def test_as_of_replay_under_concurrent_writes_contract():
    """Contract assertion (full implementation lands in P5):

       Hammering recall(as_of=X) while another coroutine writes new
       memories must produce a stable result — no half-visible writes.

    This test is currently a placeholder pinning the signature of the
    expected async API. It will be expanded in PR-5 with a real
    Postgres+Pinecone+Neo4j fixture and a writer-coroutine that runs
    in parallel with K=20 reader recalls.
    """
    pytest.skip(
        "P5 scope — implementation lands with recall_started_at ceiling. "
        "See docs/plans/async-retrieval/PLAN.md § 4 — Phase 5."
    )


# ──────────────────────────────────────────────────────────────────────
# A4 — supersession serial per memory_id (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
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
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
async def test_trace_reconstructable_under_async():
    """Contract assertion (full impl in P3): 10 concurrent recalls
    emit interleaved trace events; per-recall tree must reconstruct
    via (recall_id, parent_event_id, seq)."""
    pytest.skip(
        "P3 scope — trace schema bump. "
        "See docs/plans/async-retrieval/PLAN.md § 4 — Phase 3."
    )


# ──────────────────────────────────────────────────────────────────────
# A6 — RLS propagation through async tasks (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
async def test_rls_propagates_through_async_tasks():
    """Contract assertion (full impl in P5): each async task spawned
    inside a recall sees its parent's `attestor.current_user_id`
    session-local var. A task running with the wrong user_id would
    silently bypass tenant isolation — defense-in-depth would
    collapse."""
    pytest.skip(
        "P5 scope — connection-pool + session-local context propagation. "
        "See docs/plans/async-retrieval/PLAN.md § 2 — A6."
    )


# ──────────────────────────────────────────────────────────────────────
# A8 — deletion_audit under async (P5-scope, placeholder)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(reason="RED — async impl lands in next PR; remove when GREEN", strict=False)
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
