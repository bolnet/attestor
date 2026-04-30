"""Phase 1 — RED tests for ``hyde_search_async``.

Target API (does NOT exist yet — these tests will fail until PR-1 lands):

    from attestor.retrieval.hyde import hyde_search_async

    queries, merged = await hyde_search_async(
        question,
        vector_search=async_callable,   # Awaitable[List[Dict]]
        merge="rrf",
    )

The async version must:
  1. Run the HyDE LLM call concurrently with the original-question
     vector embed (via ``asyncio.gather``).
  2. Produce the same merged hits as the sync ``hyde_search`` given
     identical inputs (including a deterministic / temperature=0 LLM).
  3. Degrade gracefully on lane timeout or exception
     (``return_exceptions=True``).
  4. Pin ``temperature=0.0`` on the generator call (same invariant as
     HyDE v2 sync path).

See ``docs/plans/async-retrieval/PLAN.md`` § 4 — Phase 1 row.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

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
async def test_hyde_search_async_runs_lanes_concurrently():
    """Two awaitables (HyDE LLM + per-lane vector searches) must run
    via ``asyncio.gather``. With both lanes sleeping 200ms, total
    wallclock should be < 350ms (200 + ε), NOT 400ms (sequential).
    """
    from attestor.retrieval.hyde import hyde_search_async  # noqa: F401 — RED

    slow_search = make_slow_search(per_call_ms=200)

    # Mock generator so we don't make a real LLM call. The mock itself
    # sleeps 200ms to simulate the LLM round-trip — this is the call
    # that should overlap with the per-lane vector searches.
    async def slow_generator(question, **kwargs):
        await asyncio.sleep(0.2)
        from attestor.retrieval.hyde import HydeResult
        return HydeResult(
            original_question=question,
            hypothetical_answer="hypothetical narrative content",
        )

    with patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer_async",
        side_effect=slow_generator,
    ):
        async with StopwatchAsync() as sw:
            queries, merged = await hyde_search_async(
                "test question",
                vector_search=slow_search,
            )

    # Sequential would be ~600ms (LLM 200 + lane1 200 + lane2 200).
    # Parallel HyDE-with-2-lanes should be ~400ms (LLM 200 ‖ embed,
    # then 2 vector searches gathered = max 200).
    assert sw.elapsed_ms < 500, (
        f"HyDE+lanes ran sequentially? wallclock={sw.elapsed_ms:.0f}ms, "
        f"expected < 500ms with gather"
    )
    assert len(queries) == 2, "queries should be [original, hypothetical]"
    assert len(merged) >= 1, "RRF merge should produce at least one hit"


@pytest.mark.asyncio
async def test_hyde_search_async_returns_same_result_as_sync():
    """Async path is purely a latency optimization — given identical
    inputs (deterministic LLM, deterministic vector_search), the
    merged-hits ordering must match the sync version byte-for-byte.
    """
    from attestor.retrieval.hyde import hyde_search, hyde_search_async  # noqa: F401 — RED

    fixed_hits = [
        {"memory_id": "a", "session_id": "s1", "distance": 0.1},
        {"memory_id": "b", "session_id": "s2", "distance": 0.2},
    ]

    def sync_search(q: str):
        return fixed_hits

    async def async_search(q: str):
        return fixed_hits

    with patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer",
        return_value=__import__(
            "attestor.retrieval.hyde", fromlist=["HydeResult"]
        ).HydeResult(
            original_question="q", hypothetical_answer="snippet",
        ),
    ), patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer_async",
        new=AsyncMock(
            return_value=__import__(
                "attestor.retrieval.hyde", fromlist=["HydeResult"]
            ).HydeResult(
                original_question="q", hypothetical_answer="snippet",
            ),
        ),
    ):
        sync_queries, sync_merged = hyde_search("q", vector_search=sync_search)
        async_queries, async_merged = await hyde_search_async(
            "q", vector_search=async_search,
        )

    assert sync_queries == async_queries
    assert [h["memory_id"] for h in sync_merged] == [
        h["memory_id"] for h in async_merged
    ], "RRF merge order must match between sync and async"


# ──────────────────────────────────────────────────────────────────────
# Failure / degradation behavior
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hyde_search_async_handles_lane_timeout():
    """If the HyDE LLM exceeds its timeout, the original-question lane
    must still produce hits — fall back to single-lane recall, do NOT
    raise to the caller."""
    from attestor.retrieval.hyde import hyde_search_async  # RED

    fast_search = make_slow_search(per_call_ms=10)

    async def hangs_forever(question, **kwargs):
        await asyncio.sleep(60)  # would exceed any sane timeout
        from attestor.retrieval.hyde import HydeResult
        return HydeResult(original_question=question, hypothetical_answer="late")

    with patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer_async",
        side_effect=hangs_forever,
    ):
        async with StopwatchAsync() as sw:
            queries, merged = await hyde_search_async(
                "q", vector_search=fast_search, timeout=0.5,
            )

    # Timeout should fire well before the LLM "completes". Caller still
    # gets hits — just from the single original-question lane.
    assert sw.elapsed_ms < 1500, (
        f"HyDE timeout did not fire — wallclock={sw.elapsed_ms:.0f}ms"
    )
    assert len(queries) >= 1, "must have at least the original query"
    assert len(merged) >= 1, "must have at least one hit from the surviving lane"


@pytest.mark.asyncio
async def test_hyde_search_async_handles_lane_exception():
    """If one of the gathered awaitables raises, the others must keep
    running — gather(..., return_exceptions=True) is the contract."""
    from attestor.retrieval.hyde import hyde_search_async  # RED

    failing_search = make_failing_search(when="second_lane")

    async def normal_generator(question, **kwargs):
        from attestor.retrieval.hyde import HydeResult
        return HydeResult(original_question=question, hypothetical_answer="x")

    with patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer_async",
        side_effect=normal_generator,
    ):
        # Should NOT raise — exception in the second lane is swallowed
        # and merge proceeds with the first lane only.
        queries, merged = await hyde_search_async("q", vector_search=failing_search)

    assert len(queries) == 2
    # The first lane succeeded → at least one hit.
    assert len(merged) >= 1


# ──────────────────────────────────────────────────────────────────────
# Audit invariant — temperature=0 still pinned in the async path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hyde_async_preserves_temperature_zero():
    """The async generator MUST still pass temperature=0.0 to the LLM
    client. This is the determinism guarantee from HyDE v2 — without
    it, async amplifies non-determinism risk because gathered lanes
    could observe different hypothetical answers across runs."""
    from attestor.retrieval.hyde import generate_hypothetical_answer_async  # RED

    captured_kwargs = {}

    class _StubResponse:
        choices = [type("C", (), {"message": type("M", (), {"content": "snippet"})()})()]

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return _StubResponse()

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "stub"}):
        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value.chat.completions.create = fake_create
            await generate_hypothetical_answer_async("q")

    assert captured_kwargs.get("temperature") == 0.0, (
        f"async HyDE generator must pin temperature=0.0; "
        f"got {captured_kwargs.get('temperature')!r}"
    )
