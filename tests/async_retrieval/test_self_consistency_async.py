"""Phase 4 — async self-consistency K-fanout tests.

Target API:

    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    result = await answer_with_self_consistency_async(
        client=async_client,
        model="...",
        messages=[...],
        k=5,
        temperature=0.7,
    )

The async version must:
  1. Run K answerer samples CONCURRENTLY via asyncio.gather; wallclock
     ≈ max(per-sample), NOT K × per-sample.
  2. Produce the same elected ``chosen`` answer as the sync version
     given identical samples — voter logic is order-independent.
  3. Survive K-1 sample failures: the one surviving sample becomes
     the consensus.

See docs/plans/async-retrieval/PLAN.md § 4 — Phase 4.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = [pytest.mark.unit]


def _make_async_client(per_call_ms: int, content: str) -> MagicMock:
    """Build an OpenAI-compatible async client whose
    ``chat.completions.create`` sleeps then returns a fixed message
    content. Each invocation sleeps independently, so we can observe
    parallel vs sequential by total wallclock."""
    client = MagicMock()

    async def fake_create(**kwargs):
        await asyncio.sleep(per_call_ms / 1000.0)
        return MagicMock(
            choices=[MagicMock(
                message=MagicMock(content=content),
            )]
        )

    client.chat.completions.create = fake_create
    return client


def _make_async_client_with_failures(
    per_call_ms: int, success_content: str, fail_after: int,
) -> MagicMock:
    """Build an async client where the first ``fail_after`` calls raise
    and subsequent calls succeed. Used to test K-1 failure tolerance.
    """
    client = MagicMock()
    counter = {"n": 0}

    async def fake_create(**kwargs):
        await asyncio.sleep(per_call_ms / 1000.0)
        counter["n"] += 1
        if counter["n"] <= fail_after:
            raise RuntimeError(f"intentional failure at call {counter['n']}")
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=success_content))]
        )

    client.chat.completions.create = fake_create
    return client


# ──────────────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_self_consistency_async_runs_K_concurrently():
    """K=5 samples each sleeping 200ms must complete in ~250ms total
    (parallel), NOT 1000ms (sequential)."""
    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    client = _make_async_client(per_call_ms=200, content="Bob")
    messages = [{"role": "user", "content": "Who is the CTO?"}]

    t0 = time.perf_counter()
    result = await answer_with_self_consistency_async(
        client=client, model="stub", messages=messages, k=5, temperature=0.7,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 600, (
        f"K=5 ran sequentially? wallclock={elapsed_ms:.0f}ms, "
        f"expected < 600ms with gather"
    )
    assert len(result.samples) == 5
    assert result.chosen == "Bob"


# ──────────────────────────────────────────────────────────────────────
# Voter determinism — async order-independent
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_self_consistency_async_consensus_order_independent():
    """Async gather order does NOT change which sample wins. Even when
    samples complete in shuffled order, the majority bucket is the same.

    This pins audit invariant: the voter (`_majority_choice`) is
    commutative — it counts fingerprints, then picks max with first-seen
    tiebreak. First-seen is in the order samples appear in the list
    after gather, but since gather preserves submission order in its
    return list, the bucket counts are deterministic regardless of
    completion timing.
    """
    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    client = MagicMock()
    # Cycle through 3 distinct answers; majority should still pick the
    # most-frequent one regardless of arrival order.
    sequence = ["Alice", "Bob", "Bob", "Bob", "Carol"]
    counter = {"n": 0}

    async def fake_create(**kwargs):
        i = counter["n"]
        counter["n"] += 1
        # Vary the per-call sleep so gather completion order is non-monotonic.
        sleeps_ms = [50, 10, 30, 5, 40]
        await asyncio.sleep(sleeps_ms[i % len(sleeps_ms)] / 1000.0)
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=sequence[i]))]
        )

    client.chat.completions.create = fake_create

    result = await answer_with_self_consistency_async(
        client=client, model="stub",
        messages=[{"role": "user", "content": "?"}],
        k=5, temperature=0.7,
    )

    assert result.chosen == "Bob", (
        f"majority winner must be 'Bob' (3/5), got {result.chosen!r}; "
        f"breakdown={result.vote_breakdown}"
    )
    assert result.vote_breakdown.get("bob") == 3


# ──────────────────────────────────────────────────────────────────────
# Failure tolerance
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_self_consistency_async_handles_K_minus_one_failures():
    """K=5 with the first 4 calls failing — the surviving 1 sample
    becomes the consensus (single-vote winner)."""
    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    client = _make_async_client_with_failures(
        per_call_ms=20, success_content="lone_survivor", fail_after=4,
    )

    result = await answer_with_self_consistency_async(
        client=client, model="stub",
        messages=[{"role": "user", "content": "?"}],
        k=5, temperature=0.7,
    )

    assert len(result.samples) == 1
    assert result.chosen == "lone_survivor"


@pytest.mark.asyncio
async def test_self_consistency_async_handles_all_failures():
    """If ALL K calls fail, the result must be empty — no chosen,
    voter degraded gracefully."""
    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    client = _make_async_client_with_failures(
        per_call_ms=10, success_content="never_returned", fail_after=999,
    )

    result = await answer_with_self_consistency_async(
        client=client, model="stub",
        messages=[{"role": "user", "content": "?"}],
        k=5, temperature=0.7,
    )

    assert result.samples == []
    assert result.chosen == ""


@pytest.mark.asyncio
async def test_self_consistency_async_k_zero_returns_empty():
    """K=0 must short-circuit — no LLM calls, empty result."""
    from attestor.longmemeval_consistency import answer_with_self_consistency_async

    client = MagicMock()
    client.chat.completions.create = AsyncMock()

    result = await answer_with_self_consistency_async(
        client=client, model="stub",
        messages=[{"role": "user", "content": "?"}],
        k=0, temperature=0.7,
    )

    assert result.samples == []
    assert result.chosen == ""
    client.chat.completions.create.assert_not_called()
