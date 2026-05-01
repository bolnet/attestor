"""Performance regression guards for the async retrieval refactor.

Two concrete guards:

1. ``test_async_overhead_under_50ms_no_op_recall`` — bare async tax on
   a recall doing no real work must stay under 50ms. Catches event-loop
   bootstrap or context-switch regressions early.

2. ``test_sync_recall_shim_works`` — embedded users without an event
   loop must still be able to call the sync shim. The shim wraps
   ``asyncio.run`` and must produce identical results.

P1-scope tests RED today — target async API doesn't exist.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


pytestmark = [pytest.mark.unit]  # asyncio_mode=auto handles async tests; sync test below is skipped


# ──────────────────────────────────────────────────────────────────────
# Async overhead budget
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_overhead_under_50ms_no_op_recall():
    """``hyde_search_async`` with both lanes returning instantly must
    complete in < 50ms wallclock. Any regression here means the async
    layer itself (event loop bootstrap, gather overhead, context
    switches) has gotten heavy — investigate before shipping.
    """
    from attestor.retrieval.hyde import hyde_search_async  # RED

    async def instant_search(q: str):
        return [{"memory_id": "m", "session_id": "s", "distance": 0.1}]

    async def instant_generator(question, **kwargs):
        from attestor.retrieval.hyde import HydeResult
        return HydeResult(original_question=question, hypothetical_answer="x")

    with patch(
        "attestor.retrieval.hyde.generate_hypothetical_answer_async",
        side_effect=instant_generator,
    ):
        t0 = time.perf_counter()
        await hyde_search_async("q", vector_search=instant_search)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50, (
        f"async overhead regression: no-op hyde_search_async took "
        f"{elapsed_ms:.1f}ms; budget is 50ms"
    )


# ──────────────────────────────────────────────────────────────────────
# Sync shim contract
# ──────────────────────────────────────────────────────────────────────


def test_sync_recall_shim_works():
    """``hyde_search`` (sync) must remain a viable entrypoint for
    callers without an event loop — e.g. pytest sync tests, the CLI,
    one-off scripts. After the async refactor, sync may internally
    delegate to ``asyncio.run(hyde_search_async(...))`` — but the
    public sync API must not require the caller to know that.
    """
    from attestor.retrieval.hyde import hyde_search

    def sync_search(q: str):
        return [{"memory_id": "x", "session_id": "y", "distance": 0.0}]

    # Sync API exists today and must keep working post-async.
    queries, merged = hyde_search(
        "q", vector_search=sync_search,
    )
    # No assertion on content here — that's covered by the existing
    # tests/test_hyde.py suite. We're only pinning the SHAPE: sync
    # callable, returns (List[str], List[Dict]).
    assert isinstance(queries, list)
    assert isinstance(merged, list)
