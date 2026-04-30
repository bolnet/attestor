"""Shared fixtures for async retrieval tests.

These tests are RED today — they target async API surfaces that don't
exist yet (``hyde_search_async``, ``multi_query_search_async``, etc.).
The matching code lands in subsequent PRs per ``docs/plans/async-retrieval/PLAN.md``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List

import pytest


@pytest.fixture
def event_loop():
    """One event loop per test — keeps tests isolated."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def make_slow_search(per_call_ms: int) -> Callable[[str], Awaitable[List[Dict]]]:
    """Build an async vector_search closure that sleeps per_call_ms then
    returns a single hit keyed on the query string. Used to assert lanes
    actually overlap in time — if they ran sequentially, total wallclock
    would be N × per_call_ms; if parallel, it's ~per_call_ms + ε.
    """
    async def search(q: str) -> List[Dict]:
        await asyncio.sleep(per_call_ms / 1000.0)
        return [
            {
                "memory_id": f"mem-{abs(hash(q)) % 10**8}",
                "session_id": f"sess-{abs(hash(q)) % 10**6}",
                "distance": 0.1,
            }
        ]
    return search


def make_failing_search(when: str = "always") -> Callable[[str], Awaitable[List[Dict]]]:
    """Build a vector_search that raises. ``when`` selects:
        - 'always'        — every call raises
        - 'second_lane'   — second call onwards raises
    """
    counter = {"n": 0}

    async def search(q: str) -> List[Dict]:
        counter["n"] += 1
        if when == "always" or (when == "second_lane" and counter["n"] >= 2):
            raise RuntimeError(f"intentional failure at call {counter['n']} for q={q!r}")
        return [
            {
                "memory_id": f"mem-{abs(hash(q)) % 10**8}",
                "session_id": "ok",
                "distance": 0.2,
            }
        ]
    return search


class StopwatchAsync:
    """Async context manager that records elapsed wallclock in ms."""
    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0

    async def __aenter__(self) -> "StopwatchAsync":
        self._t0 = time.perf_counter()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
