"""Shared test fixtures — AgentMemory now requires live Postgres + Neo4j.

Without `POSTGRES_URL` (and optionally `NEO4J_URI`) set, the `mem` fixture
skips. Pure unit-level tests that don't build an AgentMemory instance keep
running on any dev machine.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest

from attestor import AgentMemory


# Minimal config — no external services needed for bookkeeping.
TEST_CONFIG = {
    "default_token_budget": 2000,
    "min_results": 3,
}


def _has_live_backends() -> bool:
    """Return True when env exposes at least a Postgres URL."""
    return bool(os.environ.get("POSTGRES_URL"))


@pytest.fixture
def test_config() -> dict:
    """Config dict for tests."""
    return dict(TEST_CONFIG)


@pytest.fixture
def mem_dir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mem(mem_dir: str) -> Iterator[AgentMemory]:
    if not _has_live_backends():
        pytest.skip(
            "requires a live Postgres backend (set POSTGRES_URL) — "
            "Attestor no longer ships an embedded zero-config stack"
        )
    m = AgentMemory(mem_dir, config=TEST_CONFIG)
    try:
        yield m
    finally:
        m.close()
