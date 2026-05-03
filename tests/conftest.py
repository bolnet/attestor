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


def _build_test_config() -> dict:
    """TEST_CONFIG with backend_configs derived from POSTGRES_URL when set.

    Without this, the postgres backend falls back to ``ENGINE_DEFAULTS``
    (empty password) and every live-backed test errors with
    ``no password supplied`` — even when ``POSTGRES_URL`` is set.
    """
    cfg = dict(TEST_CONFIG)
    pg_url = os.environ.get("POSTGRES_URL")
    if pg_url:
        from urllib.parse import urlparse
        parsed = urlparse(pg_url)
        cfg["backend_configs"] = {
            "postgres": {
                "url": (
                    f"postgresql://{parsed.hostname or 'localhost'}"
                    f":{parsed.port or 5432}"
                ),
                "database": (parsed.path or "/attestor").lstrip("/") or "attestor",
                "auth": {
                    "username": parsed.username or "postgres",
                    "password": parsed.password or "",
                },
            }
        }
        # Skip the embedder probe when no embedder env is set so the
        # contradiction/temporal tests don't need an LLM key just to boot.
        if not (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("VOYAGE_API_KEY")
            or os.environ.get("PINECONE_API_KEY")
        ):
            cfg["backend_configs"]["postgres"]["embedding_dim"] = 1024
    return cfg


@pytest.fixture
def test_config() -> dict:
    """Config dict for tests."""
    return _build_test_config()


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
    m = AgentMemory(mem_dir, config=_build_test_config())
    # Reset row state so tests in the same DB don't pollute each other.
    # The schema lives across tests (cheap), but row state must be clean.
    try:
        m._store.execute("TRUNCATE memories CASCADE")
    except Exception:  # pragma: no cover - schema not yet bootstrapped
        pass
    try:
        yield m
    finally:
        m.close()
