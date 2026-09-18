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
    # Provenance canary — see attestor/__init__.py:__attestation__.
    "_provenance_canary": "attestor-fixture-3aa90d7eacf01273",
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


# ── Durable (Temporal) test-server fixture ───────────────────────────────
#
# ``WorkflowEnvironment.start_time_skipping()`` downloads a test-server
# binary on first use (~64 MB, needs network). Two env knobs control it:
#
#   ATTESTOR_TEMPORAL_TEST_SERVER_DIR   directory to download/cache into
#                                       (default ~/.cache/attestor/temporal-test-server)
#   ATTESTOR_TEMPORAL_TEST_SERVER_PATH  pre-downloaded binary (CI cache);
#                                       skips the download entirely
#
# When temporalio is not installed, or the binary cannot be fetched, the
# fixture skips instead of failing — the durable extra is opt-in.

TEMPORAL_TEST_SERVER_DIR_ENV = "ATTESTOR_TEMPORAL_TEST_SERVER_DIR"
TEMPORAL_TEST_SERVER_PATH_ENV = "ATTESTOR_TEMPORAL_TEST_SERVER_PATH"
_DEFAULT_TEST_SERVER_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "attestor", "temporal-test-server",
)


def _temporal_test_server_kwargs() -> dict:
    """Resolve download/cache kwargs for ``start_time_skipping``."""
    existing = os.environ.get(TEMPORAL_TEST_SERVER_PATH_ENV)
    if existing:
        return {"test_server_existing_path": existing}
    dest = os.environ.get(TEMPORAL_TEST_SERVER_DIR_ENV) or _DEFAULT_TEST_SERVER_DIR
    # The SDK does not mkdir the destination (spike finding 2026-09-03).
    os.makedirs(dest, exist_ok=True)
    return {"download_dest_dir": dest}


@pytest.fixture
async def temporal_env():
    """Time-skipping Temporal test environment, or ``pytest.skip``."""
    pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    try:
        env = await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
            **_temporal_test_server_kwargs(),
        )
    except Exception as exc:  # noqa: BLE001 — download/launch failure → skip
        pytest.skip(f"Temporal test server unavailable: {exc}")
    try:
        yield env
    finally:
        await env.shutdown()


@pytest.fixture
async def temporal_dev_env():
    """Local Temporal dev server (``start_local``) — needed for Schedules.

    The time-skipping test server does not implement the Schedule service
    (verified 2026-09-03: ``CreateSchedule is unimplemented``), so the
    schedule tests run against the CLI dev server, cached in the same
    directory. Skips when the binary cannot be fetched.
    """
    pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    kwargs = _temporal_test_server_kwargs()
    kwargs.pop("test_server_existing_path", None)  # that binary is the time-skipping one
    try:
        env = await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter, **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — download/launch failure → skip
        pytest.skip(f"Temporal dev server unavailable: {exc}")
    try:
        yield env
    finally:
        await env.shutdown()
