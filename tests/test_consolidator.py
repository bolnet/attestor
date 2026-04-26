"""Phase 7.2 — SleepTimeConsolidator end-to-end with stubbed LLMs.

Tests run against the live v4 schema. LLM calls go through scripted
stubs so the tests are deterministic + don't burn local Ollama time.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

PG_URL = os.environ.get(
    "PG_TEST_URL",
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "attestor" / "store" / "schema.sql"


def _reachable() -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable")


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM (scripted per call, in order)
# ──────────────────────────────────────────────────────────────────────────


class _Choice:
    def __init__(self, c: str) -> None:
        self.message = type("M", (), {"content": c})()


class _Resp:
    def __init__(self, c: str) -> None:
        self.choices = [_Choice(c)]


class _Chat:
    def __init__(self, scripts: List[str]) -> None:
        self._scripts = list(scripts)
        self.calls: List[dict] = []

    def create(self, **kw: Any) -> _Resp:
        self.calls.append(kw)
        if not self._scripts:
            return _Resp('{"facts": [], "decisions": []}')
        return _Resp(self._scripts.pop(0))


class ScriptedClient:
    def __init__(self, scripts: List[str]) -> None:
        self.chat = type("Chat", (), {"completions": _Chat(scripts)})()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def admin_conn():
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture
def mem_with_episodes(admin_conn, fresh_schema, tmp_path):
    """SOLO AgentMemory with 3 freshly-ingested episodes pending consolidation."""
    from attestor.conversation import ConversationTurn
    from attestor.core import AgentMemory

    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True, "embedding_dim": 16,
        }},
    })

    # Use a stub for the synchronous extraction so ingest_round doesn't
    # actually call the LLM during fixture setup.
    sync_client = ScriptedClient([
        # Per round: user-extract, agent-extract (resolver short-circuits
        # because no existing memories yet on round 1, then also for the
        # rest since synchronous facts are minimal).
        '{"facts": []}', '{"facts": []}',
        '{"facts": []}', '{"facts": []}',
        '{"facts": []}', '{"facts": []}',
    ])
    base = datetime.now(timezone.utc)
    eps = []
    for i in range(3):
        u = ConversationTurn(
            thread_id="t-1", speaker="user", role="user",
            content=f"user message {i}", ts=base + timedelta(seconds=i * 10),
        )
        a = ConversationTurn(
            thread_id="t-1", speaker="planner", role="assistant",
            content=f"agent response {i}",
            ts=base + timedelta(seconds=i * 10 + 1),
        )
        r = mem.ingest_round(u, a,
                             extraction_client=sync_client,
                             resolver_client=sync_client)
        eps.append(r.episode)
    yield mem, eps
    mem.close()


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_run_once_drains_pending_episodes(mem_with_episodes):
    from attestor.consolidation import SleepTimeConsolidator

    mem, eps = mem_with_episodes
    # Each episode triggers: user-extract, agent-extract, resolve = 3 calls.
    # 3 episodes * 3 = up to 9 calls, but resolver short-circuits when
    # existing=[] (no LLM call). Provide enough scripts.
    client = ScriptedClient([
        '{"facts": [{"text": "fact A", "category": "preference", '
        '"entity": "user", "confidence": 0.9, "source_span": [0, 5]}]}',
        '{"facts": []}',
        '{"facts": [{"text": "fact B", "category": "preference", '
        '"entity": "user", "confidence": 0.9, "source_span": [0, 5]}]}',
        '{"facts": []}',
        '{"facts": [{"text": "fact C", "category": "preference", '
        '"entity": "user", "confidence": 0.9, "source_span": [0, 5]}]}',
        '{"facts": []}',
    ])

    cons = SleepTimeConsolidator(
        mem, model="test-model",
        extraction_client=client, resolver_client=client,
    )
    results = cons.run_once()
    assert len(results) == 3
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]


@pytest.mark.live
def test_run_once_marks_done(mem_with_episodes):
    from attestor.consolidation import (
        ConsolidationQueue,
        SleepTimeConsolidator,
    )

    mem, eps = mem_with_episodes
    client = ScriptedClient(['{"facts": []}'] * 6)
    cons = SleepTimeConsolidator(
        mem, extraction_client=client, resolver_client=client,
    )
    cons.run_once()
    q = ConsolidationQueue(mem._store._conn)
    s = q.stats()
    assert s["done"] == 3
    assert s["pending"] == 0
    assert s["processing"] == 0


@pytest.mark.live
def test_consolidator_writes_facts_with_consolidation_provenance(mem_with_episodes):
    from attestor.consolidation import SleepTimeConsolidator

    mem, eps = mem_with_episodes
    client = ScriptedClient([
        '{"facts": [{"text": "consolidated fact", "category": "preference", '
        '"entity": "user", "confidence": 0.9, "source_span": [0, 5]}]}',
        '{"facts": []}',
        '{"facts": []}', '{"facts": []}',
        '{"facts": []}', '{"facts": []}',
    ])
    cons = SleepTimeConsolidator(
        mem, model="test-strong-model",
        extraction_client=client, resolver_client=client,
    )
    results = cons.run_once()
    written = [
        mid for r in results for mid in r.written_memory_ids
    ]
    assert len(written) >= 1
    for mid in written:
        row = mem._store.get(mid)
        assert row is not None
        # provenance prefix tags this as a consolidator write
        assert row.extraction_model.startswith("consolidation:"), (
            f"expected consolidation provenance, got {row.extraction_model!r}"
        )


@pytest.mark.live
def test_consolidator_marks_failed_on_extraction_error(mem_with_episodes):
    """If the extraction client raises, the row is marked failed (not stuck)."""
    from attestor.consolidation import (
        ConsolidationQueue,
        SleepTimeConsolidator,
    )

    mem, eps = mem_with_episodes

    class ExplodingClient:
        class _Chat:
            def create(self, **kw):
                raise RuntimeError("model down")
        chat = type("C", (), {"completions": _Chat()})()

    cons = SleepTimeConsolidator(
        mem, extraction_client=ExplodingClient(),
        resolver_client=ExplodingClient(),
    )
    # The extractor's _call_llm doesn't catch; the exception bubbles up
    # to _consolidate_one's outer try/except, which marks the row failed
    # (with the error string). All 3 episodes should land in 'failed'.
    results = cons.run_once()
    assert all(not r.ok for r in results)
    assert all("model down" in (r.error or "") for r in results)
    q = ConsolidationQueue(mem._store._conn)
    s = q.stats()
    assert s["failed"] == 3
    assert s["done"] == 0
    assert s["pending"] == 0


@pytest.mark.live
def test_run_once_returns_empty_when_queue_drained(mem_with_episodes):
    from attestor.consolidation import (
        ConsolidationQueue,
        SleepTimeConsolidator,
    )

    mem, eps = mem_with_episodes
    client = ScriptedClient(['{"facts": []}'] * 6)
    cons = SleepTimeConsolidator(
        mem, extraction_client=client, resolver_client=client,
    )
    cons.run_once()
    # Second pass — nothing left
    assert cons.run_once() == []
