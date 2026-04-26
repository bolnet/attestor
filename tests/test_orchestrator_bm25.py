"""Phase 4.3 — RetrievalOrchestrator wiring of the BM25 lane.

Integration tests with a stub vector store + a real BM25Lane (live
Postgres). Verifies:
  - Vector-only path: orchestrator works with bm25_lane=None (regression)
  - BM25-only path: orchestrator returns BM25 hits when vector_store=None
  - Fusion: a memory matched only by BM25 keyword (not similar vectorally)
    appears in the result set when both lanes are wired
  - Backward compat: AgentMemory's __init__ wires bm25_lane on v4
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

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
def seeded(admin_conn, fresh_schema):
    """Seed 4 memories with distinct keyword content; return ids + conn."""
    uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-orc-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "SELECT set_config('attestor.current_user_id', %s, false)",
            (str(uid),),
        )
        ids = []
        seeds = [
            ("user prefers dark mode", ["preference"]),
            ("user lives in San Francisco", ["location"]),
            ("user's lucky number is 7919", ["personal"]),
            ("user attended PyCon last year", ["event"]),
        ]
        for content, tags in seeds:
            cur.execute(
                "INSERT INTO memories (user_id, content, tags) "
                "VALUES (%s, %s, %s) RETURNING id",
                (str(uid), content, tags),
            )
            ids.append(str(cur.fetchone()[0]))
    return str(uid), ids, admin_conn


# ──────────────────────────────────────────────────────────────────────────
# Stub vector store — controllable hits per test
# ──────────────────────────────────────────────────────────────────────────


class StubVectorStore:
    def __init__(self, hits=None) -> None:
        self._hits = hits or []
        self.search_calls: list = []

    def search(self, query: str, *, limit: int = 20, namespace=None) -> list:
        self.search_calls.append({"query": query, "limit": limit})
        return self._hits


class StubDocStore:
    """Lightweight doc store that proxies .get to a real psycopg2 conn."""
    def __init__(self, conn) -> None:
        self._conn = conn

    def get(self, mid: str):
        from attestor.models import Memory
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM memories WHERE id = %s", (mid,))
            row = cur.fetchone()
        if row is None:
            return None
        return Memory.from_row(dict(row))


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator paths
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_orchestrator_without_bm25_unchanged(seeded) -> None:
    """No bm25_lane → behaves exactly like the pre-Phase-4 vector path."""
    import psycopg2.extras  # for RealDictCursor in StubDocStore.get
    uid, ids, conn = seeded
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    vec = StubVectorStore(hits=[
        {"memory_id": ids[0], "distance": 0.1},
        {"memory_id": ids[1], "distance": 0.4},
    ])
    orch = RetrievalOrchestrator(
        store=StubDocStore(conn), vector_store=vec, bm25_lane=None,
    )
    out = orch.recall("dark", token_budget=2000)
    out_ids = [r.memory.id for r in out if r.match_source == "vector"]
    assert ids[0] in out_ids


@pytest.mark.live
def test_orchestrator_bm25_only_returns_keyword_hits(seeded) -> None:
    """vector_store=None + bm25_lane wired → keyword hits flow through."""
    import psycopg2.extras
    uid, ids, conn = seeded
    from attestor.retrieval.bm25 import BM25Lane
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    bm25 = BM25Lane(conn)
    orch = RetrievalOrchestrator(
        store=StubDocStore(conn), vector_store=None, bm25_lane=bm25,
    )
    out = orch.recall("PyCon", token_budget=2000)
    out_contents = [r.memory.content for r in out]
    assert any("PyCon" in c for c in out_contents)


@pytest.mark.live
def test_orchestrator_fusion_surfaces_bm25_only_match(seeded) -> None:
    """Vector lane misses 'lucky number 7919' but BM25 finds it."""
    import psycopg2.extras
    uid, ids, conn = seeded
    from attestor.retrieval.bm25 import BM25Lane
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    # Vector lane returns the dark-mode row (most similar embedding-wise
    # for a vague query) but NOT the lucky-number row.
    vec = StubVectorStore(hits=[
        {"memory_id": ids[0], "distance": 0.3},
    ])
    bm25 = BM25Lane(conn)
    orch = RetrievalOrchestrator(
        store=StubDocStore(conn), vector_store=vec, bm25_lane=bm25,
    )
    out = orch.recall("7919", token_budget=2000)
    out_ids = [r.memory.id for r in out]
    # The lucky-number row should appear via the BM25 lane
    lucky_id = ids[2]
    assert lucky_id in out_ids


@pytest.mark.live
def test_orchestrator_bm25_failure_does_not_break_recall(seeded) -> None:
    """BM25 lane raising → recall still returns vector hits."""
    import psycopg2.extras
    uid, ids, conn = seeded
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    class ExplodingBM25:
        def search(self, *a, **kw):
            raise RuntimeError("bm25 down")

    vec = StubVectorStore(hits=[{"memory_id": ids[0], "distance": 0.1}])
    orch = RetrievalOrchestrator(
        store=StubDocStore(conn), vector_store=vec,
        bm25_lane=ExplodingBM25(),
    )
    out = orch.recall("anything", token_budget=2000)
    assert len(out) >= 1


@pytest.mark.live
def test_agent_memory_wires_bm25_on_v4(seeded) -> None:
    """AgentMemory(v4=True) auto-attaches a BM25Lane to the orchestrator."""
    uid, ids, conn = seeded
    from attestor.core import AgentMemory

    mem = AgentMemory(
        Path("/tmp/attestor_orc_bm25"),
        config={
            "mode": "solo",
            "backends": ["postgres"],
            "backend_configs": {
                "postgres": {
                    "url": "postgresql://localhost:5432",
                    "database": "attestor_v4_test",
                    "auth": {"username": "postgres", "password": "attestor"},
                    "v4": True,
                    "skip_schema_init": True,
                    "embedding_dim": 16,
                }
            },
        },
    )
    try:
        assert mem._retrieval.bm25_lane is not None
        assert hasattr(mem._retrieval.bm25_lane, "search")
    finally:
        mem.close()


@pytest.mark.unit
def test_orchestrator_attrs_default_to_none() -> None:
    """Bare orchestrator has bm25_lane=None and sane top-K defaults."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    class _DummyStore:
        def get(self, _): return None

    orch = RetrievalOrchestrator(store=_DummyStore())
    assert orch.bm25_lane is None
    assert orch.bm25_top_k == 30
    assert orch.bm25_min_rank == 0.0
