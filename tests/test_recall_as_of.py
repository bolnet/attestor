"""Phase 5.3 — AgentMemory.recall() as_of + time_window end-to-end.

Top-level integration: builds a SOLO AgentMemory, ingests a contradictory
pair via direct insert + admin backdating, then verifies recall(as_of=)
and recall(time_window=) return the right past/window snapshot.

This is the canonical regulator/audit case from the README:
  "What did the system know on date X?"
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
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


def _has_embedder_keys() -> bool:
    """True when at least one supported embedder API key is set."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VOYAGE_API_KEY")
        or os.environ.get("PINECONE_API_KEY")
    )


pytestmark = pytest.mark.skipif(
    not (_reachable() and _has_embedder_keys()),
    reason="local Postgres unreachable or no embedder API key set",
)


@pytest.fixture(scope="module")
def admin_conn():
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "1024")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


def _build_solo_mem(tmp_path: Path):
    from attestor.core import AgentMemory
    cfg = {
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {
            "postgres": {
                "url": "postgresql://localhost:5432",
                "database": "attestor_v4_test",
                "auth": {"username": "postgres", "password": "attestor"},
                "v4": True,
                "skip_schema_init": True,
            }
        },
    }
    return AgentMemory(tmp_path, config=cfg)


@pytest.fixture
def two_facts_supersede(admin_conn, fresh_schema, tmp_path):
    """Singleton SOLO user, two memories: NYC@T0 superseded by SF@T1."""
    mem = _build_solo_mem(tmp_path)
    uid = mem.default_user.id
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

    nyc = mem.add(
        content="user lives in NYC", category="location", entity="user",
    )
    sf = mem.add(
        content="user lives in SF", category="location", entity="user",
    )
    # Backdate via admin so we control the timestamps exactly
    with admin_conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET valid_from=%s, t_created=%s, "
            "valid_until=%s, t_expired=%s, "
            "status='superseded', superseded_by=%s WHERE id=%s",
            (t0, t0, t1, t1, sf.id, nyc.id),
        )
        cur.execute(
            "UPDATE memories SET valid_from=%s, t_created=%s WHERE id=%s",
            (t1, t1, sf.id),
        )

    yield {
        "mem": mem, "uid": uid, "nyc_id": nyc.id, "sf_id": sf.id,
        "t0": t0, "t1": t1,
    }
    mem.close()


# ──────────────────────────────────────────────────────────────────────────
# Public API as_of round-trip
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_recall_no_as_of_returns_current_only(two_facts_supersede):
    s = two_facts_supersede
    results = s["mem"].recall("where does the user live?")
    contents = [r.memory.content for r in results
                if r.memory.category != "graph_relation"]
    assert any("SF" in c for c in contents), f"missing SF in {contents}"
    assert not any("NYC" in c for c in contents), (
        f"NYC leaked into current recall: {contents}"
    )


@pytest.mark.live
def test_recall_as_of_t0_returns_old_belief(two_facts_supersede):
    """Replay the past: at T0+30d, only the NYC row was known."""
    s = two_facts_supersede
    midpoint = s["t0"] + timedelta(days=30)
    results = s["mem"].recall(
        "where does the user live?", as_of=midpoint,
    )
    contents = [r.memory.content for r in results
                if r.memory.category != "graph_relation"]
    assert any("NYC" in c for c in contents), (
        f"NYC must be visible at as_of=T0+30d: got {contents}"
    )
    assert not any("SF" in c for c in contents), (
        f"SF didn't exist at T0+30d: got {contents}"
    )


@pytest.mark.live
def test_recall_as_of_after_t1_sees_sf(two_facts_supersede):
    s = two_facts_supersede
    just_after = s["t1"] + timedelta(seconds=1)
    results = s["mem"].recall(
        "where does the user live?", as_of=just_after,
    )
    contents = [r.memory.content for r in results
                if r.memory.category != "graph_relation"]
    assert any("SF" in c for c in contents)
    assert not any("NYC" in c for c in contents)


@pytest.mark.live
def test_recall_time_window_event_time(two_facts_supersede):
    """time_window narrows by valid_from/valid_until overlap."""
    from attestor.retrieval.temporal_query import TimeWindow
    s = two_facts_supersede
    # Window covers NYC's validity (T0..T1) but not SF's (T1..)
    tw = TimeWindow(
        start=s["t0"] + timedelta(days=10),
        end=s["t0"] + timedelta(days=20),
    )
    results = s["mem"].recall(
        "where does the user live?", time_window=tw,
    )
    contents = [r.memory.content for r in results
                if r.memory.category != "graph_relation"]
    assert any("NYC" in c for c in contents)
    assert not any("SF" in c for c in contents)


@pytest.mark.live
def test_recall_as_of_signature_backward_compat(two_facts_supersede):
    """Calling recall without as_of/time_window must work unchanged."""
    s = two_facts_supersede
    results = s["mem"].recall("where", budget=2000)
    # Just verify no TypeError + we get a list
    assert isinstance(results, list)
