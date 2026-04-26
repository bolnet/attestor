"""Phase 5.2 — bi-temporal as_of replay (roadmap §C.3/§C.4).

Verifies the canonical regulator/auditor case:
  "What did the system BELIEVE was true on date X?"

Pattern:
  1. Insert a memory with t_created = T0
  2. Insert a contradicting memory at T1 (T1 > T0); supersede the first
  3. Search at as_of=T0     → see only the original
  4. Search at as_of=T1     → see only the new
  5. Search with no as_of   → see only the new (current state)

Also covers TimeWindow filtering by valid_from/valid_until overlap
(event-time, distinct from transaction-time).

Tests run with admin (BYPASSRLS) so we can manipulate t_created /
t_expired directly to simulate historical state.
"""

from __future__ import annotations

import os
import uuid
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


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_reachable() and _ollama_up()),
    reason="local Postgres or Ollama unreachable",
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


@pytest.fixture
def backend(fresh_schema):
    from attestor.store.postgres_backend import PostgresBackend
    be = PostgresBackend({
        "url": "postgresql://localhost:5432",
        "database": "attestor_v4_test",
        "auth": {"username": "postgres", "password": "attestor"},
        "v4": True,
        "skip_schema_init": True,
    })
    yield be
    be.close()


@pytest.fixture
def two_facts_one_supersedes(admin_conn, backend):
    """Insert user + two memories: 'lives in NYC' (T0) → 'lives in SF' (T1).
    The NYC row is marked superseded with t_expired = T1."""
    uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-asof-{uuid.uuid4().hex[:8]}"),
        )
    backend._set_rls_user(str(uid))

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Insert via backend.insert (so embedding is generated) then patch
    # t_created/t_expired/valid_from to the historical values we want.
    from attestor.models import Memory
    nyc = backend.insert(Memory(
        content="user lives in NYC", category="location", entity="user",
        user_id=str(uid), scope="user",
    ))
    backend.add(nyc.id, "user lives in NYC")
    sf = backend.insert(Memory(
        content="user lives in SF", category="location", entity="user",
        user_id=str(uid), scope="user",
    ))
    backend.add(sf.id, "user lives in SF")

    nyc_id = nyc.id
    sf_id = sf.id
    with admin_conn.cursor() as cur:
        # Backdate NYC: created at T0, expired at T1
        cur.execute(
            "UPDATE memories SET valid_from=%s, t_created=%s, "
            "valid_until=%s, t_expired=%s, "
            "status='superseded', superseded_by=%s WHERE id=%s",
            (t0, t0, t1, t1, str(sf_id), str(nyc_id)),
        )
        # SF: created at T1
        cur.execute(
            "UPDATE memories SET valid_from=%s, t_created=%s WHERE id=%s",
            (t1, t1, str(sf_id)),
        )

    return {
        "user_id": str(uid),
        "nyc_id": str(nyc_id),
        "sf_id": str(sf_id),
        "t0": t0,
        "t1": t1,
    }


# ──────────────────────────────────────────────────────────────────────────
# search() with as_of
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_search_no_as_of_returns_current_state_only(backend, two_facts_one_supersedes):
    """No as_of, status='active' filter → only the current SF row."""
    s = two_facts_one_supersedes
    # search() in postgres_backend doesn't apply status filter — that's
    # the orchestrator's job. Here we verify the bi-temporal filter on
    # its own: with as_of=NOW, both rows are still in the index but the
    # NYC row's t_expired is T1 (past), so the policy excludes it.
    hits = backend.search("lives in", limit=20, as_of=datetime.now(timezone.utc))
    ids = {h["memory_id"] for h in hits}
    # NYC has t_expired=T1 in the past → excluded. SF has no t_expired → included.
    assert s["sf_id"] in ids
    assert s["nyc_id"] not in ids


@pytest.mark.live
def test_search_as_of_t0_returns_old_belief(backend, two_facts_one_supersedes):
    """At T0 the system only knew about NYC; SF didn't exist yet."""
    s = two_facts_one_supersedes
    # Query just BEFORE T1, AFTER T0 → only NYC
    midpoint = s["t0"] + timedelta(days=30)
    hits = backend.search("lives in", limit=20, as_of=midpoint)
    ids = {h["memory_id"] for h in hits}
    assert s["nyc_id"] in ids, "NYC must be visible at as_of in [T0, T1)"
    assert s["sf_id"] not in ids, "SF wasn't created until T1"


@pytest.mark.live
def test_search_as_of_t1_sees_sf(backend, two_facts_one_supersedes):
    """At T1 (transaction-time of supersession), SF replaces NYC.

    Both rows have valid_from = T0 / T1. At as_of=T1+epsilon:
      - NYC: t_created=T0 ≤ T1+ε, t_expired=T1 > T1+ε? No (T1 == T1+ε if ε=0;
        we need strict >). So this is exclusive. To be safe, query at T1 + 1s.
    """
    s = two_facts_one_supersedes
    just_after_t1 = s["t1"] + timedelta(seconds=1)
    hits = backend.search("lives in", limit=20, as_of=just_after_t1)
    ids = {h["memory_id"] for h in hits}
    assert s["sf_id"] in ids
    assert s["nyc_id"] not in ids


@pytest.mark.live
def test_search_as_of_before_anything_returns_empty(backend, two_facts_one_supersedes):
    """Query at as_of < any t_created → no rows known."""
    s = two_facts_one_supersedes
    way_back = s["t0"] - timedelta(days=365)
    hits = backend.search("lives in", limit=20, as_of=way_back)
    ids = {h["memory_id"] for h in hits}
    assert s["nyc_id"] not in ids
    assert s["sf_id"] not in ids


# ──────────────────────────────────────────────────────────────────────────
# search() with TimeWindow (event-time overlap)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_search_time_window_filters_by_valid_from(backend, two_facts_one_supersedes):
    """time_window narrowing the search to before T0 returns nothing."""
    from attestor.retrieval.temporal_query import TimeWindow
    s = two_facts_one_supersedes
    tw = TimeWindow(
        start=s["t0"] - timedelta(days=365),
        end=s["t0"] - timedelta(days=1),
    )
    hits = backend.search("lives in", limit=20, time_window=tw)
    assert hits == []


@pytest.mark.live
def test_search_time_window_overlapping_returns_match(backend, two_facts_one_supersedes):
    """time_window covering T0 picks up the NYC row (whose validity range
    is [T0, T1)). The SF row has valid_from=T1, outside this window."""
    from attestor.retrieval.temporal_query import TimeWindow
    s = two_facts_one_supersedes
    tw = TimeWindow(
        start=s["t0"] + timedelta(days=10),
        end=s["t0"] + timedelta(days=20),
    )
    hits = backend.search("lives in", limit=20, time_window=tw)
    ids = {h["memory_id"] for h in hits}
    assert s["nyc_id"] in ids
    assert s["sf_id"] not in ids


@pytest.mark.live
def test_search_time_window_open_ended_end(backend, two_facts_one_supersedes):
    """end=None → 'from start onward' overlap. Should pick up both rows
    since both have valid_from after start."""
    from attestor.retrieval.temporal_query import TimeWindow
    s = two_facts_one_supersedes
    tw = TimeWindow(start=s["t0"] - timedelta(days=1), end=None)
    hits = backend.search("lives in", limit=20, time_window=tw)
    ids = {h["memory_id"] for h in hits}
    assert s["nyc_id"] in ids
    assert s["sf_id"] in ids


# ──────────────────────────────────────────────────────────────────────────
# v3 backward compat — without as_of/time_window, behaves like before
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_search_v3_compat_no_temporal_args(backend, two_facts_one_supersedes):
    """No as_of, no time_window → see all rows that have an embedding."""
    s = two_facts_one_supersedes
    hits = backend.search("lives in", limit=20)
    ids = {h["memory_id"] for h in hits}
    # Both rows are present because we don't filter by status here
    assert {s["nyc_id"], s["sf_id"]}.issubset(ids)


# ──────────────────────────────────────────────────────────────────────────
# BM25 lane respects the same filters
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_bm25_as_of_replay(admin_conn, two_facts_one_supersedes):
    from attestor.retrieval.bm25 import BM25Lane
    s = two_facts_one_supersedes
    lane = BM25Lane(admin_conn)

    # No as_of → both visible
    hits_now = lane.search("lives", active_only=False)
    ids_now = {h.memory_id for h in hits_now}
    assert {s["nyc_id"], s["sf_id"]}.issubset(ids_now)

    # as_of = T0 + 30d (between T0 and T1) → only NYC
    midpoint = s["t0"] + timedelta(days=30)
    hits_past = lane.search("lives", active_only=False, as_of=midpoint)
    ids_past = {h.memory_id for h in hits_past}
    assert s["nyc_id"] in ids_past
    assert s["sf_id"] not in ids_past
