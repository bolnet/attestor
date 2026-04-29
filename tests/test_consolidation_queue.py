"""Phase 7.1 — ConsolidationQueue: enqueue / claim / done / failed."""

from __future__ import annotations

import os
import time
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


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable"),
]


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
def seeded_episodes(admin_conn, fresh_schema):
    """Insert user + session + 5 episodes via raw SQL. Returns list of ids."""
    uid = uuid.uuid4()
    sid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-q-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO sessions (id, user_id) VALUES (%s, %s)",
            (str(sid), str(uid)),
        )
    ids = []
    base = datetime.now(timezone.utc)
    for i in range(5):
        eid = uuid.uuid4()
        with admin_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO episodes (id, user_id, session_id, thread_id, "
                "user_turn_text, assistant_turn_text, user_ts, assistant_ts) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (str(eid), str(uid), str(sid), f"thr-{i}",
                 f"u-{i}", f"a-{i}",
                 base + timedelta(seconds=i),
                 base + timedelta(seconds=i + 1)),
            )
        ids.append(str(eid))
    return ids


# ──────────────────────────────────────────────────────────────────────────
# Schema columns present
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_episodes_has_consolidation_columns(admin_conn, fresh_schema):
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='episodes' AND column_name LIKE 'consolidation_%'"
        )
        cols = {r[0] for r in cur.fetchall()}
    expected = {"consolidation_state", "consolidation_claimed_at",
                "consolidation_done_at", "consolidation_error"}
    assert expected.issubset(cols), f"missing: {expected - cols}"


@pytest.mark.live
def test_pending_index_exists(admin_conn, fresh_schema):
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename='episodes' "
            "AND indexname='idx_episodes_consolidation_pending'"
        )
        assert cur.fetchone() is not None


# ──────────────────────────────────────────────────────────────────────────
# Default state + dequeue
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_episodes_default_state_is_pending(admin_conn, seeded_episodes):
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state FROM episodes "
            "WHERE id = ANY(%s::uuid[])",
            (seeded_episodes,),
        )
        states = [r[0] for r in cur.fetchall()]
    assert len(states) == 5
    assert all(s == "pending" for s in states)


@pytest.mark.live
def test_dequeue_batch_claims_oldest_first(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=3)
    assert len(batch) == 3
    # Episodes 0, 1, 2 (created earliest) should come first
    assert [e.id for e in batch] == seeded_episodes[:3]


@pytest.mark.live
def test_dequeue_marks_processing_with_lease(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=2)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state, consolidation_claimed_at "
            "FROM episodes WHERE id = ANY(%s::uuid[]) ORDER BY created_at",
            ([e.id for e in batch],),
        )
        rows = cur.fetchall()
    for state, claimed_at in rows:
        assert state == "processing"
        assert claimed_at is not None


@pytest.mark.live
def test_dequeue_skips_already_claimed(admin_conn, seeded_episodes):
    """Two consecutive dequeues never return the same row."""
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    first = q.dequeue_batch(limit=2)
    second = q.dequeue_batch(limit=2)
    overlap = {e.id for e in first} & {e.id for e in second}
    assert overlap == set(), f"queue handed out the same row twice: {overlap}"


@pytest.mark.live
def test_dequeue_returns_empty_when_drained(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    q.dequeue_batch(limit=100)  # drain
    assert q.dequeue_batch(limit=10) == []


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle transitions
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_mark_done_transitions_state(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=1)
    q.mark_done(batch[0].id)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state, consolidation_done_at "
            "FROM episodes WHERE id = %s", (batch[0].id,),
        )
        state, done_at = cur.fetchone()
    assert state == "done"
    assert done_at is not None


@pytest.mark.live
def test_mark_failed_records_error(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=1)
    q.mark_failed(batch[0].id, "model returned bad JSON")
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state, consolidation_error "
            "FROM episodes WHERE id = %s", (batch[0].id,),
        )
        state, err = cur.fetchone()
    assert state == "failed"
    assert err == "model returned bad JSON"


@pytest.mark.live
def test_mark_failed_truncates_long_error(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=1)
    long_err = "x" * 5000
    q.mark_failed(batch[0].id, long_err)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_error FROM episodes WHERE id = %s",
            (batch[0].id,),
        )
        err = cur.fetchone()[0]
    assert len(err) == 1000


@pytest.mark.live
def test_release_returns_to_pending(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=1)
    q.release(batch[0].id)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state FROM episodes WHERE id = %s",
            (batch[0].id,),
        )
        assert cur.fetchone()[0] == "pending"


# ──────────────────────────────────────────────────────────────────────────
# Re-enqueue / stale lease reclaim
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_enqueue_resets_done_to_pending(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    batch = q.dequeue_batch(limit=1)
    q.mark_done(batch[0].id)
    assert q.enqueue(batch[0].id) is True
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state FROM episodes WHERE id = %s",
            (batch[0].id,),
        )
        assert cur.fetchone()[0] == "pending"


@pytest.mark.live
def test_enqueue_no_op_on_pending_or_processing(admin_conn, seeded_episodes):
    """Re-enqueueing a row that's already pending must return False."""
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    assert q.enqueue(seeded_episodes[0]) is False  # already pending


@pytest.mark.live
def test_dequeue_reclaims_stale_processing_lease(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn, queue_lock_seconds=1)
    batch = q.dequeue_batch(limit=1)
    eid = batch[0].id
    # Sleep past the lock window
    time.sleep(2)
    # A second worker should reclaim the stale row
    second = q.dequeue_batch(limit=1)
    assert any(e.id == eid for e in second), (
        "stale processing lease was not reclaimed"
    )


# ──────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_stats_counts_each_state(admin_conn, seeded_episodes):
    from attestor.consolidation.queue import ConsolidationQueue
    q = ConsolidationQueue(admin_conn)
    # 5 pending initially
    assert q.stats()["pending"] == 5

    batch = q.dequeue_batch(limit=3)
    s = q.stats()
    assert s["pending"] == 2
    assert s["processing"] == 3

    q.mark_done(batch[0].id)
    q.mark_failed(batch[1].id, "test")
    s = q.stats()
    assert s["done"] == 1
    assert s["failed"] == 1
    assert s["processing"] == 1
    assert s["pending"] == 2
