"""Concurrency stress for ProjectRepo.ensure_inbox + SessionRepo.get_or_create_daily.

Both repos had a classic SELECT-then-INSERT race that would fire under
parallel LME runs (every sample shares the SOLO user). Asserting the
fixes hold:

  • ensure_inbox: schema UNIQUE(user_id, name) backs ON CONFLICT DO NOTHING.
    16 threads racing → exactly 1 inbox row, 16 callers all return that
    same row.
  • get_or_create_daily: pg_advisory_xact_lock(hash(user_id, day))
    serializes the get-or-create critical section. 16 threads racing →
    exactly 1 daily session, 16 callers all return that same row.

Marked ``live`` — needs a real Postgres at POSTGRES_URL with the v4
schema applied. Run with::

    POSTGRES_URL=postgresql://... .venv/bin/python -m pytest \
        tests/test_repos_race.py -v
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest


POSTGRES_URL = os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="repos race tests need POSTGRES_URL set to a live v4 Postgres",
)


@pytest.fixture
def fresh_user(tmp_path):
    """Create a brand-new user for each test so previous-run state
    doesn't pollute the race test."""
    import psycopg2
    from attestor.identity.users import UserRepo

    conn = psycopg2.connect(POSTGRES_URL)
    conn.autocommit = True
    repo = UserRepo(conn)
    ext_id = f"race-test-{uuid.uuid4().hex[:8]}"
    user = repo.create_or_get(
        external_id=ext_id, display_name="race-test", metadata={},
    )
    yield conn, user
    # Cleanup: cascade-delete the test user (drops projects+sessions).
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user.id,))
    conn.close()


@pytest.mark.live
def test_ensure_inbox_is_race_safe(fresh_user):
    conn, user = fresh_user
    from attestor.identity.projects import ProjectRepo

    N_THREADS = 16
    results: list = [None] * N_THREADS
    errors: list = []

    def worker(i: int) -> None:
        try:
            # Each thread gets its own connection — that's how parallel
            # LME samples work in practice.
            import psycopg2
            c = psycopg2.connect(POSTGRES_URL)
            c.autocommit = False
            try:
                results[i] = ProjectRepo(c).ensure_inbox(user.id)
            finally:
                c.close()
        except Exception as e:  # noqa: BLE001
            errors.append((i, e))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected errors: {errors}"

    # All 16 callers should have returned the SAME inbox project.
    ids = {p.id for p in results if p is not None}
    assert len(ids) == 1, f"expected 1 inbox, got {len(ids)} distinct ids: {ids}"

    # And the DB should hold exactly one inbox row for this user.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM projects WHERE user_id = %s "
            "AND COALESCE(metadata->>'is_inbox','false') = 'true'",
            (user.id,),
        )
        count = cur.fetchone()[0]
    assert count == 1, f"expected 1 inbox row, got {count}"


@pytest.mark.live
def test_get_or_create_daily_is_race_safe(fresh_user):
    conn, user = fresh_user
    from attestor.identity.projects import ProjectRepo
    from attestor.identity.sessions import SessionRepo

    # Need a project to anchor the session
    project = ProjectRepo(conn).ensure_inbox(user.id)
    day = "2099-12-31"   # far future date so no collision with real sessions

    N_THREADS = 16
    results: list = [None] * N_THREADS
    errors: list = []

    def worker(i: int) -> None:
        try:
            import psycopg2
            c = psycopg2.connect(POSTGRES_URL)
            c.autocommit = False
            try:
                results[i] = SessionRepo(c).get_or_create_daily(
                    user_id=user.id,
                    project_id=project.id,
                    day=day,
                )
            finally:
                c.close()
        except Exception as e:  # noqa: BLE001
            errors.append((i, e))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected errors: {errors}"

    ids = {s.id for s in results if s is not None}
    assert len(ids) == 1, f"expected 1 daily session, got {len(ids)}"

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = %s "
            "AND metadata->>'daily_key' = %s",
            (user.id, f"solo-daily-{day}"),
        )
        count = cur.fetchone()[0]
    assert count == 1, f"expected 1 daily session row, got {count}"
