"""Phase 8.3 — user_quotas table + enforcement tests."""

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


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
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
        for tbl in ("user_quotas", "memories", "episodes",
                    "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture
def seeded_user(admin_conn, fresh_schema):
    """Insert a user; the trigger auto-creates user_quotas row."""
    uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-q-{uuid.uuid4().hex[:8]}"),
        )
    return str(uid)


# ──────────────────────────────────────────────────────────────────────────
# Schema + trigger
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_user_quotas_table_exists(admin_conn, fresh_schema):
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'user_quotas'"
        )
        assert cur.fetchone() is not None


@pytest.mark.live
def test_quota_row_auto_created_on_user_insert(admin_conn, seeded_user):
    """The trigger must populate user_quotas on user creation."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, memory_count, session_count, project_count "
            "FROM user_quotas WHERE user_id = %s",
            (seeded_user,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[1] == 0 and row[2] == 0 and row[3] == 0


@pytest.mark.live
def test_default_limits_are_null(admin_conn, seeded_user):
    """Defaults: all max_* are NULL = unlimited."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT max_memories, max_sessions, max_projects, max_writes_per_day "
            "FROM user_quotas WHERE user_id = %s",
            (seeded_user,),
        )
        row = cur.fetchone()
    assert all(c is None for c in row)


# ──────────────────────────────────────────────────────────────────────────
# QuotaRepo + counter triggers
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_get_returns_quota_for_existing_user(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    q = QuotaRepo(admin_conn).get(seeded_user)
    assert q is not None
    assert q.user_id == seeded_user
    assert q.max_memories is None  # unlimited by default
    assert q.memory_count == 0


@pytest.mark.live
def test_get_returns_none_for_unknown_user(admin_conn):
    from attestor.quotas import QuotaRepo
    assert QuotaRepo(admin_conn).get(str(uuid.uuid4())) is None


@pytest.mark.live
def test_set_limits_updates_existing_row(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    repo = QuotaRepo(admin_conn)
    q = repo.set_limits(seeded_user, max_memories=100, max_sessions=10)
    assert q.max_memories == 100
    assert q.max_sessions == 10
    assert q.max_projects is None  # untouched


@pytest.mark.live
def test_set_limits_partial_update_preserves_others(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_memories=50, max_sessions=5)
    repo.set_limits(seeded_user, max_projects=3)  # only projects
    q = repo.get(seeded_user)
    # All three should now be set
    assert q.max_memories == 50
    assert q.max_sessions == 5
    assert q.max_projects == 3


@pytest.mark.live
def test_memory_insert_increments_counter(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT set_config('attestor.current_user_id', %s, false)",
            (seeded_user,),
        )
        cur.execute(
            "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
            (seeded_user, "fact 1"),
        )
        cur.execute(
            "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
            (seeded_user, "fact 2"),
        )
    q = QuotaRepo(admin_conn).get(seeded_user)
    assert q.memory_count == 2
    # writes_today also bumped
    assert q.writes_today == 2


@pytest.mark.live
def test_session_insert_increments_counter(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (user_id) VALUES (%s)", (seeded_user,),
        )
    q = QuotaRepo(admin_conn).get(seeded_user)
    assert q.session_count == 1


@pytest.mark.live
def test_project_insert_increments_counter(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO projects (user_id, name) VALUES (%s, %s)",
            (seeded_user, "p-1"),
        )
    q = QuotaRepo(admin_conn).get(seeded_user)
    assert q.project_count == 1


# ──────────────────────────────────────────────────────────────────────────
# Enforcement
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_check_memory_quota_under_limit_passes(admin_conn, seeded_user):
    from attestor.quotas import QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_memories=10)
    # Counter is 0; should not raise.
    repo.check_memory_quota(seeded_user)


@pytest.mark.live
def test_check_memory_quota_at_limit_raises(admin_conn, seeded_user):
    from attestor.quotas import QuotaExceeded, QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_memories=2)
    with admin_conn.cursor() as cur:
        for i in range(2):
            cur.execute(
                "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
                (seeded_user, f"f{i}"),
            )
    with pytest.raises(QuotaExceeded) as exc:
        repo.check_memory_quota(seeded_user)
    assert exc.value.field == "max_memories"
    assert exc.value.limit == 2
    assert exc.value.current == 2


@pytest.mark.live
def test_check_session_quota_at_limit_raises(admin_conn, seeded_user):
    from attestor.quotas import QuotaExceeded, QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_sessions=1)
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (user_id) VALUES (%s)", (seeded_user,),
        )
    with pytest.raises(QuotaExceeded) as exc:
        repo.check_session_quota(seeded_user)
    assert exc.value.field == "max_sessions"


@pytest.mark.live
def test_check_project_quota_at_limit_raises(admin_conn, seeded_user):
    from attestor.quotas import QuotaExceeded, QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_projects=1)
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO projects (user_id, name) VALUES (%s, %s)",
            (seeded_user, "p-1"),
        )
    with pytest.raises(QuotaExceeded) as exc:
        repo.check_project_quota(seeded_user)
    assert exc.value.field == "max_projects"


@pytest.mark.live
def test_check_writes_per_day_at_limit_raises(admin_conn, seeded_user):
    from attestor.quotas import QuotaExceeded, QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.set_limits(seeded_user, max_writes_per_day=2)
    with admin_conn.cursor() as cur:
        for i in range(2):
            cur.execute(
                "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
                (seeded_user, f"w{i}"),
            )
    with pytest.raises(QuotaExceeded) as exc:
        repo.check_memory_quota(seeded_user)
    # Could fire on either limit; both are at 2 / max=2
    assert exc.value.field in {"max_memories", "max_writes_per_day"}


@pytest.mark.live
def test_check_passes_when_no_quota_row(admin_conn):
    """Unknown user → no quotas → passes (caller decides whether to error)."""
    from attestor.quotas import QuotaRepo
    repo = QuotaRepo(admin_conn)
    repo.check_memory_quota(str(uuid.uuid4()))   # no raise


# ──────────────────────────────────────────────────────────────────────────
# AgentMemory enforcement (live, end-to-end)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_agent_memory_set_quota_then_add_blocks(admin_conn, fresh_schema, tmp_path):
    from attestor.core import AgentMemory
    from attestor.quotas import QuotaExceeded

    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True,
        }},
    })
    try:
        uid = mem.default_user.id
        mem.set_quota(uid, max_memories=2)

        mem.add("first", category="preference", entity="x")
        mem.add("second", category="preference", entity="x")
        with pytest.raises(QuotaExceeded) as exc:
            mem.add("third", category="preference", entity="x")
        assert exc.value.field == "max_memories"

        q = mem.get_quota(uid)
        assert q.memory_count == 2
        assert q.max_memories == 2
    finally:
        mem.close()
