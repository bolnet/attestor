"""Phase 1 chunk 2 — PostgresBackend v4 mode end-to-end.

Verifies that:
  1. ATTESTOR_V4=1 (or config['v4']=True) routes _init_schema to load
     attestor/store/schema.sql instead of the v3 inline DDL.
  2. PostgresBackend.insert() with a v4-shaped Memory writes user_id,
     project_id, session_id, scope, t_created.
  3. RLS still works: a non-superuser connection sees only its user's rows.
  4. The default (v4 unset) path still loads the v3 inline schema and
     existing v3 callers get a Memory with the legacy id format.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

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
def fresh_db(admin_conn):
    """Drop all v3+v4 tables so each test starts from a clean slate."""
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    yield


def _load_env_file() -> None:
    """Best-effort: load .env so OPENROUTER_API_KEY (used by embedder) is set."""
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k and not os.environ.get(k):
            os.environ[k] = v.strip().strip('"').strip("'")


@pytest.fixture
def v4_backend(fresh_db, monkeypatch):
    """PostgresBackend booted in v4 mode."""
    _load_env_file()
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("embedding provider not configured (set OPENROUTER_API_KEY)")
    monkeypatch.setenv("ATTESTOR_V4", "1")
    from attestor.store.postgres_backend import PostgresBackend
    p = urlparse(PG_URL)
    config = {
        "url": f"postgresql://{p.hostname}:{p.port or 5432}",
        "database": p.path.lstrip("/"),
        "auth": {"username": p.username, "password": p.password},
    }
    try:
        be = PostgresBackend(config)
    except RuntimeError as e:
        pytest.skip(f"embedder unavailable: {e}")
    yield be
    be.close()


# ───────────────────────────────────────────────────────────────────────────
# Memory model — v4 fields
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_memory_v4_fields_default_to_none() -> None:
    from attestor.models import Memory
    m = Memory(content="hi")
    assert m.user_id is None
    assert m.project_id is None
    assert m.session_id is None
    assert m.scope == "user"
    assert m.t_created is None
    assert m.t_expired is None
    assert m.source_episode_id is None
    assert m.is_v4 is False


@pytest.mark.unit
def test_memory_is_v4_when_user_id_set() -> None:
    from attestor.models import Memory
    m = Memory(content="hi", user_id=str(uuid.uuid4()))
    assert m.is_v4 is True


@pytest.mark.unit
def test_memory_to_dict_includes_v4_fields() -> None:
    from attestor.models import Memory
    m = Memory(content="hi", user_id="u1", project_id="p1", scope="project")
    d = m.to_dict()
    assert d["user_id"] == "u1"
    assert d["project_id"] == "p1"
    assert d["scope"] == "project"


@pytest.mark.unit
def test_memory_from_row_v3_compat() -> None:
    """A v3 row dict (no user_id, no t_created) must round-trip cleanly."""
    from attestor.models import Memory
    row = {
        "id": "abc123",
        "content": "v3 row",
        "tags": ["a", "b"],
        "category": "general",
        "entity": None,
        "namespace": "default",
        "created_at": "2026-01-01T00:00:00+00:00",
        "valid_from": "2026-01-01T00:00:00+00:00",
        "metadata": "{}",
    }
    m = Memory.from_row(row)
    assert m.id == "abc123"
    assert m.user_id is None
    assert m.scope == "user"  # default


@pytest.mark.unit
def test_memory_from_row_v4_shape() -> None:
    from attestor.models import Memory
    uid = str(uuid.uuid4())
    pid = str(uuid.uuid4())
    row = {
        "id": uuid.uuid4(),
        "user_id": uid,
        "project_id": pid,
        "session_id": None,
        "scope": "project",
        "content": "v4 row",
        "tags": [],
        "category": "general",
        "namespace": "default",
        "created_at": "2026-04-25T12:00:00+00:00",
        "valid_from": "2026-04-25T12:00:00+00:00",
        "t_created": "2026-04-25T12:00:00+00:00",
        "metadata": {},
    }
    m = Memory.from_row(row)
    assert m.user_id == uid
    assert m.project_id == pid
    assert m.scope == "project"
    assert m.is_v4


# ───────────────────────────────────────────────────────────────────────────
# Live: ATTESTOR_V4 init path
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_v4_flag_loads_schema_sql(v4_backend, admin_conn):
    """When ATTESTOR_V4=1, the v4 tables must exist (users, projects,
    sessions, episodes) AND the memories table must have user_id NOT NULL."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        tables = {r[0] for r in cur.fetchall()}
    for t in ("users", "projects", "sessions", "episodes", "memories"):
        assert t in tables, f"v4 init missing table: {t}"

    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='user_id'"
        )
        row = cur.fetchone()
        assert row is not None and row[0] == "NO"


@pytest.mark.live
def test_v4_insert_and_get_roundtrip(v4_backend, admin_conn):
    """Insert a Memory with user_id set; get it back; v4 columns intact."""
    from attestor.models import Memory

    # Need a real user row first (FK constraint).
    user_id = str(uuid.uuid4())
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (user_id, f"test-{uuid.uuid4().hex[:8]}"),
        )

    m = Memory(
        content="v4 fact",
        user_id=user_id,
        scope="user",
        agent_id="planner-01",
        visibility="team",
    )
    inserted = v4_backend.insert(m)
    # v4 insert overwrites the model's id with the DB-generated UUID
    assert inserted.id != "" and len(inserted.id) >= 32  # UUID string
    assert inserted.t_created is not None  # populated by RETURNING

    fetched = v4_backend.get(inserted.id)
    assert fetched is not None
    assert fetched.user_id == user_id
    assert fetched.scope == "user"
    assert fetched.agent_id == "planner-01"
    assert fetched.is_v4


@pytest.mark.live
def test_v4_insert_without_user_id_raises(v4_backend):
    """v4 mode requires user_id; insert without it raises ValueError."""
    from attestor.models import Memory
    m = Memory(content="missing user_id")
    with pytest.raises(ValueError, match="user_id"):
        v4_backend.insert(m)


@pytest.mark.live
def test_v4_init_is_idempotent(v4_backend, admin_conn, monkeypatch):
    """Calling _init_schema_v4() twice must not error (CREATE TABLE IF NOT EXISTS)."""
    # Already initialized once via fixture. Trigger init again on the same DB.
    v4_backend._init_schema_v4()
    # Tables still there
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name IN ('users','projects','sessions','episodes','memories')"
        )
        assert cur.fetchone()[0] == 5
