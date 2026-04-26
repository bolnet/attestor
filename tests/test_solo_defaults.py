"""Phase 2 chunk 4 — SOLO defaults: singleton user + daily session + Inbox.

Verifies the zero-config "AgentMemory().add('foo')" path:
  1. SOLO mode boots a singleton user with external_id='local'.
  2. Repeated boots return the same user (idempotent).
  3. Inbox is provisioned alongside the user, immutable.
  4. Daily session helper rotates per-day, idempotent within the day.
  5. add() with no identity params writes a fully-resolved memory.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

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
    """Drop + reload v4 schema. Function-scoped so each test starts clean."""
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture(scope="module")
def app_role(admin_conn):
    role = f"attestor_solo_{uuid.uuid4().hex[:8]}"
    pw = "test"
    with admin_conn.cursor() as cur:
        cur.execute(f"CREATE ROLE {role} LOGIN NOBYPASSRLS PASSWORD '{pw}'")
        cur.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}"
        )
        cur.execute(
            f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"
        )
        # Grant on future tables too (schema is reloaded per-test)
        cur.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {role}"
        )
    p = urlparse(PG_URL)
    netloc = f"{role}:{pw}@{p.hostname}{':' + str(p.port) if p.port else ''}"
    yield urlunparse(p._replace(netloc=netloc))
    with admin_conn.cursor() as cur:
        cur.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {role}"
        )
        cur.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}")
        cur.execute(f"REVOKE ALL ON SCHEMA public FROM {role}")
        cur.execute(f"DROP ROLE {role}")


def _build_solo_mem(role_url: str, tmp_path: Path):
    from attestor.core import AgentMemory
    p = urlparse(role_url)
    cfg = {
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {
            "postgres": {
                "url": f"postgresql://{p.hostname}:{p.port or 5432}",
                "database": p.path.lstrip("/"),
                "auth": {"username": p.username, "password": p.password},
                "v4": True,
                "skip_schema_init": True,
                "embedding_dim": 16,
            }
        },
    }
    return AgentMemory(tmp_path, config=cfg)


# ───────────────────────────────────────────────────────────────────────────
# SOLO bootstrap (singleton user)
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_solo_boot_creates_singleton_user(app_role, fresh_schema, tmp_path):
    """First AgentMemory() in SOLO mode creates the 'local' user + Inbox."""
    from attestor.mode import AttestorMode, SOLO_USER_EXTERNAL_ID

    mem = _build_solo_mem(app_role, tmp_path)
    try:
        assert mem.mode is AttestorMode.SOLO
        assert mem.default_user is not None
        assert mem.default_user.external_id == SOLO_USER_EXTERNAL_ID

        # Inbox should already exist for that user
        inbox = mem.ensure_inbox(mem.default_user.id)
        assert inbox.is_inbox
    finally:
        mem.close()


@pytest.mark.live
def test_solo_boot_is_idempotent(app_role, fresh_schema, tmp_path):
    """Constructing AgentMemory twice in SOLO mode returns the same user_id."""
    mem1 = _build_solo_mem(app_role, tmp_path)
    uid1 = mem1.default_user.id
    mem1.close()

    mem2 = _build_solo_mem(app_role, tmp_path)
    try:
        assert mem2.default_user.id == uid1, (
            "SOLO singleton drifted between boots"
        )
    finally:
        mem2.close()


# ───────────────────────────────────────────────────────────────────────────
# Zero-config add() / recall() round-trip
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_solo_zero_config_add_and_get(app_role, fresh_schema, tmp_path):
    """``mem.add('hi')`` with no identity params resolves to the SOLO user,
    Inbox project, today's session — and the row is fetchable."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        m = mem.add(content="zero-config fact")
        assert m.user_id == mem.default_user.id
        assert m.project_id is not None
        assert m.session_id is not None  # autostart=True for add()
        assert m.scope == "user"

        fetched = mem._store.get(m.id)
        assert fetched is not None
        assert fetched.content == "zero-config fact"
    finally:
        mem.close()


@pytest.mark.live
def test_solo_two_adds_same_session(app_role, fresh_schema, tmp_path):
    """Two adds in the same boot land in the same daily session."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        m1 = mem.add(content="first")
        m2 = mem.add(content="second")
        assert m1.session_id == m2.session_id, (
            "daily session must be reused within the day"
        )
    finally:
        mem.close()


# ───────────────────────────────────────────────────────────────────────────
# Daily session rotation
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_daily_session_rotates_by_day(app_role, fresh_schema, tmp_path):
    """get_or_create_daily called for two different days returns two sessions."""
    from attestor.identity.defaults import get_or_create_daily_session

    mem = _build_solo_mem(app_role, tmp_path)
    try:
        uid = mem.default_user.id
        inbox = mem.ensure_inbox(uid)

        s_today = get_or_create_daily_session(
            uid, inbox.id, mem._session_repo(), day="2026-04-26",
        )
        s_today_again = get_or_create_daily_session(
            uid, inbox.id, mem._session_repo(), day="2026-04-26",
        )
        s_tomorrow = get_or_create_daily_session(
            uid, inbox.id, mem._session_repo(), day="2026-04-27",
        )

        assert s_today.id == s_today_again.id, "same day must reuse session"
        assert s_today.id != s_tomorrow.id, "different days must be different sessions"
    finally:
        mem.close()


# ───────────────────────────────────────────────────────────────────────────
# Inbox immutability under default flow
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_solo_inbox_immutable(app_role, fresh_schema, tmp_path):
    from attestor.identity.projects import InboxImmutableError

    mem = _build_solo_mem(app_role, tmp_path)
    try:
        inbox = mem.ensure_inbox(mem.default_user.id)
        with pytest.raises(InboxImmutableError):
            mem._project_repo().delete(inbox.id)
        with pytest.raises(InboxImmutableError):
            mem._project_repo().archive(inbox.id)
    finally:
        mem.close()
