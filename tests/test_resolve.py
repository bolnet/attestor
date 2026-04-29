"""Phase 2 chunk 4 — _resolve() correctness across mode/scope combos.

Verifies the resolution chain (defaults.md §5):

  User:    explicit user_id wins. Else SOLO singleton. Else raise.
  Project: explicit project_id wins. Else session.project_id. Else Inbox.
  Session: explicit session_id wins. Else autostart (daily for SOLO).
           If autostart=False → None.

Also covers cross-tenant authz: an explicit project_id or session_id
that doesn't belong to the resolved user raises LookupError (NOT 403,
to avoid leaking existence).
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


@pytest.fixture(scope="module")
def app_role(admin_conn):
    role = f"attestor_resolve_{uuid.uuid4().hex[:8]}"
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


def _mem(role_url: str, tmp_path: Path, mode: str = "solo"):
    from attestor.core import AgentMemory
    p = urlparse(role_url)
    cfg = {
        "mode": mode,
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
# User resolution
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_resolve_solo_no_user_id_uses_singleton(app_role, fresh_schema, tmp_path):
    mem = _mem(app_role, tmp_path)
    try:
        rc = mem._resolve()
        assert rc.user.id == mem.default_user.id
        assert rc.user.external_id == "local"
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_hosted_no_user_id_raises(app_role, fresh_schema, tmp_path):
    """In HOSTED mode there is no default user — caller must pass one."""
    mem = _mem(app_role, tmp_path, mode="hosted")
    try:
        with pytest.raises(PermissionError, match="user_id is required"):
            mem._resolve()
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_explicit_user_id_wins(app_role, fresh_schema, tmp_path, admin_conn):
    """Explicit user_id is honored even in SOLO mode."""
    other_uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(other_uid), f"u-other-{uuid.uuid4().hex[:8]}"),
        )

    mem = _mem(app_role, tmp_path)
    try:
        rc = mem._resolve(user_id=str(other_uid))
        assert rc.user.id == str(other_uid)
        assert rc.user.id != mem.default_user.id
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_unknown_user_id_raises(app_role, fresh_schema, tmp_path):
    mem = _mem(app_role, tmp_path)
    try:
        with pytest.raises(LookupError, match="not found"):
            mem._resolve(user_id=str(uuid.uuid4()))
    finally:
        mem.close()


# ───────────────────────────────────────────────────────────────────────────
# Project resolution
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_resolve_no_project_id_uses_inbox(app_role, fresh_schema, tmp_path):
    mem = _mem(app_role, tmp_path)
    try:
        rc = mem._resolve()
        assert rc.project.is_inbox, (
            f"expected Inbox, got {rc.project.name!r} metadata={rc.project.metadata}"
        )
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_explicit_project_id_wins(app_role, fresh_schema, tmp_path):
    mem = _mem(app_role, tmp_path)
    try:
        custom = mem.create_project(
            user_id=mem.default_user.id, name="custom-work",
        )
        rc = mem._resolve(project_id=custom.id)
        assert rc.project.id == custom.id
        assert not rc.project.is_inbox
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_session_implies_project(app_role, fresh_schema, tmp_path):
    """If only session_id is given, the session's project_id is used."""
    mem = _mem(app_role, tmp_path)
    try:
        custom = mem.create_project(
            user_id=mem.default_user.id, name="session-project",
        )
        sess = mem.start_session(
            user_id=mem.default_user.id, project_id=custom.id,
        )
        rc = mem._resolve(session_id=sess.id)
        assert rc.project.id == custom.id
        assert rc.session.id == sess.id
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_cross_user_project_raises(app_role, fresh_schema, tmp_path, admin_conn):
    """Explicit project_id belonging to another user → LookupError (404 not 403)."""
    other_uid = uuid.uuid4()
    other_pid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(other_uid), f"u-x-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO projects (id, user_id, name) VALUES (%s, %s, %s)",
            (str(other_pid), str(other_uid), "secret"),
        )

    mem = _mem(app_role, tmp_path)
    try:
        with pytest.raises(LookupError):
            mem._resolve(project_id=str(other_pid))
    finally:
        mem.close()


# ───────────────────────────────────────────────────────────────────────────
# Session resolution
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_resolve_autostart_solo_returns_daily(app_role, fresh_schema, tmp_path):
    """SOLO autostart returns the daily session, reused across calls."""
    mem = _mem(app_role, tmp_path)
    try:
        rc1 = mem._resolve(autostart=True)
        rc2 = mem._resolve(autostart=True)
        assert rc1.session is not None
        assert rc1.session.id == rc2.session.id, (
            "SOLO daily session must be reused"
        )
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_autostart_false_returns_no_session(app_role, fresh_schema, tmp_path):
    """recall() passes autostart=False — no session needed for read-only."""
    mem = _mem(app_role, tmp_path)
    try:
        rc = mem._resolve(autostart=False)
        assert rc.session is None
    finally:
        mem.close()


@pytest.mark.live
def test_resolve_explicit_session_wins(app_role, fresh_schema, tmp_path):
    mem = _mem(app_role, tmp_path)
    try:
        sess = mem.start_session(user_id=mem.default_user.id, title="manual")
        rc = mem._resolve(session_id=sess.id, autostart=True)
        assert rc.session.id == sess.id
        # Should NOT be the daily session
        from attestor.identity.defaults import get_or_create_daily_session
        daily = get_or_create_daily_session(
            mem.default_user.id, rc.project.id, mem._session_repo(),
        )
        assert sess.id != daily.id
    finally:
        mem.close()
