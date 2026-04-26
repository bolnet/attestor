"""Phase 1 — UserRepo, ProjectRepo, SessionRepo CRUD tests.

Run against the dedicated `attestor_v4_test` database (auto-created by the
test_v4_schema.py fixture). Schema is reset between modules.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone, timedelta
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
        conn = psycopg2.connect(PG_URL, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable")


@pytest.fixture(scope="module")
def conn():
    """Fresh schema per module."""
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with c.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield c
    c.close()


@pytest.fixture
def fresh_user(conn):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    ext = f"u-{uuid.uuid4().hex[:8]}"
    return repo.create(external_id=ext, display_name="Tester")


# ───────────────────────────────────────────────────────────────────────────
# UserRepo
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_create_and_get(conn):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    ext = f"u-{uuid.uuid4().hex[:8]}"
    user = repo.create(external_id=ext, email="x@y.z", display_name="Alice")
    assert user.external_id == ext
    assert user.email == "x@y.z"
    assert user.status == "active"
    fetched = repo.get(user.id)
    assert fetched is not None
    assert fetched.id == user.id


@pytest.mark.unit
def test_user_create_or_get_is_idempotent(conn):
    """First call creates; second call with same external_id returns the same row."""
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    ext = f"u-{uuid.uuid4().hex[:8]}"
    a = repo.create_or_get(external_id=ext, display_name="First")
    b = repo.create_or_get(external_id=ext, display_name="Second")
    assert a.id == b.id, "create_or_get must be idempotent on external_id"


@pytest.mark.unit
def test_user_create_strict_raises_on_duplicate(conn):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    ext = f"u-{uuid.uuid4().hex[:8]}"
    repo.create(external_id=ext)
    with pytest.raises(psycopg2.IntegrityError):
        repo.create(external_id=ext)
    conn.rollback()


@pytest.mark.unit
def test_user_find_by_external_id(conn):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    ext = f"u-{uuid.uuid4().hex[:8]}"
    created = repo.create(external_id=ext)
    found = repo.find_by_external_id(ext)
    assert found is not None and found.id == created.id
    assert repo.find_by_external_id(f"missing-{uuid.uuid4().hex[:8]}") is None


@pytest.mark.unit
def test_user_soft_delete_then_find_returns_none(conn, fresh_user):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    assert repo.soft_delete(fresh_user.id)
    # find_by_external_id filters status='active' — soft-deleted user invisible
    assert repo.find_by_external_id(fresh_user.external_id) is None
    # But get() still returns the row (status='deleted')
    g = repo.get(fresh_user.id)
    assert g is not None and g.status == "deleted"


@pytest.mark.unit
def test_user_purge_removes_row(conn, fresh_user):
    from attestor.identity import UserRepo
    repo = UserRepo(conn)
    assert repo.purge(fresh_user.id)
    assert repo.get(fresh_user.id) is None


# ───────────────────────────────────────────────────────────────────────────
# ProjectRepo
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_project_create_and_unique_name(conn, fresh_user):
    from attestor.identity import ProjectRepo
    repo = ProjectRepo(conn)
    p = repo.create(user_id=fresh_user.id, name="Work")
    assert p.user_id == fresh_user.id
    assert p.name == "Work"
    with pytest.raises(psycopg2.IntegrityError):
        repo.create(user_id=fresh_user.id, name="Work")
    conn.rollback()


@pytest.mark.unit
def test_inbox_is_idempotent(conn, fresh_user):
    from attestor.identity import ProjectRepo
    repo = ProjectRepo(conn)
    inbox_a = repo.ensure_inbox(fresh_user.id)
    inbox_b = repo.ensure_inbox(fresh_user.id)
    assert inbox_a.id == inbox_b.id
    assert inbox_a.is_inbox is True


@pytest.mark.unit
def test_project_list_excludes_inbox_by_default(conn, fresh_user):
    from attestor.identity import ProjectRepo
    repo = ProjectRepo(conn)
    repo.ensure_inbox(fresh_user.id)
    repo.create(user_id=fresh_user.id, name="Work")
    repo.create(user_id=fresh_user.id, name="Personal")
    listed = repo.list_for_user(fresh_user.id)
    assert {p.name for p in listed} == {"Work", "Personal"}
    listed_with = repo.list_for_user(fresh_user.id, include_inbox=True)
    assert "Inbox" in {p.name for p in listed_with}


@pytest.mark.unit
def test_inbox_cannot_be_archived_or_deleted(conn, fresh_user):
    from attestor.identity import ProjectRepo
    from attestor.identity.projects import InboxImmutableError
    repo = ProjectRepo(conn)
    inbox = repo.ensure_inbox(fresh_user.id)
    with pytest.raises(InboxImmutableError):
        repo.archive(inbox.id)
    with pytest.raises(InboxImmutableError):
        repo.delete(inbox.id)


@pytest.mark.unit
def test_project_archive_soft_state(conn, fresh_user):
    from attestor.identity import ProjectRepo
    repo = ProjectRepo(conn)
    p = repo.create(user_id=fresh_user.id, name="X")
    assert repo.archive(p.id)
    fetched = repo.get(p.id)
    assert fetched is not None and fetched.status == "archived"
    # Archived project is excluded from list_for_user
    listed = repo.list_for_user(fresh_user.id)
    assert all(x.id != p.id for x in listed)


# ───────────────────────────────────────────────────────────────────────────
# SessionRepo
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_inbox(conn, fresh_user):
    from attestor.identity import ProjectRepo
    return ProjectRepo(conn).ensure_inbox(fresh_user.id)


@pytest.mark.unit
def test_session_create(conn, fresh_user, fresh_inbox):
    from attestor.identity import SessionRepo
    repo = SessionRepo(conn)
    s = repo.create(user_id=fresh_user.id, project_id=fresh_inbox.id, title="Hi")
    assert s.user_id == fresh_user.id
    assert s.project_id == fresh_inbox.id
    assert s.status == "active"
    assert s.message_count == 0


@pytest.mark.unit
def test_session_get_or_create_daily_idempotent(conn, fresh_user, fresh_inbox):
    from attestor.identity import SessionRepo
    repo = SessionRepo(conn)
    a = repo.get_or_create_daily(fresh_user.id, fresh_inbox.id, "2026-04-25")
    b = repo.get_or_create_daily(fresh_user.id, fresh_inbox.id, "2026-04-25")
    assert a.id == b.id
    c = repo.get_or_create_daily(fresh_user.id, fresh_inbox.id, "2026-04-26")
    assert c.id != a.id


@pytest.mark.unit
def test_session_lifecycle(conn, fresh_user, fresh_inbox):
    from attestor.identity import SessionRepo
    from attestor.identity.sessions import SessionStateError
    repo = SessionRepo(conn)
    s = repo.create(user_id=fresh_user.id, project_id=fresh_inbox.id)
    # Bump activity → message_count goes up
    repo.bump_activity(s.id, message_increment=3)
    refreshed = repo.get(s.id)
    assert refreshed.message_count == 3
    # End → status=ended
    ended = repo.end(s.id)
    assert ended is not None and ended.status == "ended"
    assert ended.ended_at is not None
    # Can't write to ended? assert_writable should still allow ended sessions
    # (they're queryable, just not actively running). Only archived blocks.
    repo.assert_writable(s.id)
    # Archive → assert_writable raises
    repo.archive(s.id)
    with pytest.raises(SessionStateError):
        repo.assert_writable(s.id)


@pytest.mark.unit
def test_session_list_for_user(conn, fresh_user, fresh_inbox):
    from attestor.identity import SessionRepo
    repo = SessionRepo(conn)
    s1 = repo.create(user_id=fresh_user.id, project_id=fresh_inbox.id, title="a")
    s2 = repo.create(user_id=fresh_user.id, project_id=fresh_inbox.id, title="b")
    listed = repo.list_for_user(fresh_user.id, status="active")
    titles = {s.title for s in listed}
    assert {"a", "b"}.issubset(titles)
    # Filter to a different project → no results
    other = repo.list_for_user(
        fresh_user.id, project_id=str(uuid.uuid4()), status="active"
    )
    assert other == []


# ───────────────────────────────────────────────────────────────────────────
# AgentContext v4 fields
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_agent_context_for_chat_builds_namespace():
    from attestor.context import AgentContext
    from attestor.models import MemoryScope
    ctx = AgentContext.for_chat(
        user_id="u1", project_id="p1", session_id="s1"
    )
    assert ctx.user_id == "u1"
    assert ctx.project_id == "p1"
    assert ctx.session_id == "s1"
    assert ctx.scope_default == MemoryScope.USER
    assert ctx.namespace == "user:u1/project:p1/session:s1"


@pytest.mark.unit
def test_agent_context_as_agent_propagates_v4_fields():
    from attestor.context import AgentContext, AgentRole
    ctx = AgentContext.for_chat(user_id="u1", project_id="p1", session_id="s1")
    sub = ctx.as_agent("worker", role=AgentRole.RESEARCHER)
    assert sub.user_id == "u1"
    assert sub.project_id == "p1"
    assert sub.scope_default == ctx.scope_default
    assert sub.role == AgentRole.RESEARCHER


@pytest.mark.unit
def test_agent_context_v3_compat_user_id_optional():
    """Existing v3 callers create AgentContext without user_id; that should
    still work — user_id is None until a v4 caller sets it via for_chat()."""
    from attestor.context import AgentContext
    ctx = AgentContext(agent_id="a", namespace="default")
    assert ctx.user_id is None
    assert ctx.project_id is None
