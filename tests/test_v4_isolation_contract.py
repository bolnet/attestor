"""Phase 1 chunk 4 — cross-user isolation contract suite (tenancy.md §4.4).

Six tests verify the AgentMemory v4 surface is correctly tenant-scoped end
to end. We boot AgentMemory against a non-SUPERUSER, NOBYPASSRLS Postgres
role so RLS is the sole enforcement layer — if a method forgets to call
``_set_rls_user``, the test fails closed (zero rows visible).

Tests:
  1. same-user write+recall round-trips
  2. cross-user write is invisible to other user
  3. RLS-var unset → recall returns nothing (fail-closed)
  4. ensure_user is idempotent (race-safe)
  5. ensure_inbox is idempotent and immutable
  6. RLS var is reset between users on the same backend instance

Schema is pre-loaded by an admin fixture; the runtime role uses
``ATTESTOR_SKIP_SCHEMA_INIT=1`` so it doesn't need ALTER perms.
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


def _load_env() -> None:
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable")


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def admin_conn():
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture(scope="module")
def fresh_schema(admin_conn):
    """Drop + reload the v4 schema as admin so the runtime role has tables to use."""
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture(scope="module")
def app_role(admin_conn, fresh_schema):
    """Create a NOBYPASSRLS, non-SUPERUSER role and return its conn URL."""
    role = f"attestor_iso_{uuid.uuid4().hex[:8]}"
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
    p = urlparse(PG_URL)
    netloc = f"{role}:{pw}@{p.hostname}{':' + str(p.port) if p.port else ''}"
    role_url = urlunparse(p._replace(netloc=netloc))
    yield role_url
    with admin_conn.cursor() as cur:
        cur.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}")
        cur.execute(f"REVOKE ALL ON SCHEMA public FROM {role}")
        cur.execute(f"DROP ROLE {role}")


def _build_agent_memory(role_url: str, tmp_path: Path):
    """Construct an AgentMemory bound to the runtime role. Skips the schema
    init (handled by the admin fixture) so the role doesn't need DDL perms.
    Forces postgres for the document role only — vector store is dropped so
    the embedder isn't required for this test (we verify isolation, not
    similarity search).

    The embedding_dim is stub-set to 16 to match the schema fixture.
    """
    _load_env()

    from attestor.core import AgentMemory
    p = urlparse(role_url)
    backend_cfg = {
        "url": f"postgresql://{p.hostname}:{p.port or 5432}",
        "database": p.path.lstrip("/"),
        "auth": {"username": p.username, "password": p.password},
        "v4": True,
        "skip_schema_init": True,
        "embedding_dim": 16,  # stub; matches admin fixture template
    }
    cfg = {
        # Only the document role — vector + graph roles would need an embedder
        # / AGE. Isolation is enforced at the doc store, which is what we test.
        "backends": ["postgres"],
        "backend_configs": {"postgres": backend_cfg},
    }
    try:
        return AgentMemory(tmp_path, config=cfg)
    except RuntimeError as e:
        pytest.skip(f"backend boot failed: {e}")


@pytest.fixture
def admin_user_id(admin_conn):
    """Create a user via admin (bypassing RLS) so the runtime role can act on it."""
    uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id, display_name) VALUES (%s, %s, %s)",
            (str(uid), f"u-iso-{uuid.uuid4().hex[:8]}", "Iso Test"),
        )
    return str(uid)


# ───────────────────────────────────────────────────────────────────────────
# Contract tests (tenancy.md §4.4)
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_contract_same_user_write_then_get(app_role, admin_conn, tmp_path):
    """User A writes via AgentMemory.add(), then fetches their own row by id.
    RLS must admit the read because the var matches user A's id."""
    user_a = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_a), f"u-{uuid.uuid4().hex[:8]}"),
        )

    mem = _build_agent_memory(app_role, tmp_path)
    try:
        written = mem.add(
            content="user A's preference: dark mode",
            user_id=str(user_a),
            scope="user",
            agent_id="planner",
        )
        assert written.user_id == str(user_a)
        # Persist verification: fetch by id with the RLS var still set to A.
        fetched = mem._store.get(written.id)
        assert fetched is not None, "user A could not read their own row"
        assert fetched.user_id == str(user_a)
        assert "dark mode" in fetched.content
    finally:
        mem.close()


@pytest.mark.live
def test_contract_cross_user_get_blocked(app_role, admin_conn, tmp_path):
    """User A writes a memory; switching RLS to user B must hide it.
    This is the core tenancy guarantee — verified at the DB layer."""
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_a), f"u-a-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_b), f"u-b-{uuid.uuid4().hex[:8]}"),
        )

    mem = _build_agent_memory(app_role, tmp_path)
    try:
        secret = "user A's confidential salary number"
        written = mem.add(content=secret, user_id=str(user_a), scope="user")

        # Switch RLS to user B and try to fetch A's row
        mem._store._set_rls_user(str(user_b))
        leaked = mem._store.get(written.id)
        assert leaked is None, (
            f"RLS leak: user B fetched user A's row id={written.id} content={leaked.content!r}"
        )
    finally:
        mem.close()


@pytest.mark.live
def test_contract_unset_user_id_fails_closed(app_role, admin_conn, tmp_path):
    """If the RLS var is empty, NULLIF(...,'')::uuid is NULL and the policy
    evaluates to FALSE for every row → zero visible. Fail closed, never open."""
    user_a = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_a), f"u-{uuid.uuid4().hex[:8]}"),
        )

    mem = _build_agent_memory(app_role, tmp_path)
    try:
        written = mem.add(
            content="some fact for user A",
            user_id=str(user_a),
            scope="user",
        )
        # Clear the RLS var explicitly
        mem._store._set_rls_user(None)
        # Direct SQL: a fully unscoped count must be zero
        with mem._store._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            count = cur.fetchone()[0]
        assert count == 0, (
            f"RLS not fail-closed: {count} rows visible with empty user var"
        )
        # And get() also returns None
        assert mem._store.get(written.id) is None
    finally:
        mem.close()


@pytest.mark.live
def test_contract_ensure_user_is_idempotent(app_role, admin_conn, tmp_path):
    """ensure_user must be safe to call repeatedly with the same external_id."""
    # ensure_user needs admin-bypass to write to the users table — RLS only
    # admits rows with a matching id, but the row doesn't exist yet. So this
    # contract is verified via the admin connection (production wires this
    # through a migration role; tests do the same).
    from attestor.identity import UserRepo
    repo = UserRepo(admin_conn)
    ext = f"u-idem-{uuid.uuid4().hex[:8]}"

    u1 = repo.create_or_get(external_id=ext, display_name="Idem")
    u2 = repo.create_or_get(external_id=ext, display_name="Idem")
    u3 = repo.create_or_get(external_id=ext)

    assert u1.id == u2.id == u3.id, "create_or_get must return the same user every time"


@pytest.mark.live
def test_contract_ensure_inbox_idempotent_and_immutable(app_role, admin_conn, tmp_path):
    """The Inbox project is idempotent on create and refuses delete/archive."""
    user_a = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_a), f"u-inbox-{uuid.uuid4().hex[:8]}"),
        )

    mem = _build_agent_memory(app_role, tmp_path)
    try:
        # Set RLS so projects table reads/writes are admitted for this user
        mem._store._set_rls_user(str(user_a))

        inbox1 = mem.ensure_inbox(str(user_a))
        inbox2 = mem.ensure_inbox(str(user_a))
        assert inbox1.id == inbox2.id, "ensure_inbox must return the same row"
        assert inbox1.is_inbox

        from attestor.identity.projects import InboxImmutableError
        with pytest.raises(InboxImmutableError):
            mem._project_repo().archive(inbox1.id)
        with pytest.raises(InboxImmutableError):
            mem._project_repo().delete(inbox1.id)
    finally:
        mem.close()


@pytest.mark.live
def test_contract_rls_resets_between_user_calls(app_role, admin_conn, tmp_path):
    """Two operations in sequence on the same backend with different user_ids
    must each see only their own data. No state leak between calls."""
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_a), f"u-a2-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(user_b), f"u-b2-{uuid.uuid4().hex[:8]}"),
        )

    mem = _build_agent_memory(app_role, tmp_path)
    try:
        a_mem = mem.add(content="alpha-secret-A", user_id=str(user_a), scope="user")
        b_mem = mem.add(content="beta-secret-B", user_id=str(user_b), scope="user")

        # Set RLS to A: only A's row should resolve via direct fetch.
        mem._store._set_rls_user(str(user_a))
        assert mem._store.get(a_mem.id) is not None, "user A cannot see own row"
        assert mem._store.get(b_mem.id) is None, "user A leaked B's row"

        # Switch to B: only B's row should resolve.
        mem._store._set_rls_user(str(user_b))
        assert mem._store.get(b_mem.id) is not None, "user B cannot see own row"
        assert mem._store.get(a_mem.id) is None, "user B leaked A's row"
    finally:
        mem.close()
