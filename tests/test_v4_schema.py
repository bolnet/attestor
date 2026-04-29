"""Phase 0 — schema integrity + RLS isolation tests.

These tests load ``attestor/store/schema.sql`` into a live Postgres database
and verify the v4 schema is correct: tables exist, NOT NULL constraints are
in place, indexes exist, and row-level security policies do isolate users.

Run:
    PG_TEST_URL='postgresql://postgres:attestor@localhost:5432/attestor' \\
        poetry run python -m pytest tests/test_v4_schema.py -v

Defaults to the local Docker stack URL if PG_TEST_URL is unset.
Skips entirely if no Postgres is reachable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

try:
    import psycopg2
    import psycopg2.extras
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

PG_URL = os.environ.get(
    "PG_TEST_URL",
    # Separate DB so this test doesn't trash the live `attestor` DB.
    # Tests auto-create `attestor_v4_test` if missing (see admin_conn fixture).
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
PG_ADMIN_URL = os.environ.get(
    "PG_ADMIN_URL",
    "postgresql://postgres:attestor@localhost:5432/postgres",
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "attestor" / "store" / "schema.sql"


def _pg_reachable(url: str) -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        conn = psycopg2.connect(url, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


def _ensure_test_db_exists() -> bool:
    """Create the test database if missing; return True if reachable."""
    if not HAVE_PSYCOPG2:
        return False
    if _pg_reachable(PG_URL):
        return True
    if not _pg_reachable(PG_ADMIN_URL):
        return False
    # Connect to admin DB, CREATE the test DB.
    from urllib.parse import urlparse
    target_db = urlparse(PG_URL).path.lstrip("/")
    try:
        conn = psycopg2.connect(PG_ADMIN_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{target_db}"')
        conn.close()
    except Exception:
        return False
    return _pg_reachable(PG_URL)


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ensure_test_db_exists(),
        reason=f"local Postgres unreachable at {PG_URL}",
    ),
]


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def admin_conn():
    """Superuser connection on the dedicated test DB. Tests fully control
    schema state here; the live `attestor` DB is never touched."""
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def fresh_schema(admin_conn):
    """Drop + reload schema for a clean slate."""
    raw = SCHEMA_PATH.read_text()
    # Substitute the embedding dimension placeholder with a small test value.
    schema_sql = raw.replace("{embedding_dim}", "16")
    with admin_conn.cursor() as cur:
        # Drop in reverse FK order to avoid CASCADE noise.
        for table in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        cur.execute(schema_sql)
    yield


# ───────────────────────────────────────────────────────────────────────────
# Schema integrity
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"missing {SCHEMA_PATH}"
    text = SCHEMA_PATH.read_text()
    assert "CREATE TABLE IF NOT EXISTS users" in text
    assert "CREATE TABLE IF NOT EXISTS memories" in text
    assert "ENABLE ROW LEVEL SECURITY" in text


@pytest.mark.unit
def test_schema_sql_has_no_legacy_namespace_column() -> None:
    """v4 deletes the `namespace` column entirely (replaced by user_id+scope)."""
    text = SCHEMA_PATH.read_text()
    # Search the memories CREATE TABLE block for `namespace`.
    # The word should not appear as a column declaration.
    in_memories = False
    for line in text.splitlines():
        stripped = line.strip()
        if "CREATE TABLE IF NOT EXISTS memories" in stripped:
            in_memories = True
            continue
        if in_memories and stripped.startswith(");"):
            in_memories = False
            continue
        if in_memories and "namespace" in stripped.lower():
            pytest.fail(f"v4 memories table must not have a namespace column: {line!r}")


@pytest.mark.unit
def test_schema_has_bitemporal_columns() -> None:
    text = SCHEMA_PATH.read_text()
    for col in ("valid_from", "valid_until", "t_created", "t_expired"):
        assert col in text, f"bi-temporal column missing: {col}"


@pytest.mark.unit
def test_schema_has_provenance_columns() -> None:
    text = SCHEMA_PATH.read_text()
    for col in ("source_episode_id", "source_span", "extraction_model",
                "agent_id", "parent_agent_id", "signature"):
        assert col in text, f"provenance column missing: {col}"


# ───────────────────────────────────────────────────────────────────────────
# Live: tables, columns, indexes, constraints
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_all_v4_tables_created(admin_conn, fresh_schema) -> None:
    expected = {"users", "projects", "sessions", "episodes", "memories"}
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        actual = {r[0] for r in cur.fetchall()}
    missing = expected - actual
    assert not missing, f"missing tables: {missing}"


@pytest.mark.live
def test_memories_user_id_is_not_null(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='user_id'"
        )
        row = cur.fetchone()
        assert row is not None, "user_id column missing on memories"
        assert row[0] == "NO", "user_id must be NOT NULL"


@pytest.mark.live
def test_memories_has_no_namespace_column(admin_conn, fresh_schema) -> None:
    """v4 removes namespace; isolation is by (user_id, project_id, session_id, scope)."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='namespace'"
        )
        assert cur.fetchone() is None, "memories.namespace must be removed in v4"


@pytest.mark.live
def test_memories_scope_column_present(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT data_type, column_default FROM information_schema.columns "
            "WHERE table_name='memories' AND column_name='scope'"
        )
        row = cur.fetchone()
        assert row is not None, "memories.scope column missing"
        assert "user" in (row[1] or ""), "scope default should be 'user'"


@pytest.mark.live
def test_critical_indexes_exist(admin_conn, fresh_schema) -> None:
    """Without these the recall query won't perform."""
    expected = {
        "idx_memories_user_scope_status",
        "idx_memories_user_project",
        "idx_memories_user_session",
        "idx_memories_temporal",
        "idx_memories_embedding_hnsw",
        "idx_episodes_user_session_ts",
        "idx_sessions_user_project_active",
        "idx_users_external_id",
    }
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname='public'"
        )
        actual = {r[0] for r in cur.fetchall()}
    missing = expected - actual
    assert not missing, f"missing indexes: {missing}"


@pytest.mark.live
def test_rls_enabled_on_tenant_tables(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT tablename, rowsecurity FROM pg_tables "
            "WHERE schemaname='public' "
            "AND tablename IN ('users','projects','sessions','episodes','memories')"
        )
        rows = dict(cur.fetchall())
    for table in ("users", "projects", "sessions", "episodes", "memories"):
        assert rows.get(table) is True, f"RLS not enabled on {table}"


@pytest.mark.live
def test_rls_policies_present(admin_conn, fresh_schema) -> None:
    expected = {
        # users: split into 4 (SELECT/UPDATE/DELETE strict, INSERT permissive
        # so the bootstrap path can create the caller's own user row).
        "tenant_isolation_users_select",
        "tenant_isolation_users_update",
        "tenant_isolation_users_delete",
        "tenant_isolation_users_insert",
        "tenant_isolation_projects",
        "tenant_isolation_sessions",
        "tenant_isolation_episodes",
        "tenant_isolation_memories",
    }
    with admin_conn.cursor() as cur:
        cur.execute("SELECT policyname FROM pg_policies WHERE schemaname='public'")
        actual = {r[0] for r in cur.fetchall()}
    missing = expected - actual
    assert not missing, f"missing RLS policies: {missing}"


# ───────────────────────────────────────────────────────────────────────────
# Live: RLS isolation actually works
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app_role_url(admin_conn, fresh_schema):
    """Create a non-superuser role and yield a connection URL using it."""
    role = f"attestor_test_{uuid.uuid4().hex[:8]}"
    pw = "test"
    with admin_conn.cursor() as cur:
        cur.execute(f"CREATE ROLE {role} LOGIN NOBYPASSRLS PASSWORD '{pw}'")
        # Grant on the schema + tables we test against
        cur.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}"
        )
        cur.execute(
            f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"
        )
    # Build a URL by substituting credentials.
    from urllib.parse import urlparse, urlunparse
    p = urlparse(PG_URL)
    netloc = f"{role}:{pw}@{p.hostname}{':' + str(p.port) if p.port else ''}"
    role_url = urlunparse(p._replace(netloc=netloc))
    yield role_url
    with admin_conn.cursor() as cur:
        cur.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}")
        cur.execute(f"REVOKE ALL ON SCHEMA public FROM {role}")
        cur.execute(f"DROP ROLE {role}")


@pytest.fixture
def two_users(admin_conn, fresh_schema):
    """Insert two distinct users, return (user_a_id, user_b_id)."""
    ua = uuid.uuid4()
    ub = uuid.uuid4()
    ext_a = f"test-a-{uuid.uuid4().hex[:8]}"
    ext_b = f"test-b-{uuid.uuid4().hex[:8]}"
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id, display_name) VALUES (%s, %s, %s)",
            (str(ua), ext_a, "User A"),
        )
        cur.execute(
            "INSERT INTO users (id, external_id, display_name) VALUES (%s, %s, %s)",
            (str(ub), ext_b, "User B"),
        )
        cur.execute(
            "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
            (str(ua), "user A's secret fact"),
        )
        cur.execute(
            "INSERT INTO memories (user_id, content) VALUES (%s, %s)",
            (str(ub), "user B's secret fact"),
        )
    yield ua, ub


@pytest.mark.live
def test_rls_isolates_users_when_role_does_not_bypass(app_role_url, two_users) -> None:
    """Connect as the non-superuser role, set the user_id var, verify only
    that user's rows are visible — even with a SELECT that has no WHERE clause.
    This is the contract: app-level forgetting → DB-level zero rows."""
    user_a, user_b = two_users
    with psycopg2.connect(app_role_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Set RLS variable to user_a
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (str(user_a),),
            )
            cur.execute("SELECT user_id, content FROM memories")
            rows = cur.fetchall()
            user_ids = {str(r[0]) for r in rows}
            contents = [r[1] for r in rows]
            assert user_ids == {str(user_a)}, (
                f"RLS leaked user B rows: got user_ids={user_ids}"
            )
            assert any("A" in c for c in contents)
            assert not any("B" in c for c in contents)

            # Switch to user_b — should now see only B's rows
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (str(user_b),),
            )
            cur.execute("SELECT user_id, content FROM memories")
            rows = cur.fetchall()
            user_ids = {str(r[0]) for r in rows}
            assert user_ids == {str(user_b)}


@pytest.mark.live
def test_rls_unset_variable_returns_zero_rows(app_role_url, two_users) -> None:
    """If the app forgets to set the RLS var, the policy is empty → zero rows.
    Fail closed, never fail open."""
    with psycopg2.connect(app_role_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Explicitly clear the var
            cur.execute(
                "SELECT set_config('attestor.current_user_id', '', false)"
            )
            cur.execute("SELECT COUNT(*) FROM memories")
            assert cur.fetchone()[0] == 0, "RLS must fail closed when var is unset"


@pytest.mark.live
def test_rls_blocks_cross_user_insert_lookup(app_role_url, two_users) -> None:
    """Even after inserting as user_a, querying as user_b returns zero of A's rows."""
    user_a, user_b = two_users
    with psycopg2.connect(app_role_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (str(user_a),),
            )
            cur.execute(
                "INSERT INTO memories (user_id, content) VALUES (%s, %s) RETURNING id",
                (str(user_a), "fresh A row"),
            )
            new_id = cur.fetchone()[0]

            cur.execute(
                "SELECT set_config('attestor.current_user_id', %s, false)",
                (str(user_b),),
            )
            cur.execute(
                "SELECT 1 FROM memories WHERE id = %s", (str(new_id),)
            )
            assert cur.fetchone() is None, "user_b must not see user_a's freshly inserted row"
