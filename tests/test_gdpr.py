"""Phase 8.5 — GDPR delete + export + audit log tests."""

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
        for tbl in ("deletion_audit", "user_quotas", "memories", "episodes",
                    "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture
def populated_user(admin_conn, fresh_schema):
    """User with a project + session + 2 episodes + 3 memories."""
    uid = uuid.uuid4()
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    ext = f"u-gdpr-{uuid.uuid4().hex[:8]}"
    base = datetime.now(timezone.utc)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT set_config('attestor.current_user_id', %s, false)",
            (str(uid),),
        )
        cur.execute(
            "INSERT INTO users (id, external_id, email, display_name) "
            "VALUES (%s, %s, %s, %s)",
            (str(uid), ext, "alice@example.com", "Alice"),
        )
        cur.execute(
            "INSERT INTO projects (id, user_id, name) VALUES (%s, %s, %s)",
            (str(pid), str(uid), "p-1"),
        )
        cur.execute(
            "INSERT INTO sessions (id, user_id, project_id) "
            "VALUES (%s, %s, %s)",
            (str(sid), str(uid), str(pid)),
        )
        for i in range(2):
            cur.execute(
                "INSERT INTO episodes (user_id, session_id, thread_id, "
                "user_turn_text, assistant_turn_text, user_ts, assistant_ts) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (str(uid), str(sid), f"thr-{i}", f"u-{i}", f"a-{i}",
                 base + timedelta(seconds=i),
                 base + timedelta(seconds=i + 1)),
            )
        for i in range(3):
            cur.execute(
                "INSERT INTO memories (user_id, content, category, entity) "
                "VALUES (%s, %s, %s, %s)",
                (str(uid), f"fact-{i}", "preference", "user"),
            )
    return {"user_id": str(uid), "external_id": ext,
            "project_id": str(pid), "session_id": str(sid)}


# ──────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_deletion_audit_table_exists(admin_conn, fresh_schema):
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'deletion_audit'"
        )
        assert cur.fetchone() is not None


@pytest.mark.live
def test_deletion_audit_is_rls_exempt(admin_conn, fresh_schema):
    """The audit log MUST outlive the user it logs — no RLS."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT rowsecurity FROM pg_tables "
            "WHERE schemaname='public' AND tablename='deletion_audit'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] is False, "deletion_audit must NOT have RLS enabled"


# ──────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_export_user_includes_all_sections(admin_conn, populated_user):
    from attestor.gdpr import export_user
    result = export_user(admin_conn, populated_user["external_id"])
    payload = result.to_dict()
    assert payload["user"]["external_id"] == populated_user["external_id"]
    assert payload["user"]["email"] == "alice@example.com"
    assert len(payload["projects"]) == 1
    assert len(payload["sessions"]) == 1
    assert len(payload["episodes"]) == 2
    assert len(payload["memories"]) == 3
    assert payload["row_counts"]["memories"] == 3
    assert "exported_at" in payload


@pytest.mark.live
def test_export_user_unknown_raises(admin_conn, fresh_schema):
    from attestor.gdpr import export_user
    with pytest.raises(LookupError, match="not found"):
        export_user(admin_conn, "no-such-user")


@pytest.mark.live
def test_export_user_payload_is_jsonable(admin_conn, populated_user):
    """to_dict() output must round-trip through json.dumps."""
    import json
    from attestor.gdpr import export_user
    payload = export_user(
        admin_conn, populated_user["external_id"],
    ).to_dict()
    serialized = json.dumps(payload)
    assert "alice@example.com" in serialized


# ──────────────────────────────────────────────────────────────────────────
# Purge
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_purge_user_cascades_through_all_tables(admin_conn, populated_user):
    from attestor.gdpr import purge_user
    result = purge_user(
        admin_conn, populated_user["external_id"],
        reason="gdpr_request", deleted_by="admin-1",
    )
    assert result.user_existed is True
    assert result.audit_id is not None
    assert result.counts["memories"] == 3
    assert result.counts["episodes"] == 2
    assert result.counts["sessions"] == 1
    assert result.counts["projects"] == 1

    # Every table reports 0 rows for this user
    uid = populated_user["user_id"]
    with admin_conn.cursor() as cur:
        for table in ("memories", "episodes", "sessions", "projects",
                      "user_quotas"):
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = %s::uuid",
                (uid,),
            )
            assert cur.fetchone()[0] == 0, (
                f"{table} still has rows for purged user"
            )
        # Users row itself is gone
        cur.execute(
            "SELECT COUNT(*) FROM users WHERE id = %s::uuid", (uid,),
        )
        assert cur.fetchone()[0] == 0


@pytest.mark.live
def test_purge_writes_audit_entry(admin_conn, populated_user):
    """Audit row carries everything regulators need to verify the deletion."""
    from attestor.gdpr import purge_user
    result = purge_user(
        admin_conn, populated_user["external_id"],
        reason="gdpr_request", deleted_by="alice",
    )
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT external_id, deleted_by, reason, counts "
            "FROM deletion_audit WHERE id = %s::uuid",
            (result.audit_id,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == populated_user["external_id"]
    assert row[1] == "alice"
    assert row[2] == "gdpr_request"
    counts = row[3]
    assert counts["memories"] == 3


@pytest.mark.live
def test_audit_persists_after_user_purged(admin_conn, populated_user):
    """The whole point — auditor can confirm a deletion happened later."""
    from attestor.gdpr import list_audit_log, purge_user
    purge_user(admin_conn, populated_user["external_id"])

    audit = list_audit_log(admin_conn)
    assert len(audit) == 1
    assert audit[0]["external_id"] == populated_user["external_id"]
    assert "deleted_at" in audit[0]


@pytest.mark.live
def test_purge_unknown_user_returns_no_op(admin_conn, fresh_schema):
    from attestor.gdpr import purge_user
    r = purge_user(admin_conn, "no-such-user")
    assert r.user_existed is False
    assert r.audit_id is None
    assert r.counts == {}


@pytest.mark.live
def test_purge_quota_row_also_removed(admin_conn, populated_user):
    """user_quotas FK has ON DELETE CASCADE — must vanish with the user."""
    from attestor.gdpr import purge_user
    uid = populated_user["user_id"]
    purge_user(admin_conn, populated_user["external_id"])
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM user_quotas WHERE user_id = %s::uuid",
            (uid,),
        )
        assert cur.fetchone()[0] == 0


# ──────────────────────────────────────────────────────────────────────────
# AgentMemory wrapper
# ──────────────────────────────────────────────────────────────────────────


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_agent_memory_purge_user_round_trip(admin_conn, fresh_schema, tmp_path):
    from attestor.core import AgentMemory

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
        # Default SOLO user is "local"; add a memory for them
        mem.add("delete-me", category="preference", entity="x")
        export = mem.export_user("local")
        assert len(export["memories"]) >= 1

        result = mem.purge_user("local", reason="gdpr_request")
        assert result["user_existed"] is True
        assert result["counts"]["memories"] >= 1

        audit = mem.deletion_audit_log()
        assert any(e["external_id"] == "local" for e in audit)
    finally:
        mem.close()
