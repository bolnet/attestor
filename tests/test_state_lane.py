"""Tests for the typed profile memory lane (``attestor.state``).

The state lane stores small, type-checked profile facts (preferences,
capability_set, language) in a Postgres table that lives next to the
v4 retrieval schema. It is **not** retrieval-based -- callers fetch by
key, not by similarity. Bi-temporal: every set/delete supersedes the
previous active row instead of mutating in place.

Most tests run in pure unit mode against an in-memory fake of the
Postgres connection so they don't require a live database. A small
``@pytest.mark.live`` block at the bottom round-trips against the
shared ``attestor_v4_test`` DB to lock down the SQL.
"""

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

from attestor.context import AgentContext, AgentRole
from attestor.state import (
    StateRecord,
    StateRepo,
    StateValidationError,
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory fake of the psycopg2 connection used by repos. Just enough to
# exercise StateRepo's SQL path without a live database. Each test creates
# a fresh fake so rows don't bleed across tests.
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self._last_result: list[dict] = []
        self._desc = False

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    @property
    def description(self) -> bool:
        return self._desc

    def fetchone(self) -> dict | None:
        return self._last_result[0] if self._last_result else None

    def fetchall(self) -> list[dict]:
        return list(self._last_result)


class _FakeConn:
    """In-memory stand-in for a psycopg2 connection.

    Stores rows in a list; StateRepo's SQL is interpreted by hand. The
    test only needs three operations: SELECT active rows, INSERT a new
    row, UPDATE t_expired on the previous active row.

    The fake intentionally over-approximates SQL semantics — it's a test
    double, not a Postgres clone.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.committed = False

    def cursor(self, cursor_factory: object | None = None) -> _FakeCursor:
        return _FakeCursor(self.rows)

    def commit(self) -> None:
        self.committed = True


def _now_iso() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python repo (in-memory mode)
# ─────────────────────────────────────────────────────────────────────────────
#
# StateRepo accepts a pluggable backend so tests can drive it without a live
# DB. The default constructor uses Postgres; the in-memory backend below
# exercises the public surface.


@pytest.fixture
def repo() -> StateRepo:
    from attestor.state.repo import InMemoryStateBackend

    return StateRepo(backend=InMemoryStateBackend())


USER_A = str(uuid.uuid4())
USER_B = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Roundtrip + history
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_set_get_roundtrip(repo: StateRepo) -> None:
    rec = repo.set(
        "preferences.theme",
        "dark",
        user_id=USER_A,
    )
    assert isinstance(rec, StateRecord)
    assert rec.value == "dark"
    assert rec.key == "preferences.theme"
    assert rec.user_id == USER_A
    assert rec.t_expired is None

    assert repo.get("preferences.theme", user_id=USER_A) == "dark"


@pytest.mark.unit
def test_get_missing_returns_none(repo: StateRepo) -> None:
    assert repo.get("nope", user_id=USER_A) is None


@pytest.mark.unit
def test_set_then_set_supersedes(repo: StateRepo) -> None:
    repo.set("preferences.theme", "dark", user_id=USER_A)
    repo.set("preferences.theme", "light", user_id=USER_A)

    assert repo.get("preferences.theme", user_id=USER_A) == "light"
    history = repo.history("preferences.theme", user_id=USER_A)
    assert len(history) == 2
    # First write must now be expired
    assert history[0].t_expired is not None
    # Latest write is still open
    assert history[-1].t_expired is None


@pytest.mark.unit
def test_history_returns_chronological(repo: StateRepo) -> None:
    for v in ("a", "b", "c"):
        repo.set("k", v, user_id=USER_A)
    history = repo.history("k", user_id=USER_A)
    assert [r.value for r in history] == ["a", "b", "c"]


@pytest.mark.unit
def test_as_of_replays_past_state(repo: StateRepo) -> None:
    import time as _time

    t0 = _now_iso()
    repo.set("preferences.theme", "dark", user_id=USER_A)
    # Capture a wall-clock instant strictly between the two writes. We
    # sleep enough either side so the second set lands AFTER ``t_between``,
    # making the bi-temporal slice deterministic across machine speeds.
    _time.sleep(0.01)
    t_between = _now_iso()
    _time.sleep(0.01)
    repo.set("preferences.theme", "light", user_id=USER_A)

    # Between the two writes the value must be 'dark'
    assert (
        repo.as_of("preferences.theme", ts=t_between, user_id=USER_A) == "dark"
    )
    # Far enough in the future, returns the latest
    far_future = _now_iso() + timedelta(days=1)
    assert (
        repo.as_of("preferences.theme", ts=far_future, user_id=USER_A) == "light"
    )
    # Before any write, returns None
    far_past = t0 - timedelta(days=1)
    assert repo.as_of("preferences.theme", ts=far_past, user_id=USER_A) is None


# ─────────────────────────────────────────────────────────────────────────────
# List + delete
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_list_with_prefix(repo: StateRepo) -> None:
    repo.set("preferences.theme", "dark", user_id=USER_A)
    repo.set("preferences.lang", "en", user_id=USER_A)
    repo.set("other.key", 42, user_id=USER_A)

    listed = repo.list(user_id=USER_A, prefix="preferences.")
    assert set(listed.keys()) == {"preferences.theme", "preferences.lang"}
    assert listed["preferences.theme"] == "dark"
    assert listed["preferences.lang"] == "en"


@pytest.mark.unit
def test_list_without_prefix_returns_all_active(repo: StateRepo) -> None:
    repo.set("a", 1, user_id=USER_A)
    repo.set("b", 2, user_id=USER_A)
    listed = repo.list(user_id=USER_A)
    assert listed == {"a": 1, "b": 2}


@pytest.mark.unit
def test_delete_marks_expired(repo: StateRepo) -> None:
    repo.set("preferences.theme", "dark", user_id=USER_A)
    assert repo.delete("preferences.theme", user_id=USER_A) is True
    assert repo.get("preferences.theme", user_id=USER_A) is None
    # History still has the row
    history = repo.history("preferences.theme", user_id=USER_A)
    assert len(history) == 1
    assert history[0].t_expired is not None


@pytest.mark.unit
def test_delete_missing_returns_false(repo: StateRepo) -> None:
    assert repo.delete("nope", user_id=USER_A) is False


# ─────────────────────────────────────────────────────────────────────────────
# Tenancy / isolation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_namespace_isolation(repo: StateRepo) -> None:
    """User A and User B must not see each other's state."""
    repo.set("preferences.theme", "dark", user_id=USER_A)
    repo.set("preferences.theme", "light", user_id=USER_B)

    assert repo.get("preferences.theme", user_id=USER_A) == "dark"
    assert repo.get("preferences.theme", user_id=USER_B) == "light"
    assert repo.list(user_id=USER_A) == {"preferences.theme": "dark"}
    assert repo.list(user_id=USER_B) == {"preferences.theme": "light"}


@pytest.mark.unit
def test_project_scope_isolation(repo: StateRepo) -> None:
    pid_x = str(uuid.uuid4())
    pid_y = str(uuid.uuid4())
    repo.set("k", "x-val", user_id=USER_A, project_id=pid_x)
    repo.set("k", "y-val", user_id=USER_A, project_id=pid_y)
    assert repo.get("k", user_id=USER_A, project_id=pid_x) == "x-val"
    assert repo.get("k", user_id=USER_A, project_id=pid_y) == "y-val"


# ─────────────────────────────────────────────────────────────────────────────
# JSON-Schema validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_schema_validation_pass(repo: StateRepo) -> None:
    rec = repo.set(
        "preferences",
        {"theme": "dark", "language": "en"},
        user_id=USER_A,
        schema="user_preferences_v1",
    )
    assert rec.schema_id == "user_preferences_v1"
    assert rec.value == {"theme": "dark", "language": "en"}


@pytest.mark.unit
def test_schema_validation_fail_raises(repo: StateRepo) -> None:
    with pytest.raises(StateValidationError) as exc:
        repo.set(
            "preferences",
            {"theme": 12345},  # not a string -> invalid
            user_id=USER_A,
            schema="user_preferences_v1",
        )
    # Make sure the message names the schema so debugging is easy.
    assert "user_preferences_v1" in str(exc.value)


@pytest.mark.unit
def test_schema_unknown_raises(repo: StateRepo) -> None:
    with pytest.raises(StateValidationError) as exc:
        repo.set(
            "x",
            "y",
            user_id=USER_A,
            schema="does_not_exist_v9",
        )
    assert "does_not_exist_v9" in str(exc.value)


@pytest.mark.unit
def test_agent_capability_schema_pass(repo: StateRepo) -> None:
    rec = repo.set(
        "agent.cap",
        {
            "capability_set": ["read", "search"],
            "max_tokens": 1024,
            "allowed_tools": ["echo"],
        },
        user_id=USER_A,
        schema="agent_capability_v1",
    )
    assert rec.schema_id == "agent_capability_v1"


@pytest.mark.unit
def test_agent_capability_schema_fail(repo: StateRepo) -> None:
    with pytest.raises(StateValidationError):
        repo.set(
            "agent.cap",
            {"capability_set": "not-a-list"},
            user_id=USER_A,
            schema="agent_capability_v1",
        )


# ─────────────────────────────────────────────────────────────────────────────
# RBAC integration via AgentContext
# ─────────────────────────────────────────────────────────────────────────────


def _ctx_with_state(role: AgentRole, *, read_only: bool = False) -> AgentContext:
    """Build a context whose memory exposes a state repo via ``state``."""
    from attestor.state.repo import InMemoryStateBackend

    repo = StateRepo(backend=InMemoryStateBackend())

    class _MemStub:
        def __init__(self, state_repo: StateRepo) -> None:
            self.state = state_repo

    return AgentContext(
        agent_id=f"agent-{role.value}",
        role=role,
        memory=_MemStub(repo),
        read_only=read_only,
        user_id=USER_A,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "role",
    [AgentRole.ORCHESTRATOR, AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.RESEARCHER],
)
def test_rbac_write_allowed_roles_can_set(role: AgentRole) -> None:
    ctx = _ctx_with_state(role)
    rec = ctx.state_set("preferences.theme", "dark")
    assert rec.value == "dark"


@pytest.mark.unit
def test_rbac_reviewer_cannot_write() -> None:
    ctx = _ctx_with_state(AgentRole.REVIEWER)
    with pytest.raises(PermissionError):
        ctx.state_set("preferences.theme", "dark")


@pytest.mark.unit
def test_rbac_monitor_cannot_write() -> None:
    ctx = _ctx_with_state(AgentRole.MONITOR)
    with pytest.raises(PermissionError):
        ctx.state_set("k", "v")


@pytest.mark.unit
def test_rbac_read_only_strips_writes() -> None:
    """``read_only=True`` must veto writes even for ORCHESTRATOR."""
    ctx = _ctx_with_state(AgentRole.ORCHESTRATOR, read_only=True)
    with pytest.raises(PermissionError):
        ctx.state_set("k", "v")


@pytest.mark.unit
def test_rbac_reviewer_can_read() -> None:
    """Setup as orchestrator, then read as reviewer (READ is universal)."""
    from attestor.state.repo import InMemoryStateBackend

    repo = StateRepo(backend=InMemoryStateBackend())
    repo.set("preferences.theme", "dark", user_id=USER_A)

    class _MemStub:
        def __init__(self) -> None:
            self.state = repo

    ctx = AgentContext(
        agent_id="reviewer-1",
        role=AgentRole.REVIEWER,
        memory=_MemStub(),
        user_id=USER_A,
    )
    assert ctx.state_get("preferences.theme") == "dark"


# ─────────────────────────────────────────────────────────────────────────────
# Live (Postgres) round-trip — env-gated
# ─────────────────────────────────────────────────────────────────────────────

PG_URL = os.environ.get(
    "PG_TEST_URL",
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "attestor" / "store" / "schema.sql"
)


def _live_pg_reachable() -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


@pytest.fixture
def admin_conn():
    if not _live_pg_reachable():
        pytest.skip("local Postgres unreachable")
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "1024")
    with admin_conn.cursor() as cur:
        for tbl in (
            "state",
            "user_quotas",
            "memories",
            "episodes",
            "sessions",
            "projects",
            "users",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    return


@pytest.fixture
def seeded_user(admin_conn, fresh_schema):
    uid = str(uuid.uuid4())
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (uid, f"u-state-{uuid.uuid4().hex[:8]}"),
        )
    return uid


@pytest.mark.live
def test_live_state_table_exists(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'state'"
        )
        assert cur.fetchone() is not None


@pytest.mark.live
def test_live_set_get_roundtrip(admin_conn, seeded_user) -> None:
    repo = StateRepo(conn=admin_conn)
    repo.set("preferences.theme", "dark", user_id=seeded_user)
    assert repo.get("preferences.theme", user_id=seeded_user) == "dark"


@pytest.mark.live
def test_live_supersession(admin_conn, seeded_user) -> None:
    repo = StateRepo(conn=admin_conn)
    repo.set("k", "v1", user_id=seeded_user)
    repo.set("k", "v2", user_id=seeded_user)
    assert repo.get("k", user_id=seeded_user) == "v2"
    history = repo.history("k", user_id=seeded_user)
    assert len(history) == 2
    assert history[0].t_expired is not None
    assert history[-1].t_expired is None


@pytest.mark.live
def test_live_namespace_isolation(admin_conn, fresh_schema) -> None:
    a = str(uuid.uuid4())
    b = str(uuid.uuid4())
    with admin_conn.cursor() as cur:
        for u in (a, b):
            cur.execute(
                "INSERT INTO users (id, external_id) VALUES (%s, %s)",
                (u, f"u-{u[:8]}"),
            )
    repo = StateRepo(conn=admin_conn)
    repo.set("k", "a-val", user_id=a)
    repo.set("k", "b-val", user_id=b)
    assert repo.get("k", user_id=a) == "a-val"
    assert repo.get("k", user_id=b) == "b-val"
