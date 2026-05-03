"""Regression tests for the 5 tenancy / RBAC gaps surfaced by the
2026-05-02 4-domain audit (post-PR #135 supersession sweep).

Each test maps 1:1 to a production gap. The gaps were invisible because
the existing test suite either:

  - exercised the WRITE path for ``visibility`` but never asserted any
    READ path filtered on it
  - asserted vector / list_memories namespace filtering on v4 but left
    BM25 + tag_search unscoped
  - covered ``AgentContext.as_agent`` for read_only but on the WRITE
    capability matrix only — never confirmed read_only propagated
    through to the child
  - left ``max_writes_per_agent`` quota gate untested in CI

Tests that would currently REGRESS production are marked ``xfail`` with
``strict=True`` so flipping the underlying behavior surfaces the moment
the corresponding fix lands. After the fix lands in this same PR the
xfails are removed.

See post-PR-135 audit notes in long-term memory for the full taxonomy.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from attestor.context import AgentContext, AgentRole, Visibility


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _tag() -> str:
    """Per-test unique entity/category suffix to isolate state."""
    return uuid.uuid4().hex[:10]


def _build_v4_backend_stub(
    captured_sql: list, captured_params: list,
) -> Any:
    """Construct a PostgresBackend in v4 mode without going through
    schema init. Mirrors the helper in ``test_v4_namespace_roundtrip``
    so the unit tests don't need a live DB.
    """
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = True

    def _capture(sql: str, params: Any | None = None) -> list:
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    backend._execute = _capture
    return backend


def _build_v3_backend_stub(
    captured_sql: list, captured_params: list,
) -> Any:
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = False

    def _capture(sql: str, params: Any | None = None) -> list:
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    backend._execute = _capture
    return backend


# ──────────────────────────────────────────────────────────────────────
# Gap A1 — visibility column is written but never filtered on read.
#
# ``_postgres_document.py:96`` writes ``visibility VARCHAR(32) DEFAULT
# 'team'``; ``Visibility.PRIVATE`` exists in ``context.py``. But neither
# ``list_memories``, ``get``, ``tag_search`` nor any vector/BM25 path
# enforces ``visibility = 'private' AND agent_id != requester_agent_id``.
# Two agents in the same user/namespace can read each other's PRIVATE
# memories.
#
# Backward-compat constraint: when the caller does NOT pass a
# ``requester_agent_id``, visibility is NOT filtered (preserves every
# existing call site). When the caller DOES pass one, private rows
# belonging to OTHER agents must be excluded.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_v4_list_memories_no_requester_skips_visibility_filter() -> None:
    """No requester_agent_id → no visibility filter (back-compat)."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.list_memories()

    assert sqls
    sql = sqls[0]
    assert "visibility" not in sql, (
        f"caller did not pass requester_agent_id — visibility filter "
        f"must NOT be added; got SQL: {sql}"
    )


@pytest.mark.unit
def test_v4_list_memories_requester_filters_private_rows() -> None:
    """When the caller passes ``requester_agent_id``, the SQL must
    exclude rows where ``visibility='private' AND agent_id != requester``.
    """
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.list_memories(requester_agent_id="agent-a")

    assert sqls
    sql = sqls[0]
    assert "visibility" in sql, (
        f"requester_agent_id passed — visibility filter must be added; "
        f"got SQL: {sql}"
    )
    # The exact SQL we settled on: parens around the OR so it composes
    # safely with other AND filters.
    assert "agent_id" in sql, (
        f"visibility predicate must reference agent_id; got SQL: {sql}"
    )
    assert params_log[0].get("requester_agent_id") == "agent-a"


@pytest.mark.unit
def test_v4_get_with_requester_filters_private_other_agents() -> None:
    """``get(memory_id, requester_agent_id=...)`` must filter on
    visibility too — the per-id read path is the obvious bypass."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.get(
        "00000000-0000-0000-0000-000000000abc",
        requester_agent_id="agent-a",
    )

    assert sqls
    sql = sqls[0]
    assert "visibility" in sql, (
        f"get() with requester_agent_id must filter visibility; "
        f"got SQL: {sql}"
    )


@pytest.mark.unit
def test_v4_tag_search_with_requester_filters_private() -> None:
    """``tag_search`` must respect visibility too — same reasoning as
    ``list_memories``."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.tag_search(
        tags=["finance"],
        namespace="default",
        requester_agent_id="agent-a",
    )

    assert sqls
    sql = sqls[0]
    assert "visibility" in sql, (
        f"tag_search() with requester_agent_id must filter visibility; "
        f"got SQL: {sql}"
    )


# ──────────────────────────────────────────────────────────────────────
# Gap A2 — BM25 lane has zero namespace scoping.
#
# ``BM25Lane.search()`` claims "RLS scopes the search automatically",
# but the RLS policy at ``memories`` filters by user_id only. For the
# LME-S benchmark setup (one user, many per-sample namespaces) BM25
# leaks across namespaces — the same bug PR #77 fixed for vector.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_bm25_search_signature_accepts_namespace() -> None:
    """``BM25Lane.search`` must accept a ``namespace`` keyword.

    Pre-fix the parameter didn't exist; callers with namespace=...
    got a TypeError. We pin the signature so the orchestrator wire-up
    can flow namespace through without crashing.
    """
    from attestor.retrieval.bm25 import BM25Lane

    import inspect
    sig = inspect.signature(BM25Lane.search)
    assert "namespace" in sig.parameters, (
        f"BM25Lane.search must accept namespace kw; got {list(sig.parameters)}"
    )


@pytest.mark.unit
def test_bm25_search_namespace_filter_in_sql() -> None:
    """When ``namespace`` is passed, the generated SQL must restrict to
    ``metadata->>'_namespace' = %(...)s``. Mirrors v4 vector / list."""
    from attestor.retrieval.bm25 import BM25Lane

    captured_sql: list = []
    captured_params: list = []

    class _StubCursor:
        def __enter__(self) -> "_StubCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: dict[str, Any]) -> None:
            captured_sql.append(sql)
            captured_params.append(params)

        def fetchall(self) -> list:
            return []

    class _StubConn:
        def cursor(self) -> _StubCursor:
            return _StubCursor()

    lane = BM25Lane(_StubConn())
    lane.search("dark mode", namespace="lme_sample_42")

    assert captured_sql, "BM25Lane.search did not execute SQL"
    sql = captured_sql[0]
    assert "metadata->>'_namespace'" in sql, (
        f"BM25 search must filter on metadata->>'_namespace' when "
        f"namespace passed; got SQL: {sql}"
    )
    assert captured_params[0].get("namespace") == "lme_sample_42"


@pytest.mark.unit
def test_bm25_search_no_namespace_skips_filter() -> None:
    """No namespace → no filter (back-compat with the SOLO single-tenant
    happy path)."""
    from attestor.retrieval.bm25 import BM25Lane

    captured_sql: list = []
    captured_params: list = []

    class _StubCursor:
        def __enter__(self) -> "_StubCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: dict[str, Any]) -> None:
            captured_sql.append(sql)
            captured_params.append(params)

        def fetchall(self) -> list:
            return []

    class _StubConn:
        def cursor(self) -> _StubCursor:
            return _StubCursor()

    lane = BM25Lane(_StubConn())
    lane.search("dark mode")

    assert captured_sql
    sql = captured_sql[0]
    assert "metadata->>'_namespace'" not in sql, (
        f"unscoped BM25 search must NOT add the namespace filter; "
        f"got SQL: {sql}"
    )


# ──────────────────────────────────────────────────────────────────────
# Gap A3 — ``tag_search`` broken on v4.
#
# Pre-fix it referenced ``namespace = %(namespace)s`` (column gone in
# v4) and ``ORDER BY created_at`` (column gone in v4). On v4 calls it
# raised ``UndefinedColumn``. The mixin already branches on ``self._v4``
# for ``list_memories``; we need the same branch here.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_v4_tag_search_uses_metadata_namespace_filter() -> None:
    """v4 ``tag_search(namespace=...)`` must filter via
    ``metadata->>'_namespace'``, not the dropped column."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.tag_search(tags=["finance"], namespace="tenant-acme")

    assert sqls
    sql = sqls[0]
    assert "metadata->>'_namespace' = %(namespace)s" in sql, (
        f"v4 tag_search must filter on metadata->>'_namespace'; got SQL: {sql}"
    )
    assert "namespace = %(namespace)s" not in sql, (
        f"v4 tag_search must NOT reference dropped namespace column; "
        f"got SQL: {sql}"
    )


@pytest.mark.unit
def test_v4_tag_search_orders_by_t_created() -> None:
    """v4 has ``t_created TIMESTAMPTZ`` (not ``created_at TEXT``).
    Pre-fix the ORDER BY referenced ``created_at`` and crashed with
    UndefinedColumn on every v4 call."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_stub(sqls, params_log)

    backend.tag_search(tags=["finance"])

    assert sqls
    sql = sqls[0]
    assert "ORDER BY t_created" in sql, (
        f"v4 tag_search must ORDER BY t_created; got SQL: {sql}"
    )


@pytest.mark.unit
def test_v3_tag_search_keeps_legacy_columns() -> None:
    """v3 keeps ``namespace = %(namespace)s`` and ``ORDER BY created_at``
    — regression guard for legacy v3 deployments."""
    sqls: list = []
    params_log: list = []
    backend = _build_v3_backend_stub(sqls, params_log)

    backend.tag_search(tags=["finance"], namespace="legacy-tenant")

    assert sqls
    sql = sqls[0]
    assert "namespace = %(namespace)s" in sql
    assert "ORDER BY created_at" in sql
    assert "metadata->>" not in sql


# ──────────────────────────────────────────────────────────────────────
# Gap A4 — ``AgentContext.as_agent`` doesn't propagate ``read_only=True``
# from the parent.
#
# Pre-fix: the child's ``read_only`` came from the explicit kwarg only
# (default=False). A read-only orchestrator could spawn a writeable
# sub-agent simply by not passing read_only=True to as_agent — the
# kill-switch leaked.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_as_agent_propagates_read_only_from_parent() -> None:
    """Read-only parent → read-only child by default. Caller can still
    explicitly pass read_only=False to override (different concern; we
    pin the SAFE default here)."""
    parent = AgentContext(
        agent_id="orchestrator-01",
        role=AgentRole.ORCHESTRATOR,
        read_only=True,
    )

    child = parent.as_agent("worker-01", role=AgentRole.EXECUTOR)

    assert child.read_only is True, (
        "read_only=True on parent must propagate to child by default; "
        "otherwise the kill-switch leaks the moment the orchestrator "
        "spawns a sub-agent"
    )


@pytest.mark.unit
def test_as_agent_writeable_parent_keeps_default_writeable_child() -> None:
    """The fix must not turn the writeable case into read-only. Pin
    the symmetric guarantee."""
    parent = AgentContext(
        agent_id="orchestrator-01",
        role=AgentRole.ORCHESTRATOR,
        read_only=False,
    )
    child = parent.as_agent("worker-01")
    assert child.read_only is False


# ──────────────────────────────────────────────────────────────────────
# Gap A5 — write-quota dead code.
#
# ``AgentContext.add_memory`` checks ``len(memories_written) >=
# max_writes_per_agent`` and raises RuntimeError. The branch has zero
# test references; nothing in CI ever exercised it. The test pins the
# behavior so a future refactor can't silently drop the gate.
# ──────────────────────────────────────────────────────────────────────


class _FakeMemoryBackend:
    """Minimal AgentMemory stub for AgentContext.add_memory.

    Returns a fresh Memory with an incrementing id so the context
    tracker can append to memories_written without hitting any real
    backend. Visibility / namespace plumbing isn't exercised here —
    Gap A1 covers that.
    """

    def __init__(self) -> None:
        self._counter = 0

    def add(self, **kwargs: Any) -> Any:
        self._counter += 1

        class _M:
            id = f"mem-{self._counter}"

        return _M()


@pytest.mark.unit
def test_add_memory_quota_gate_raises_on_overflow() -> None:
    """Set quota=2; the third add_memory must raise."""
    ctx = AgentContext(
        agent_id="agent-a",
        role=AgentRole.EXECUTOR,
        memory=_FakeMemoryBackend(),
        max_writes_per_agent=2,
    )

    ctx.add_memory("first")
    ctx.add_memory("second")

    with pytest.raises(RuntimeError, match=r"(?i)quota"):
        ctx.add_memory("third")


@pytest.mark.unit
def test_add_memory_quota_gate_allows_under_limit() -> None:
    """Below the limit, no raise. Pin the negative case so the gate
    can't be over-tightened (e.g. off-by-one) in a future refactor."""
    ctx = AgentContext(
        agent_id="agent-a",
        role=AgentRole.EXECUTOR,
        memory=_FakeMemoryBackend(),
        max_writes_per_agent=3,
    )
    for i in range(3):
        ctx.add_memory(f"msg-{i}")
    assert len(ctx.memories_written) == 3


# ──────────────────────────────────────────────────────────────────────
# Visibility — end-to-end live test (v4 Postgres) confirming Gap A1 is
# closed at the row layer, not just the SQL stub.
#
# Uses raw psycopg2 against the v4 schema (mirrors test_v4_postgres_backend
# / test_bm25_lane). The conftest ``mem`` fixture boots v3 only — visibility
# is a v4-schema column so we go around it.
# ──────────────────────────────────────────────────────────────────────


def _v4_pg_url() -> str:
    import os
    return os.environ.get(
        "PG_TEST_URL",
        "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
    )


def _v4_pg_reachable() -> bool:
    try:
        import psycopg2

        c = psycopg2.connect(_v4_pg_url(), connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(
    not _v4_pg_reachable(),
    reason="local v4 Postgres unreachable",
)
def test_visibility_private_blocks_other_agent_v4_live() -> None:
    """End-to-end: write a memory with visibility='private' under
    agent-a; ``list_memories(requester_agent_id='agent-b')`` must not
    see it; ``list_memories(requester_agent_id='agent-a')`` must.
    Confirms the SQL filter we asserted in the unit tests above
    actually drops the row at the row layer.
    """
    import psycopg2
    from pathlib import Path
    from attestor.store.postgres_backend import PostgresBackend

    schema_path = (
        Path(__file__).resolve().parent.parent
        / "attestor" / "store" / "schema.sql"
    )

    # Bootstrap the v4 schema fresh — drop and reload, mirrors the
    # ``fresh_schema`` fixture in test_bm25_lane / test_v4_postgres_backend.
    admin = psycopg2.connect(_v4_pg_url())
    admin.autocommit = True
    try:
        raw = schema_path.read_text().replace("{embedding_dim}", "1024")
        with admin.cursor() as cur:
            for tbl in (
                "memories", "episodes", "sessions",
                "projects", "users", "user_quotas",
                "audit_log",
            ):
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            cur.execute(raw)

        # Build a backend that points at the prepared schema. We use
        # __new__ to skip _init_schema (the schema is already in place
        # and the backend bootstrap currently only knows about the v3
        # inline DDL — pre-existing limitation, not in scope here).
        backend = PostgresBackend.__new__(PostgresBackend)
        backend._conn = admin
        backend._v4 = True
        backend._embedder = None
        backend._embedding_fn = None
        backend._embedding_dim = 1024

        from attestor.store.postgres_backend import PostgresBackend as _PB
        backend._execute = _PB._execute.__get__(backend, _PB)
        backend._execute_scalar = _PB._execute_scalar.__get__(backend, _PB)

        # Seed a user (FK requires it) and INSERT two memories: one
        # PRIVATE under agent-a, one TEAM under agent-a. agent-b must
        # only see the TEAM row.
        uid = str(uuid.uuid4())
        with admin.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, external_id) VALUES (%s, %s)",
                (uid, f"u-vis-{uuid.uuid4().hex[:8]}"),
            )

        cat = f"vis_{_tag()}"
        from attestor.models import Memory

        private_row = backend.insert(Memory(
            content="private to agent-a",
            category=cat,
            user_id=uid,
            agent_id="agent-a",
            visibility="private",
        ))
        team_row = backend.insert(Memory(
            content="team-readable",
            category=cat,
            user_id=uid,
            agent_id="agent-a",
            visibility="team",
        ))

        # No requester → no filter (back-compat). Both rows visible.
        unfiltered = backend.list_memories(category=cat)
        ids_unfiltered = {m.id for m in unfiltered}
        assert private_row.id in ids_unfiltered
        assert team_row.id in ids_unfiltered

        # agent-a (the writer) sees both.
        a_view = backend.list_memories(
            category=cat, requester_agent_id="agent-a",
        )
        a_ids = {m.id for m in a_view}
        assert private_row.id in a_ids, (
            "writer must see its own private row"
        )
        assert team_row.id in a_ids

        # agent-b (other agent) sees only the team row.
        b_view = backend.list_memories(
            category=cat, requester_agent_id="agent-b",
        )
        b_ids = {m.id for m in b_view}
        assert team_row.id in b_ids, (
            "team-visibility row must be readable by other agents"
        )
        assert private_row.id not in b_ids, (
            f"agent-b must NOT see agent-a's private row; got {b_ids}"
        )
    finally:
        admin.close()
