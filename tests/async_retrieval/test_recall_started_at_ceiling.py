"""Phase 5 — ``recall_started_at`` ceiling primitive + Postgres
integration.

Activates audit invariants **A2** (no post-recall writes leak in) and
**A6** (per-recall metadata propagates through ``asyncio.create_task``
/ ``asyncio.gather``).

The primitive itself is tested at the contextvar level; integration is
tested by capturing the SQL that ``PostgresBackend.search()`` /
``list_memories()`` emit when a scope is active.

See ``docs/plans/async-retrieval/PLAN.md`` § 4 — Phase 5.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


pytestmark = [pytest.mark.unit]


# ──────────────────────────────────────────────────────────────────────
# Primitive: recall_started_at_scope + current_recall_started_at
# ──────────────────────────────────────────────────────────────────────


def test_recall_started_at_default_returns_none():
    """Outside any scope, the ceiling is None — backwards-compat path
    for pre-Phase-5 callers."""
    from attestor.recall_context import current_recall_started_at
    assert current_recall_started_at() is None


def test_recall_started_at_scope_sets_ceiling():
    """Inside a scope, the ceiling is set to the explicit timestamp
    or ``NOW()`` when not specified."""
    from attestor.recall_context import (
        current_recall_started_at, recall_started_at_scope,
    )

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned) as ts:
        assert ts == pinned
        assert current_recall_started_at() == pinned


def test_recall_started_at_scope_resets_on_exit():
    """The contextvar must be restored on scope exit. A second
    independent scope must NOT see the first scope's value."""
    from attestor.recall_context import (
        current_recall_started_at, recall_started_at_scope,
    )

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned):
        assert current_recall_started_at() == pinned
    # Out of scope — back to None.
    assert current_recall_started_at() is None


def test_recall_started_at_default_is_now_utc():
    """When called without an explicit timestamp, the ceiling is the
    current UTC wall-clock — the same clock Postgres ``NOW()`` returns,
    so the comparison is meaningful across the wire."""
    from attestor.recall_context import recall_started_at_scope

    before = datetime.now(timezone.utc)
    with recall_started_at_scope() as ts:
        after = datetime.now(timezone.utc)
        assert ts.tzinfo == timezone.utc
        assert before <= ts <= after


# ──────────────────────────────────────────────────────────────────────
# Async propagation — A6: ContextVar copies on asyncio.create_task
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_started_at_propagates_to_async_tasks():
    """The audit-invariant load-bearing test: a recall_started_at_scope
    opened on the parent coroutine must be visible to every task spawned
    inside (including across asyncio.gather)."""
    from attestor.recall_context import (
        current_recall_started_at, recall_started_at_scope,
    )

    seen: list = []

    async def child():
        seen.append(current_recall_started_at())

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned):
        await asyncio.gather(child(), child(), child())

    assert seen == [pinned, pinned, pinned], (
        f"all 3 child tasks must see the same ceiling; got {seen}"
    )


@pytest.mark.asyncio
async def test_recall_started_at_async_scope_helper():
    """``recall_started_at_scope_async`` is a syntactic helper for
    ``async with``; it sets the same contextvar."""
    from attestor.recall_context import (
        current_recall_started_at, recall_started_at_scope_async,
    )

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    async with recall_started_at_scope_async(started_at=pinned):
        assert current_recall_started_at() == pinned
    assert current_recall_started_at() is None


# ──────────────────────────────────────────────────────────────────────
# Postgres backend integration: SQL must include the ceiling clause
# when a scope is active. Same stub-cursor pattern as
# tests/test_v4_namespace_roundtrip.py.
# ──────────────────────────────────────────────────────────────────────


def _build_v4_backend_capturing_sql(captured_sql: list, captured_params: list):
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = True

    def _capture(sql, params=None):
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    backend._execute = _capture
    backend._embed = lambda text: [0.0] * 1024
    return backend


def _build_v3_backend_capturing_sql(captured_sql: list, captured_params: list):
    from attestor.store.postgres_backend import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._v4 = False

    def _capture(sql, params=None):
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    backend._execute = _capture
    backend._embed = lambda text: [0.0] * 1024
    return backend


def test_postgres_search_includes_ceiling_when_scope_active():
    """Inside a recall_started_at_scope, search() SQL must include
    ``t_created <= %s`` as part of the WHERE clause, with the ceiling
    timestamp passed as a positional param."""
    from attestor.recall_context import recall_started_at_scope

    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_capturing_sql(sqls, params_log)

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned):
        backend.search("any query")

    assert sqls, "search did not call _execute"
    sql = sqls[0]
    assert "t_created <= %s" in sql, (
        f"v4 search must filter on t_created when ceiling active; "
        f"got SQL: {sql}"
    )
    assert pinned in (params_log[0] or []), (
        f"ceiling timestamp must be in params; got {params_log[0]}"
    )


def test_postgres_search_excludes_ceiling_when_no_scope():
    """No active scope = no ceiling clause = pre-Phase-5 behavior
    preserved (backwards-compat for callers that don't open a scope)."""
    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_capturing_sql(sqls, params_log)

    backend.search("any query")  # no scope

    sql = sqls[0]
    assert "t_created <= %s" not in sql, (
        f"no scope = no ceiling filter; got SQL with t_created clause: {sql}"
    )


def test_postgres_list_memories_includes_ceiling_when_scope_active():
    """Same ceiling propagation in the list_memories() WHERE builder
    (uses %(...)s named placeholders, not positional)."""
    from attestor.recall_context import recall_started_at_scope

    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_capturing_sql(sqls, params_log)

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned):
        backend.list_memories()

    sql = sqls[0]
    assert "t_created <= %(_recall_ceiling)s" in sql, (
        f"list_memories must filter on t_created when ceiling active; "
        f"got SQL: {sql}"
    )
    assert params_log[0].get("_recall_ceiling") == pinned


def test_postgres_v3_does_not_apply_ceiling():
    """v3 schema has no ``t_created`` column. The ceiling must NOT
    apply on v3 — backends without bi-temporal columns are out of
    Phase 5 scope and keep working as before."""
    from attestor.recall_context import recall_started_at_scope

    sqls: list = []
    params_log: list = []
    backend = _build_v3_backend_capturing_sql(sqls, params_log)

    pinned = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=pinned):
        backend.search("any query")

    sql = sqls[0]
    assert "t_created" not in sql, (
        f"v3 search must NOT reference t_created; got SQL: {sql}"
    )


# ──────────────────────────────────────────────────────────────────────
# A1 — as_of replay still works alongside the new ceiling
# ──────────────────────────────────────────────────────────────────────


def test_postgres_search_with_both_as_of_and_ceiling_active():
    """When BOTH as_of (audit invariant A1: replay past belief) and
    recall_started_at (A2: no post-recall writes) are active, both
    filters must be applied. They're independent — one captures
    'what we believed then', the other captures 'what's visible now
    in this snapshot'."""
    from attestor.recall_context import recall_started_at_scope

    sqls: list = []
    params_log: list = []
    backend = _build_v4_backend_capturing_sql(sqls, params_log)

    ceiling = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    as_of = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    with recall_started_at_scope(started_at=ceiling):
        backend.search("any query", as_of=as_of)

    sql = sqls[0]
    # Ceiling clause present.
    assert "t_created <= %s" in sql
    # As_of bi-temporal filter present (different shape).
    assert "tstzrange(valid_from" in sql
    # Both ceiling + as_of are in params.
    assert ceiling in (params_log[0] or [])
    assert as_of in (params_log[0] or [])
