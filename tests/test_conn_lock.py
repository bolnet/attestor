"""attestor.store.conn_lock — ONE re-entrant lock per DBAPI connection.

Review finding (Phase 3 governance): the worker shares a single psycopg2
connection across a 4-thread activity pool, and the raw-SQL helpers in
``compliance/retention.py`` bypassed the backend's ``_conn_lock``. The
registry here is the single source of truth: ``PostgresBackend`` and
every raw helper resolve the SAME lock for the same connection object.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from attestor.store import conn_lock
from tests._lockprobe import held_by_caller

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append(sql)
        self._conn.lock_held.append(held_by_caller(conn_lock.lock_for_connection(self._conn)))

    def fetchone(self) -> Any:
        return {"count": 2, "id": "x"}

    def fetchall(self) -> list[Any]:
        return [{"id": "a"}, {"id": "b"}]


class _Conn:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.lock_held: list[bool] = []
        self.cursor_kwargs: list[dict[str, Any]] = []
        self.commits = 0

    def cursor(self, **kw: Any) -> _Cursor:
        self.cursor_kwargs.append(kw)
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1


class _Slotted:
    """Not weak-referenceable and no ``__dict__`` — the worst-case connection."""

    __slots__ = ()


def test_same_connection_resolves_same_lock():
    conn = _Conn()
    assert conn_lock.lock_for_connection(conn) is conn_lock.lock_for_connection(conn)
    assert conn_lock.lock_for_connection(conn) is not conn_lock.lock_for_connection(_Conn())


def test_lock_is_reentrant():
    lock = conn_lock.lock_for_connection(_Conn())
    with lock, lock:  # a nested _execute inside a locked block must not deadlock
        assert held_by_caller(lock)
    assert not held_by_caller(lock)


def test_lock_serialises_threads():
    lock = conn_lock.lock_for_connection(_Conn())
    order: list[str] = []
    started = threading.Event()

    def holder() -> None:
        with lock:
            started.set()
            order.append("holder-in")
            threading.Event().wait(0.05)
            order.append("holder-out")

    t = threading.Thread(target=holder)
    t.start()
    started.wait()
    with lock:
        order.append("main")
    t.join()
    assert order == ["holder-in", "holder-out", "main"]


def test_non_weakrefable_connection_still_gets_a_lock():
    conn = _Slotted()
    lock = conn_lock.lock_for_connection(conn)
    with lock:
        assert held_by_caller(lock)


def test_locked_cursor_holds_the_lock_and_forwards_kwargs():
    conn = _Conn()
    with conn_lock.locked_cursor(conn, cursor_factory="dict") as cur:
        cur.execute("SELECT 1")
        assert held_by_caller(conn_lock.lock_for_connection(conn))
    assert conn.lock_held == [True]
    assert conn.cursor_kwargs == [{"cursor_factory": "dict"}]
    assert not held_by_caller(conn_lock.lock_for_connection(conn))


def test_postgres_backend_lock_is_the_registry_lock():
    """``_execute`` and the raw helpers must serialise on ONE lock."""
    from attestor.store.postgres_backend import PostgresBackend

    conn = _Conn()
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._conn = conn
    assert backend._get_conn_lock() is conn_lock.lock_for_connection(conn)
    backend._set_rls_user("u-1")
    assert conn.lock_held == [True]


# ── retention.py raw helpers go through the lock ────────────────────────

def test_retention_raw_helpers_execute_inside_the_connection_lock():
    from attestor.compliance import retention

    conn = _Conn()
    retention._count_user_doc_rows(conn, "u-1")
    retention._count_user_state_rows(conn, "u-1")
    retention._delete_user_doc_rows(conn, "u-1")
    retention._delete_user_state_rows(conn, "u-1")
    retention._count_matches(conn, ["status = %s"], ["active"])
    retention._apply_delete(conn, ["status = %s"], ["active"])
    retention._write_apply_audit(conn, policy_id="p-1", affected=2, initiated_by=None)
    assert conn.lock_held, "no SQL executed"
    assert all(conn.lock_held), f"unlocked SQL: {conn.executed}"
    assert not held_by_caller(conn_lock.lock_for_connection(conn))
