"""attestor.compliance.forget_lanes — per-backend GDPR lanes the saga calls.

Fake psycopg2-shaped connection; asserts the SQL shape (explicit audit id,
``ON CONFLICT DO NOTHING``), rollback on error, and the vector/graph
"skipped" semantics when a backend lacks ``delete_by_user``.
"""

from __future__ import annotations

from typing import Any

import pytest

from attestor.compliance import forget_lanes as lanes
from attestor.store.conn_lock import lock_for_connection
from tests._lockprobe import held_by_caller

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._rows: list[Any] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append((sql, tuple(params or ())))
        self._conn.lock_held.append(held_by_caller(lock_for_connection(self._conn)))
        if self._conn.fail is not None:
            raise self._conn.fail
        low = sql.lower()
        if "insert into forget_audit" in low:
            audit_id = params[0]
            if audit_id in self._conn.audit_ids:
                self._rows = []  # ON CONFLICT DO NOTHING → no row returned
            else:
                self._conn.audit_ids.append(audit_id)
                self._rows = [{"id": audit_id}]
        elif "count(*)" in low and "from memories" in low:
            self._rows = [{"count": self._conn.doc_rows}]
        elif "count(*)" in low and "from state" in low:
            if self._conn.no_state_table:
                raise RuntimeError('relation "state" does not exist')
            self._rows = [{"count": self._conn.state_rows}]
        elif low.startswith("delete from memories"):
            self._rows = [{"id": f"m{i}"} for i in range(self._conn.doc_rows)]
        elif low.startswith("delete from state"):
            if self._conn.no_state_table:
                raise RuntimeError('relation "state" does not exist')
            self._rows = [{"id": f"s{i}"} for i in range(self._conn.state_rows)]
        else:
            self._rows = []

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class _Conn:
    def __init__(self, *, doc_rows: int = 3, state_rows: int = 2, no_state_table: bool = False,
                 fail: Exception | None = None) -> None:
        self.doc_rows, self.state_rows = doc_rows, state_rows
        self.no_state_table = no_state_table
        self.fail = fail
        self.executed: list[tuple[str, tuple]] = []
        self.lock_held: list[bool] = []
        self.audit_ids: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **_: Any) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _Store:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn


class _Mem:
    def __init__(self, conn: _Conn, vector: Any = None, graph: Any = None) -> None:
        self._store = _Store(conn)
        self._vector_store = vector
        self._graph = graph


class _Vector:
    def __init__(self, n: int = 5, fail: Exception | None = None) -> None:
        self.n, self.fail = n, fail
        self.calls: list[str] = []

    def delete_by_user(self, user_id: str) -> int:
        self.calls.append(user_id)
        if self.fail:
            raise self.fail
        return self.n


class _Graph:
    def delete_by_user(self, user_id: str) -> tuple[int, int]:
        return (4, 7)


def test_count_user_rows_reads_doc_and_state():
    counts = lanes.count_user_rows(_Mem(_Conn(doc_rows=9, state_rows=4)), "u-1")
    assert (counts.doc_rows, counts.state_rows) == (9, 4)


def test_count_user_rows_tolerates_missing_state_table():
    counts = lanes.count_user_rows(_Mem(_Conn(no_state_table=True)), "u-1")
    assert counts.state_rows == 0


def test_write_forget_audit_uses_explicit_id_and_is_idempotent():
    conn = _Conn()
    mem = _Mem(conn)
    first = lanes.write_forget_audit(
        mem, audit_id="a-1", user_id="u-1", doc_rows=3, state_rows=2, initiated_by="ops",
    )
    second = lanes.write_forget_audit(
        mem, audit_id="a-1", user_id="u-1", doc_rows=3, state_rows=2, initiated_by="ops",
    )
    assert first == second == "a-1"
    assert conn.audit_ids == ["a-1"]  # retry after commit did not double-write
    sql, params = conn.executed[0]
    assert "on conflict (id) do nothing" in sql.lower()
    assert params[:3] == ("a-1", "user", "u-1")
    assert "ops" in params
    assert conn.commits >= 1


def test_write_forget_audit_rejects_blank_ids():
    with pytest.raises(ValueError, match="audit_id"):
        lanes.write_forget_audit(_Mem(_Conn()), audit_id="", user_id="u", doc_rows=0,
                                 state_rows=0, initiated_by=None)
    with pytest.raises(ValueError, match="user_id"):
        lanes.write_forget_audit(_Mem(_Conn()), audit_id="a", user_id=" ", doc_rows=0,
                                 state_rows=0, initiated_by=None)


def test_forget_doc_deletes_and_commits():
    conn = _Conn(doc_rows=3)
    assert lanes.forget_doc(_Mem(conn), "u-1") == 3
    sql, params = conn.executed[-1]
    assert sql.lower().startswith("delete from memories where user_id = %s")
    assert params == ("u-1",)
    assert conn.commits == 1


def test_forget_doc_rolls_back_and_reraises_on_error():
    conn = _Conn(fail=RuntimeError("connection reset"))
    with pytest.raises(RuntimeError, match="connection reset"):
        lanes.forget_doc(_Mem(conn), "u-1")
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_forget_state_deletes_or_skips_when_table_missing():
    assert lanes.forget_state(_Mem(_Conn(state_rows=2)), "u-1") == lanes.LaneDelete(deleted=2)
    out = lanes.forget_state(_Mem(_Conn(no_state_table=True)), "u-1")
    assert out == lanes.LaneDelete(deleted=0, skipped=True)


def test_forget_state_reraises_real_errors():
    conn = _Conn(fail=RuntimeError("deadlock detected"))
    with pytest.raises(RuntimeError, match="deadlock"):
        lanes.forget_state(_Mem(conn), "u-1")
    assert conn.rollbacks == 1


def test_forget_vector_counts_skips_and_raises():
    vec = _Vector(n=5)
    assert lanes.forget_vector(_Mem(_Conn(), vector=vec), "u-1") == lanes.LaneDelete(deleted=5)
    assert vec.calls == ["u-1"]
    assert lanes.forget_vector(_Mem(_Conn(), vector=None), "u-1") == lanes.LaneDelete(
        deleted=0, skipped=True,
    )
    assert lanes.forget_vector(_Mem(_Conn(), vector=object()), "u-1") == lanes.LaneDelete(
        deleted=0, skipped=True,
    )
    failing = _Vector(fail=RuntimeError("pinecone down"))
    with pytest.raises(RuntimeError, match="pinecone down"):
        lanes.forget_vector(_Mem(_Conn(), vector=failing), "u-1")


def test_forget_graph_counts_nodes_and_edges():
    out = lanes.forget_graph(_Mem(_Conn(), graph=_Graph()), "u-1")
    assert out == lanes.GraphDelete(nodes=4, edges=7)
    assert lanes.forget_graph(_Mem(_Conn(), graph=None), "u-1") == lanes.GraphDelete(
        nodes=0, edges=0, skipped=True,
    )


def test_forget_lanes_never_import_durable_or_temporalio():
    import subprocess
    import sys

    code = (
        "import sys, attestor.compliance.forget_lanes, attestor.identity.sessions\n"
        "print(','.join(sorted(m for m in sys.modules "
        "if m.startswith(('temporalio', 'attestor.durable')))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


# ── every lane serialises on the connection lock (worker thread pool) ───

@pytest.mark.parametrize("run", [
    lambda mem: lanes.count_user_rows(mem, "u-1"),
    lambda mem: lanes.write_forget_audit(
        mem, audit_id="a-1", user_id="u-1", doc_rows=1, state_rows=1, initiated_by=None,
    ),
    lambda mem: lanes.forget_doc(mem, "u-1"),
    lambda mem: lanes.forget_state(mem, "u-1"),
])
def test_lane_sql_runs_inside_the_connection_lock(run):
    conn = _Conn()
    run(_Mem(conn))
    assert conn.lock_held, "lane executed no SQL"
    assert all(conn.lock_held), f"unlocked SQL: {conn.executed}"
    assert not held_by_caller(lock_for_connection(conn)), "lock leaked after the lane"


def test_lane_releases_the_lock_after_a_failure():
    conn = _Conn(fail=RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        lanes.forget_doc(_Mem(conn), "u-1")
    assert conn.rollbacks == 1
    assert not held_by_caller(lock_for_connection(conn))


# ── configured-but-uninitialised backend is an OUTAGE, not "skipped" ────
#
# Live e2e 2026-09-03: the worker built its AgentMemory while Neo4j was
# down, ``_graph`` was None, the graph lane reported ``skipped`` and the
# saga COMPLETED with the user's nodes still in the graph.

def _outage_mem(*, vector_failed: bool = False, graph_failed: bool = False) -> _Mem:
    m = _Mem(_Conn(), vector=None, graph=None)
    m._vector_init_failed = vector_failed
    m._graph_init_failed = graph_failed
    return m


def test_forget_graph_raises_when_graph_store_failed_to_initialise():
    with pytest.raises(lanes.StoreNotInitialisedError, match="graph"):
        lanes.forget_graph(_outage_mem(graph_failed=True), "u-1")


def test_forget_vector_raises_when_vector_store_failed_to_initialise():
    with pytest.raises(lanes.StoreNotInitialisedError, match="vector"):
        lanes.forget_vector(_outage_mem(vector_failed=True), "u-1")


def test_forget_lanes_still_skip_when_role_is_not_configured():
    assert lanes.forget_graph(_outage_mem(), "u-1").skipped is True
    assert lanes.forget_vector(_outage_mem(), "u-1").skipped is True
