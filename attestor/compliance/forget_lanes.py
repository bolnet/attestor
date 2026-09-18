# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Per-backend GDPR forget lanes — the units the durable ``ForgetUser`` saga retries.

``retention.forget_user`` walks every backend in one call and records
failures on the result. The saga needs each backend as its own retryable
unit instead, with two extra properties:

* the audit row is written under a caller-minted ``audit_id`` with
  ``ON CONFLICT (id) DO NOTHING`` — an activity retry after a committed
  insert never produces a second audit row;
* every lane raises on a real backend error (so Temporal retries) and
  rolls the connection back first, so the next attempt does not start
  inside an aborted transaction.

No Temporal import here: this module is plain SQL + backend calls and is
also usable in-process.

Thread safety: the worker runs these lanes on a thread pool over ONE
shared connection, so every statement runs under
``attestor.store.conn_lock.lock_for_connection(conn)`` — the same
re-entrant lock ``PostgresBackend._execute`` holds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from attestor.compliance.retention import _conn_of, _count_user_doc_rows
from attestor.store.conn_lock import lock_for_connection

logger = logging.getLogger("attestor.compliance.forget_lanes")

AUDIT_SCOPE_USER = "user"
_MISSING_STATE_TABLE_MARKERS: tuple[str, ...] = (
    'relation "state" does not exist',
    "undefinedtable",
)


@dataclass(frozen=True)
class UserRowCounts:
    doc_rows: int
    state_rows: int


@dataclass(frozen=True)
class LaneDelete:
    """Rows removed by one lane; ``skipped`` = backend absent / unsupported."""

    deleted: int = 0
    skipped: bool = False


@dataclass(frozen=True)
class GraphDelete:
    nodes: int = 0
    edges: int = 0
    skipped: bool = False


def connection_of(mem: Any) -> Any:
    """The psycopg2 connection behind ``mem`` (AgentMemory / store / raw)."""
    return _conn_of(mem)


def _commit(conn: Any) -> None:
    if hasattr(conn, "commit"):
        conn.commit()


def _rollback(conn: Any) -> None:
    rollback = getattr(conn, "rollback", None)
    if rollback is None:
        return
    try:
        rollback()
    except Exception as exc:
        logger.debug("rollback after failed forget lane raised: %s", exc)


def _is_missing_state_table(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _MISSING_STATE_TABLE_MARKERS)


def _scalar(row: Any, key: str) -> int:
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get(key, 0))
    return int(row[0])


def _require_id(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _run(conn: Any, sql: str, params: tuple[Any, ...], *, commit: bool = True) -> list[Any]:
    """ONE statement under the connection lock; commit, or roll back and re-raise."""
    with lock_for_connection(conn):
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = list(cur.fetchall() or [])
            if commit:
                _commit(conn)
        except Exception:
            _rollback(conn)
            raise
    return rows


# ── counts + audit ───────────────────────────────────────────────────────

def count_user_rows(mem: Any, user_id: str) -> UserRowCounts:
    """Pre-delete counts for the audit row (state table optional on v3)."""
    conn = connection_of(mem)
    with lock_for_connection(conn):  # both counts from one consistent view
        doc_rows = _count_user_doc_rows(conn, user_id)
        state_rows = _count_user_state_rows(conn, user_id)
    return UserRowCounts(doc_rows=doc_rows, state_rows=state_rows)


def _count_user_state_rows(conn: Any, user_id: str) -> int:
    try:
        rows = _run(
            conn, "SELECT COUNT(*) AS count FROM state WHERE user_id = %s", (user_id,),
            commit=False,
        )
    except Exception as exc:
        if _is_missing_state_table(exc):
            return 0
        raise
    return _scalar(rows[0] if rows else None, "count")


def write_forget_audit(
    mem: Any,
    *,
    audit_id: str,
    user_id: str,
    doc_rows: int,
    state_rows: int,
    initiated_by: str | None,
) -> str:
    """Append the ``forget_audit`` row under ``audit_id`` (idempotent by id).

    Vector / graph counts are unknown before the wipe and are recorded as
    0, exactly like the in-process path; the saga result carries the
    actual per-backend counts.
    """
    _require_id(audit_id, "audit_id")
    _require_id(user_id, "user_id")
    rows = _run(
        connection_of(mem),
        "INSERT INTO forget_audit (id, scope, target_user_id, policy_id, "
        "doc_rows_deleted, vector_rows_deleted, graph_nodes_deleted, "
        "graph_edges_deleted, state_rows_deleted, initiated_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (id) DO NOTHING RETURNING id",
        (
            audit_id, AUDIT_SCOPE_USER, user_id, None,
            int(doc_rows), 0, 0, 0, int(state_rows), initiated_by,
        ),
    )
    if not rows:
        logger.info("forget_audit %s already present (idempotent retry)", audit_id)
    return audit_id


# ── backend lanes ────────────────────────────────────────────────────────

def forget_doc(mem: Any, user_id: str) -> int:
    """Physically delete the user's ``memories`` rows; returns the count."""
    rows = _run(
        connection_of(mem), "DELETE FROM memories WHERE user_id = %s RETURNING id", (user_id,),
    )
    return len(rows)


def forget_state(mem: Any, user_id: str) -> LaneDelete:
    """Delete the user's ``state`` rows; a v3 store without the table is skipped."""
    try:
        rows = _run(
            connection_of(mem), "DELETE FROM state WHERE user_id = %s RETURNING id", (user_id,),
        )
    except Exception as exc:
        if _is_missing_state_table(exc):
            return LaneDelete(deleted=0, skipped=True)
        raise
    return LaneDelete(deleted=len(rows))


class StoreNotInitialisedError(RuntimeError):
    """A derived store is configured but failed to initialise (outage at
    start-up). Raised instead of reporting ``skipped`` so the durable saga
    retries the lane; a lane that is genuinely unconfigured stays skipped."""


def _require_initialised(mem: Any, store: Any, *, role: str, flag: str) -> None:
    if store is None and getattr(mem, flag, False):
        raise StoreNotInitialisedError(
            f"{role} store is configured but not initialised (backend outage at start-up)"
        )


def forget_vector(mem: Any, user_id: str) -> LaneDelete:
    """``delete_by_user`` on the vector backend; unconfigured → skipped;
    configured-but-uninitialised → :class:`StoreNotInitialisedError`."""
    store = getattr(mem, "_vector_store", None)
    _require_initialised(mem, store, role="vector", flag="_vector_init_failed")
    method = getattr(store, "delete_by_user", None) if store is not None else None
    if method is None:
        return LaneDelete(deleted=0, skipped=True)
    return LaneDelete(deleted=int(method(user_id) or 0))


def forget_graph(mem: Any, user_id: str) -> GraphDelete:
    """``delete_by_user`` on the graph backend → ``(nodes, edges)``; absent → skipped."""
    graph = getattr(mem, "_graph", None)
    _require_initialised(mem, graph, role="graph", flag="_graph_init_failed")
    method = getattr(graph, "delete_by_user", None) if graph is not None else None
    if method is None:
        return GraphDelete(skipped=True)
    nodes, edges = method(user_id)
    return GraphDelete(nodes=int(nodes or 0), edges=int(edges or 0))
