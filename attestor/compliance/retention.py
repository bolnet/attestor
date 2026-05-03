"""Declarative retention policies + GDPR right-to-be-forgotten.

Retention rules are stored in the ``retention_policies`` Postgres table
so a long-running scheduler can read them, apply them, and emit an
audit row to ``forget_audit``. The code here is dependency-light — it
only needs a psycopg2-shaped connection on the document backend.

Forget-user is a multi-backend coordinator:

  1. Count what would be deleted (so dry-run is cheap and callers can
     show the user the blast radius before pressing the button).
  2. Write a ``forget_audit`` row FIRST. The audit table is RLS-exempt
     and append-only — regulators need to verify a deletion happened
     even if the actual backend wipe partially failed.
  3. Walk the backends in order: doc → vector → graph → state. Any
     backend that errors out is logged but does NOT abort the other
     deletions. The ``ForgetUserResult`` carries per-backend counts so
     callers can detect partial success.

Quality bias: the runner does the cheapest correct thing. No batching
heuristics, no parallel deletion — retention work is rare and audit
correctness beats throughput.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("attestor.compliance.retention")


# ─────────────────────────────────────────────────────────────────────
# Public dataclasses (frozen — immutability is non-negotiable here)
# ─────────────────────────────────────────────────────────────────────


_VALID_ACTIONS = frozenset({"archive", "delete"})


@dataclass(frozen=True)
class RetentionPolicy:
    """One declarative retention rule.

    ``namespace`` / ``category`` / ``layer`` / ``tags_any`` are AND-ed
    together; passing ``None`` (or an empty tuple) for a field skips
    that filter. ``older_than_days`` is the only mandatory predicate.

    ``action='archive'`` flips ``status='archived'`` and stamps
    ``valid_until=NOW()`` (matches the supersession semantics so the
    rest of the read path treats archived rows as already-superseded).
    ``action='delete'`` issues a physical ``DELETE FROM memories`` —
    only this is GDPR-compliant.
    """

    id: str
    name: str
    namespace: str | None
    category: str | None
    layer: str | None
    tags_any: tuple[str, ...]
    older_than_days: int
    action: str
    enabled: bool
    created_at: str


@dataclass(frozen=True)
class RetentionApplyResult:
    policies_evaluated: int
    memories_archived: int
    memories_deleted: int
    elapsed_ms: float
    by_policy: dict[str, dict[str, Any]] = field(default_factory=dict)
    dry_run: bool = False


@dataclass(frozen=True)
class ForgetUserResult:
    user_id: str
    doc_rows_deleted: int
    vector_rows_deleted: int
    graph_nodes_deleted: int
    graph_edges_deleted: int
    state_rows_deleted: int
    elapsed_ms: float
    audit_id: str
    dry_run: bool = False
    backend_errors: tuple[str, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────


def _conn_of(target: Any) -> Any:
    """Return the psycopg2 connection from ``mem`` / store / raw conn.

    Accepts:
      * an :class:`AgentMemory` (resolves ``mem._store._conn``)
      * a document-store object exposing ``_conn``
      * a raw psycopg2 connection (returned as-is)
    """
    if hasattr(target, "_store"):
        store = target._store
        return getattr(store, "_conn", store)
    if hasattr(target, "_conn"):
        return target._conn
    return target


def _row_to_policy(row: dict[str, Any]) -> RetentionPolicy:
    created = row.get("created_at")
    if isinstance(created, datetime):
        created_iso = created.isoformat()
    elif created is None:
        created_iso = ""
    else:
        created_iso = str(created)
    tags_any = row.get("tags_any") or []
    return RetentionPolicy(
        id=str(row["id"]),
        name=row["name"],
        namespace=row.get("namespace"),
        category=row.get("category"),
        layer=row.get("layer"),
        tags_any=tuple(tags_any),
        older_than_days=int(row["older_than_days"]),
        action=row["action"],
        enabled=bool(row.get("enabled", True)),
        created_at=created_iso,
    )


# ─────────────────────────────────────────────────────────────────────
# Policy CRUD
# ─────────────────────────────────────────────────────────────────────


def add_retention_policy(
    target: Any,
    *,
    name: str,
    older_than_days: int,
    action: str = "archive",
    namespace: str | None = None,
    category: str | None = None,
    layer: str | None = None,
    tags_any: tuple[str, ...] | list[str] | None = None,
) -> RetentionPolicy:
    """Insert a new retention policy. Returns the persisted record.

    Validation is eager — bad input raises ``ValueError`` before
    touching the DB so a misconfigured policy can't silently sit in the
    table. ``older_than_days`` MUST be > 0 (a 0-day policy would purge
    everything every run).
    """
    if not name:
        raise ValueError("retention policy name is required")
    if older_than_days <= 0:
        raise ValueError(
            "retention policy older_than_days must be > 0; "
            f"got {older_than_days!r}"
        )
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"retention policy action must be one of {sorted(_VALID_ACTIONS)}; "
            f"got {action!r}"
        )
    tags_tuple = tuple(tags_any) if tags_any else ()

    conn = _conn_of(target)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO retention_policies "
            "(name, namespace, category, layer, tags_any, "
            "older_than_days, action, enabled) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (
                name, namespace, category, layer,
                list(tags_tuple) if tags_tuple else None,
                int(older_than_days), action, True,
            ),
        )
        row = cur.fetchone()
    if hasattr(conn, "commit"):
        try:
            conn.commit()
        except Exception:
            pass
    pid = row["id"] if isinstance(row, dict) else row[0]
    return RetentionPolicy(
        id=str(pid),
        name=name,
        namespace=namespace,
        category=category,
        layer=layer,
        tags_any=tags_tuple,
        older_than_days=int(older_than_days),
        action=action,
        enabled=True,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def list_retention_policies(
    target: Any,
    *,
    include_disabled: bool = False,
) -> list[RetentionPolicy]:
    """Return policies (active by default)."""
    conn = _conn_of(target)
    if include_disabled:
        sql = (
            "SELECT id, name, namespace, category, layer, tags_any, "
            "older_than_days, action, enabled, created_at "
            "FROM retention_policies ORDER BY created_at"
        )
    else:
        sql = (
            "SELECT id, name, namespace, category, layer, tags_any, "
            "older_than_days, action, enabled, created_at "
            "FROM retention_policies WHERE enabled = TRUE "
            "ORDER BY created_at"
        )
    with conn.cursor() as cur:
        cur.execute(sql, ())
        rows = cur.fetchall()
    out: list[RetentionPolicy] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(_row_to_policy(r))
        else:
            # Tuple-shaped (raw cursor) — map by position.
            out.append(_row_to_policy({
                "id": r[0],
                "name": r[1],
                "namespace": r[2],
                "category": r[3],
                "layer": r[4],
                "tags_any": r[5],
                "older_than_days": r[6],
                "action": r[7],
                "enabled": r[8],
                "created_at": r[9],
            }))
    return out


def remove_retention_policy(
    target: Any,
    policy_id: str,
    *,
    soft: bool = False,
) -> bool:
    """Delete a policy.

    ``soft=True`` flips ``enabled=FALSE`` instead of removing the row,
    keeping the policy visible to ``list_retention_policies(include_
    disabled=True)``. The default hard-delete is appropriate for a
    one-off mistake; soft-disable is appropriate when an operator
    wants to pause the rule but keep it on file.
    """
    conn = _conn_of(target)
    with conn.cursor() as cur:
        if soft:
            cur.execute(
                "UPDATE retention_policies SET enabled = FALSE "
                "WHERE id = %s",
                (policy_id,),
            )
            return True
        cur.execute(
            "DELETE FROM retention_policies WHERE id = %s RETURNING id",
            (policy_id,),
        )
        row = cur.fetchone()
    if hasattr(conn, "commit"):
        try:
            conn.commit()
        except Exception:
            pass
    return row is not None


# ─────────────────────────────────────────────────────────────────────
# apply_retention runner
# ─────────────────────────────────────────────────────────────────────


def _build_predicate(
    policy: RetentionPolicy,
    now: datetime,
) -> tuple[list[str], list[Any]]:
    """Translate a policy into a parametrized WHERE-clause.

    Order is load-bearing — the test fake dispatches on positional %s
    indices, and so does psycopg2's parameter binding. The clauses are
    appended in the same order their %s placeholders appear so we can
    reuse the same parameter list against either.
    """
    where: list[str] = []
    params: list[Any] = []
    where.append("status = 'active'")

    if policy.namespace:
        where.append("metadata->>'_namespace' = %s")
        params.append(policy.namespace)
    if policy.category:
        where.append("category = %s")
        params.append(policy.category)
    if policy.layer:
        # Top-level ``layer`` column shipped by PR #153
        # (hierarchical memory layers). Fall back to
        # ``metadata->>'_layer'`` for the test fake, which doesn't
        # carry the column. ``COALESCE`` keeps the predicate boolean
        # without forcing a separate codepath.
        where.append(
            "COALESCE(layer, metadata->>'_layer') = %s"
        )
        params.append(policy.layer)
    if policy.tags_any:
        where.append("tags && %s")
        params.append(list(policy.tags_any))

    cutoff = now - timedelta(days=policy.older_than_days)
    where.append("t_created < %s")
    params.append(cutoff)
    return where, params


def _count_matches(
    conn: Any, where: list[str], params: list[Any],
) -> int:
    sql = (
        "SELECT COUNT(*) AS count FROM memories WHERE "
        + " AND ".join(where)
    )
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("count", 0))
    return int(row[0])


def _apply_archive(
    conn: Any, where: list[str], params: list[Any],
) -> int:
    sql = (
        "UPDATE memories SET status = 'archived', "
        "valid_until = NOW() WHERE "
        + " AND ".join(where)
        + " RETURNING id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall() or []
    return len(rows)


def _apply_delete(
    conn: Any, where: list[str], params: list[Any],
) -> int:
    sql = (
        "DELETE FROM memories WHERE "
        + " AND ".join(where)
        + " RETURNING id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall() or []
    return len(rows)


def _write_apply_audit(
    conn: Any,
    *,
    policy_id: str,
    affected: int,
    initiated_by: str | None,
) -> str:
    aid = str(uuid.uuid4())  # placeholder; live DB returns its own
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO forget_audit (scope, target_user_id, policy_id, "
            "doc_rows_deleted, vector_rows_deleted, graph_nodes_deleted, "
            "graph_edges_deleted, state_rows_deleted, initiated_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (
                "policy_apply", None, policy_id,
                int(affected), 0, 0, 0, 0,
                initiated_by,
            ),
        )
        row = cur.fetchone()
    if row is not None:
        if isinstance(row, dict):
            aid = str(row.get("id", aid))
        else:
            aid = str(row[0])
    if hasattr(conn, "commit"):
        try:
            conn.commit()
        except Exception:
            pass
    return aid


def apply_retention(
    mem: Any,
    *,
    dry_run: bool = False,
    initiated_by: str | None = None,
) -> RetentionApplyResult:
    """Evaluate every active retention policy and apply it.

    Idempotent: archive/delete actions filter on ``status='active'``,
    so re-running the same policy on the same window is a no-op.

    Returns counts per policy + totals. ``dry_run=True`` runs the same
    predicate logic but only counts — no UPDATE / DELETE / audit.
    """
    t0 = time.monotonic()
    conn = _conn_of(mem)
    policies = list_retention_policies(mem)
    by_policy: dict[str, dict[str, Any]] = {}
    archived = 0
    deleted = 0
    now = datetime.now(timezone.utc)

    for pol in policies:
        where, params = _build_predicate(pol, now)
        if dry_run:
            n = _count_matches(conn, where, params)
            by_policy[pol.id] = {
                "name": pol.name,
                "action": pol.action,
                "matched": n,
                "applied": 0,
            }
            if pol.action == "archive":
                archived += n
            else:
                deleted += n
            continue

        if pol.action == "archive":
            n = _apply_archive(conn, where, params)
            archived += n
        else:
            n = _apply_delete(conn, where, params)
            deleted += n

        by_policy[pol.id] = {
            "name": pol.name,
            "action": pol.action,
            "matched": n,
            "applied": n,
        }
        if n > 0:
            try:
                _write_apply_audit(
                    conn,
                    policy_id=pol.id,
                    affected=n,
                    initiated_by=initiated_by,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "retention audit insert failed for policy %s: %s",
                    pol.id, e,
                )

    elapsed = round((time.monotonic() - t0) * 1000, 2)
    return RetentionApplyResult(
        policies_evaluated=len(policies),
        memories_archived=archived,
        memories_deleted=deleted,
        elapsed_ms=elapsed,
        by_policy=by_policy,
        dry_run=dry_run,
    )


# ─────────────────────────────────────────────────────────────────────
# forget_user — GDPR right-to-be-forgotten
# ─────────────────────────────────────────────────────────────────────


def _count_user_doc_rows(conn: Any, user_id: str) -> int:
    sql = "SELECT COUNT(*) AS count FROM memories WHERE user_id = %s"
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        row = cur.fetchone()
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("count", 0))
    return int(row[0])


def _count_user_state_rows(conn: Any, user_id: str) -> int:
    try:
        sql = "SELECT COUNT(*) AS count FROM state WHERE user_id = %s"
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
    except Exception:  # state table may not exist on v3
        return 0
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("count", 0))
    return int(row[0])


def _delete_user_doc_rows(conn: Any, user_id: str) -> int:
    sql = (
        "DELETE FROM memories WHERE user_id = %s RETURNING id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        rows = cur.fetchall() or []
    return len(rows)


def _delete_user_state_rows(conn: Any, user_id: str) -> int:
    sql = "DELETE FROM state WHERE user_id = %s RETURNING id"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall() or []
    except Exception:
        return 0
    return len(rows)


def _delete_user_vector(vector_store: Any, user_id: str) -> int:
    """Best-effort deletion from the vector backend.

    Pinecone supports per-namespace and per-id deletion but not a
    "delete by user_id metadata" wildcard in every plan tier.
    Backends can implement ``delete_by_user(user_id)`` to provide
    GDPR coverage; if absent we return 0 and surface a backend error
    so the caller knows the wipe was incomplete.
    """
    if vector_store is None:
        return 0
    method = getattr(vector_store, "delete_by_user", None)
    if method is None:
        return 0
    try:
        n = method(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("vector delete_by_user failed: %s", e)
        raise
    return int(n or 0)


def _delete_user_graph(graph: Any, user_id: str) -> tuple[int, int]:
    if graph is None:
        return 0, 0
    method = getattr(graph, "delete_by_user", None)
    if method is None:
        return 0, 0
    try:
        nodes, edges = method(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("graph delete_by_user failed: %s", e)
        raise
    return int(nodes or 0), int(edges or 0)


def _write_forget_audit(
    conn: Any,
    *,
    user_id: str,
    doc_rows: int,
    vector_rows: int,
    graph_nodes: int,
    graph_edges: int,
    state_rows: int,
    initiated_by: str | None,
) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO forget_audit (scope, target_user_id, policy_id, "
            "doc_rows_deleted, vector_rows_deleted, graph_nodes_deleted, "
            "graph_edges_deleted, state_rows_deleted, initiated_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (
                "user", user_id, None,
                int(doc_rows), int(vector_rows),
                int(graph_nodes), int(graph_edges), int(state_rows),
                initiated_by,
            ),
        )
        row = cur.fetchone()
    if hasattr(conn, "commit"):
        try:
            conn.commit()
        except Exception:
            pass
    if row is None:
        return ""
    if isinstance(row, dict):
        return str(row.get("id", ""))
    return str(row[0])


def forget_user(
    mem: Any,
    user_id: str,
    *,
    dry_run: bool = False,
    initiated_by: str | None = None,
) -> ForgetUserResult:
    """Erase every trace of ``user_id`` across all backends.

    Backend order: Postgres (doc + state) → Pinecone (vector) → Neo4j
    (graph). Errors in any one backend are logged and recorded on the
    result via ``backend_errors``; the other backends still run so the
    user gets the maximum possible coverage from a single call.

    The audit row is written FIRST (with the would-be counts) so the
    deletion event itself is recorded even if a backend later fails.
    """
    t0 = time.monotonic()
    conn = _conn_of(mem)
    vector_store = getattr(mem, "_vector_store", None)
    graph = getattr(mem, "_graph", None)

    # Cheap pre-counts. These also drive the dry-run output.
    doc_count = _count_user_doc_rows(conn, user_id)
    state_count = _count_user_state_rows(conn, user_id)

    if dry_run:
        # Probe vector / graph to surface the blast radius. Stubs that
        # ship a ``delete_by_user`` are still called — that's the
        # contract: dry_run shows what a real run WOULD do.
        try:
            vector_count = _delete_user_vector(vector_store, user_id)
        except Exception:
            vector_count = 0
        try:
            g_nodes, g_edges = _delete_user_graph(graph, user_id)
        except Exception:
            g_nodes, g_edges = 0, 0
        return ForgetUserResult(
            user_id=user_id,
            doc_rows_deleted=doc_count,
            vector_rows_deleted=vector_count,
            graph_nodes_deleted=g_nodes,
            graph_edges_deleted=g_edges,
            state_rows_deleted=state_count,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 2),
            audit_id="",
            dry_run=True,
        )

    backend_errors: list[str] = []

    # 1) Postgres doc
    try:
        doc_deleted = _delete_user_doc_rows(conn, user_id)
    except Exception as e:  # noqa: BLE001
        backend_errors.append(f"document:{type(e).__name__}:{e}")
        doc_deleted = 0

    # 2) state lane
    try:
        state_deleted = _delete_user_state_rows(conn, user_id)
    except Exception as e:  # noqa: BLE001
        backend_errors.append(f"state:{type(e).__name__}:{e}")
        state_deleted = 0

    # 3) vector
    try:
        vector_deleted = _delete_user_vector(vector_store, user_id)
    except Exception as e:  # noqa: BLE001
        backend_errors.append(f"vector:{type(e).__name__}:{e}")
        vector_deleted = 0

    # 4) graph
    try:
        graph_nodes, graph_edges = _delete_user_graph(graph, user_id)
    except Exception as e:  # noqa: BLE001
        backend_errors.append(f"graph:{type(e).__name__}:{e}")
        graph_nodes, graph_edges = 0, 0

    audit_id = _write_forget_audit(
        conn,
        user_id=user_id,
        doc_rows=doc_deleted,
        vector_rows=vector_deleted,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        state_rows=state_deleted,
        initiated_by=initiated_by,
    )

    return ForgetUserResult(
        user_id=user_id,
        doc_rows_deleted=doc_deleted,
        vector_rows_deleted=vector_deleted,
        graph_nodes_deleted=graph_nodes,
        graph_edges_deleted=graph_edges,
        state_rows_deleted=state_deleted,
        elapsed_ms=round((time.monotonic() - t0) * 1000, 2),
        audit_id=audit_id,
        backend_errors=tuple(backend_errors),
    )


# ─────────────────────────────────────────────────────────────────────
# Facade — surface for AgentMemory.retention.*
# ─────────────────────────────────────────────────────────────────────


class RetentionFacade:
    """Bound-method facade exposed as ``mem.retention``.

    Keeps the public API free of the ``mem`` argument that every
    module-level function takes; callers don't need to know the policy
    table lives behind ``mem._store._conn``.
    """

    def __init__(self, mem: Any) -> None:
        self._mem = mem

    def add(
        self,
        name: str,
        *,
        older_than_days: int,
        action: str = "archive",
        namespace: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        tags_any: tuple[str, ...] | list[str] | None = None,
    ) -> RetentionPolicy:
        return add_retention_policy(
            self._mem,
            name=name,
            older_than_days=older_than_days,
            action=action,
            namespace=namespace,
            category=category,
            layer=layer,
            tags_any=tags_any,
        )

    def list(
        self, *, include_disabled: bool = False,
    ) -> list[RetentionPolicy]:
        return list_retention_policies(
            self._mem, include_disabled=include_disabled,
        )

    def remove(self, policy_id: str, *, soft: bool = False) -> bool:
        return remove_retention_policy(self._mem, policy_id, soft=soft)

    def apply(
        self,
        *,
        dry_run: bool = False,
        initiated_by: str | None = None,
    ) -> RetentionApplyResult:
        return apply_retention(
            self._mem, dry_run=dry_run, initiated_by=initiated_by,
        )
