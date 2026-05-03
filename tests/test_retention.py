"""Phase 8.6 — declarative data retention policies + GDPR forget_user.

The retention module ships two surfaces:

  * Policies — declarative CRUD rules ("delete episodic memories older
    than 90 days", "archive category=conversation in namespace=demo
    after 30 days"). Persisted as ``retention_policies`` rows so a
    long-running runner can pick them up.
  * ``forget_user(user_id)`` — GDPR-style physical deletion across the
    document + vector + graph + state lanes. Audit row is written to
    ``forget_audit`` BEFORE any backend delete so the audit survives
    even if a backend partially fails.

Most tests run against a small in-memory ``_FakePostgres`` connection
that interprets just enough SQL — same pattern as
``tests/test_state_lane.py``. Live ``@pytest.mark.live`` tests at the
bottom round-trip against the shared ``attestor_v4_test`` DB so the
SQL is locked down end-to-end.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "attestor" / "store" / "schema.sql"
)


# ─────────────────────────────────────────────────────────────────────
# Lightweight Postgres fake — interprets the small subset of SQL the
# retention module emits. Just enough to drive add/list/remove + apply
# without a live DB.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRow:
    """Mutable row shape — tests poke directly to simulate aged data."""
    id: str
    user_id: str
    content: str
    category: str
    namespace: str
    tags: tuple[str, ...]
    status: str
    valid_until: datetime | None
    t_created: datetime
    metadata: dict[str, Any]


class _FakeCursor:
    """SQL-shaped cursor that the retention module's queries can drive."""

    def __init__(self, db: "_FakePostgres") -> None:
        self._db = db
        self._rows: list[Any] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._rows = self._db.execute(sql, params or ())

    def executemany(self, sql: str, seq: list[Any]) -> None:
        for p in seq:
            self.execute(sql, p)

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    @property
    def description(self) -> Any:
        if not self._rows:
            return None
        first = self._rows[0]
        if isinstance(first, dict):
            return [(k,) for k in first.keys()]
        return None


class _FakePostgres:
    """Tiny SQL interpreter — handles only the queries retention.py emits."""

    def __init__(self) -> None:
        self.policies: list[dict[str, Any]] = []
        self.memories: list[_FakeRow] = []
        self.forget_audit: list[dict[str, Any]] = []
        self.state: list[dict[str, Any]] = []
        self.committed = False

    def cursor(self, cursor_factory: object | None = None) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        return None

    @property
    def autocommit(self) -> bool:
        return True

    @autocommit.setter
    def autocommit(self, _value: bool) -> None:  # tolerate setter
        return None

    # ── SQL dispatcher ──

    def execute(self, sql: str, params: tuple) -> list[dict[str, Any]]:
        s = sql.strip()
        low = s.lower()

        if low.startswith("insert into retention_policies"):
            row = self._insert_policy(params)
            return [{"id": row["id"]}]

        if low.startswith("select") and "from retention_policies" in low:
            return self._select_policies(s, params)

        if low.startswith("delete from retention_policies"):
            pid = params[0]
            before = len(self.policies)
            self.policies = [p for p in self.policies if p["id"] != pid]
            return [{"id": pid}] if len(self.policies) < before else []

        if low.startswith("update retention_policies"):
            pid = params[-1]
            for p in self.policies:
                if p["id"] == pid:
                    p["enabled"] = False
            return []

        if low.startswith("update memories"):
            return self._update_memories(s, params)

        if low.startswith("delete from memories"):
            return self._delete_memories(s, params)

        if low.startswith("select") and "from memories" in low:
            return self._select_memories(s, params)

        if low.startswith("insert into forget_audit"):
            row = self._insert_audit(params)
            return [{"id": row["id"]}]

        if low.startswith("select") and "from forget_audit" in low:
            return [dict(r) for r in self.forget_audit]

        if low.startswith("delete from state"):
            uid = params[0]
            before = len(self.state)
            self.state = [r for r in self.state if r["user_id"] != uid]
            return [{"id": i} for i in range(before - len(self.state))]

        if low.startswith("select") and "from state" in low:
            if "count(*)" in low:
                uid = params[0] if params else None
                n = sum(1 for r in self.state if r["user_id"] == uid)
                return [{"count": n}]
            return [dict(r) for r in self.state]

        # Default: no-op (helpers like SET / SELECT 1 fall through here).
        return []

    # ── Policy CRUD helpers ──

    def _insert_policy(self, params: tuple) -> dict[str, Any]:
        pid = str(uuid.uuid4())
        (
            name, namespace, category, layer, tags_any,
            older_than_days, action, enabled,
        ) = params
        row = {
            "id": pid,
            "name": name,
            "namespace": namespace,
            "category": category,
            "layer": layer,
            "tags_any": list(tags_any) if tags_any else [],
            "older_than_days": int(older_than_days),
            "action": action,
            "enabled": bool(enabled),
            "created_at": datetime.now(timezone.utc),
        }
        self.policies.append(row)
        return row

    def _select_policies(
        self, sql: str, params: tuple,
    ) -> list[dict[str, Any]]:
        # Two shapes: list (with optional `WHERE enabled = TRUE`) and
        # by-id (`WHERE id = %s`).
        s_low = sql.lower()
        out = list(self.policies)
        if "enabled = true" in s_low:
            out = [p for p in out if p["enabled"]]
        if "where id = " in s_low and params:
            out = [p for p in out if p["id"] == params[0]]
        return [dict(p) for p in out]

    # ── Memory query helpers ──

    def _matches_filter(
        self,
        row: _FakeRow,
        sql: str,
        params: tuple,
    ) -> bool:
        # The retention runner's WHERE clauses are simple ANDs that we
        # interpret token-by-token.
        s_low = sql.lower()

        if "user_id = %s" in s_low:
            # Order matters; the runner places user_id first.
            if row.user_id != params[0]:
                return False

        ns_match = re.search(r"_namespace[^=]*=\s*%s", s_low)
        if ns_match and "namespace" in s_low:
            ns_idx = self._param_index(sql, "_namespace")
            if ns_idx is not None and row.namespace != params[ns_idx]:
                return False

        cat_match = re.search(r"category\s*=\s*%s", s_low)
        if cat_match:
            cat_idx = self._param_index(sql, "category =")
            if cat_idx is not None and row.category != params[cat_idx]:
                return False

        layer_match = re.search(r"_layer[^=]*=\s*%s", s_low)
        if layer_match:
            l_idx = self._param_index(sql, "_layer")
            if l_idx is not None:
                if row.metadata.get("_layer") != params[l_idx]:
                    return False

        tag_match = re.search(r"tags\s*&&\s*%s", s_low)
        if tag_match:
            t_idx = self._param_index(sql, "tags &&")
            if t_idx is not None:
                want = set(params[t_idx])
                if not (set(row.tags) & want):
                    return False

        # Older-than predicate: ``t_created < %s``
        old_match = re.search(r"t_created\s*<\s*%s", s_low)
        if old_match:
            o_idx = self._param_index(sql, "t_created <")
            if o_idx is not None and row.t_created >= params[o_idx]:
                return False

        # Active-only filter is implicit on archive paths.
        if "status = 'active'" in s_low and row.status != "active":
            return False

        return True

    def _param_index(self, sql: str, marker: str) -> int | None:
        """Return the positional index of the %s following ``marker``."""
        marker_low = marker.lower()
        s_low = sql.lower()
        idx = s_low.find(marker_low)
        if idx < 0:
            return None
        # Count %s placeholders that appear in the query up to and
        # including the one immediately after the marker.
        before = s_low[:idx].count("%s")
        return before

    def _update_memories(
        self, sql: str, params: tuple,
    ) -> list[dict[str, Any]]:
        affected: list[dict[str, Any]] = []
        for row in self.memories:
            if not self._matches_filter(row, sql, params):
                continue
            if "set status = 'archived'" in sql.lower():
                row.status = "archived"
                row.valid_until = datetime.now(timezone.utc)
                affected.append({"id": row.id})
        return affected

    def _delete_memories(
        self, sql: str, params: tuple,
    ) -> list[dict[str, Any]]:
        keep: list[_FakeRow] = []
        deleted: list[dict[str, Any]] = []
        for row in self.memories:
            if self._matches_filter(row, sql, params):
                deleted.append({"id": row.id})
            else:
                keep.append(row)
        self.memories = keep
        return deleted

    def _select_memories(
        self, sql: str, params: tuple,
    ) -> list[dict[str, Any]]:
        # Minimal — only used for COUNT(*) probes (e.g. dry-run).
        s_low = sql.lower()
        if "count(*)" in s_low:
            n = sum(1 for r in self.memories
                    if self._matches_filter(r, sql, params))
            return [{"count": n}]
        return [
            {"id": r.id} for r in self.memories
            if self._matches_filter(r, sql, params)
        ]

    def _insert_audit(self, params: tuple) -> dict[str, Any]:
        aid = str(uuid.uuid4())
        (
            scope, target_user_id, policy_id,
            doc_rows, vector_rows, graph_nodes, graph_edges, state_rows,
            initiated_by,
        ) = params
        row = {
            "id": aid,
            "scope": scope,
            "target_user_id": target_user_id,
            "policy_id": policy_id,
            "doc_rows_deleted": int(doc_rows),
            "vector_rows_deleted": int(vector_rows),
            "graph_nodes_deleted": int(graph_nodes),
            "graph_edges_deleted": int(graph_edges),
            "state_rows_deleted": int(state_rows),
            "initiated_by": initiated_by,
            "initiated_at": datetime.now(timezone.utc),
        }
        self.forget_audit.append(row)
        return row


# ─────────────────────────────────────────────────────────────────────
# Vector + graph + state stubs for forget_user
# ─────────────────────────────────────────────────────────────────────


class _VectorStub:
    def __init__(self) -> None:
        self.deleted_user_ids: list[str] = []
        self.calls: list[str] = []

    def delete_by_user(self, user_id: str) -> int:
        self.deleted_user_ids.append(user_id)
        return 7  # arbitrary deterministic count


class _GraphStub:
    def __init__(self) -> None:
        self.deleted_user_ids: list[str] = []

    def delete_by_user(self, user_id: str) -> tuple[int, int]:
        self.deleted_user_ids.append(user_id)
        return (3, 5)  # nodes, edges


# ─────────────────────────────────────────────────────────────────────
# Memory factory
# ─────────────────────────────────────────────────────────────────────


def _populate(
    db: _FakePostgres,
    *,
    user_id: str | None = None,
    n: int = 1,
    namespace: str = "default",
    category: str = "general",
    layer: str | None = None,
    tags: tuple[str, ...] = (),
    age_days: int = 0,
    status: str = "active",
) -> list[_FakeRow]:
    rows: list[_FakeRow] = []
    for i in range(n):
        row = _FakeRow(
            id=str(uuid.uuid4()),
            user_id=user_id or str(uuid.uuid4()),
            content=f"mem-{i}",
            category=category,
            namespace=namespace,
            tags=tags,
            status=status,
            valid_until=None,
            t_created=datetime.now(timezone.utc) - timedelta(days=age_days),
            metadata={"_layer": layer} if layer else {},
        )
        db.memories.append(row)
        rows.append(row)
    return rows


@pytest.fixture
def db() -> _FakePostgres:
    return _FakePostgres()


# ─────────────────────────────────────────────────────────────────────
# Policy CRUD
# ─────────────────────────────────────────────────────────────────────


def test_add_policy_persists(db: _FakePostgres) -> None:
    from attestor.compliance.retention import add_retention_policy

    p = add_retention_policy(
        db,
        name="archive-90d",
        older_than_days=90,
        action="archive",
    )
    assert p.id
    assert p.name == "archive-90d"
    assert p.older_than_days == 90
    assert p.action == "archive"
    assert p.enabled is True
    assert len(db.policies) == 1


def test_list_policies_filters_disabled(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        list_retention_policies,
        remove_retention_policy,
    )

    p1 = add_retention_policy(
        db, name="a", older_than_days=30, action="archive",
    )
    p2 = add_retention_policy(
        db, name="b", older_than_days=60, action="delete",
    )
    # Soft-disable p2
    remove_retention_policy(db, p2.id, soft=True)

    visible = list_retention_policies(db)
    assert len(visible) == 1
    assert visible[0].id == p1.id

    # Hard list still sees both
    all_pol = list_retention_policies(db, include_disabled=True)
    assert len(all_pol) == 2


def test_remove_policy_deletes_row(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        list_retention_policies,
        remove_retention_policy,
    )

    p = add_retention_policy(
        db, name="x", older_than_days=10, action="archive",
    )
    assert remove_retention_policy(db, p.id) is True
    assert list_retention_policies(db, include_disabled=True) == []


# ─────────────────────────────────────────────────────────────────────
# Apply runner — filters
# ─────────────────────────────────────────────────────────────────────


class _MemStub:
    """Stub for the AgentMemory surface ``apply_retention`` reaches into."""

    def __init__(
        self,
        db: _FakePostgres,
        vector: _VectorStub | None = None,
        graph: _GraphStub | None = None,
    ) -> None:
        self._store = _StoreStub(db)
        self._vector_store = vector
        self._graph = graph
        self.state = _StateStub(db) if db is not None else None


class _StoreStub:
    def __init__(self, db: _FakePostgres) -> None:
        self._conn = db
        self._v4 = True


class _StateStub:
    def __init__(self, db: _FakePostgres) -> None:
        self._db = db


def test_apply_dry_run_no_writes(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="archive-old",
        older_than_days=30, action="archive",
    )
    _populate(db, n=3, age_days=60)

    mem = _MemStub(db)
    result = apply_retention(mem, dry_run=True)

    assert result.dry_run is True
    assert result.memories_archived == 3
    assert result.memories_deleted == 0
    # No actual mutation
    assert all(r.status == "active" for r in db.memories)


def test_apply_archive_action(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="archive-30d",
        older_than_days=30, action="archive",
    )
    old = _populate(db, n=2, age_days=45)
    fresh = _populate(db, n=1, age_days=1)

    mem = _MemStub(db)
    result = apply_retention(mem)

    assert result.memories_archived == 2
    assert result.memories_deleted == 0
    for row in old:
        assert row.status == "archived"
        assert row.valid_until is not None
    for row in fresh:
        assert row.status == "active"


def test_apply_delete_action(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="delete-180d",
        older_than_days=180, action="delete",
    )
    _populate(db, n=2, age_days=200)
    keep = _populate(db, n=1, age_days=10)

    mem = _MemStub(db)
    result = apply_retention(mem)

    assert result.memories_deleted == 2
    assert len(db.memories) == 1
    assert db.memories[0].id == keep[0].id


def test_apply_namespace_filter(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="ns-only",
        namespace="ephemeral",
        older_than_days=7, action="delete",
    )
    _populate(db, n=2, age_days=30, namespace="ephemeral")
    other = _populate(db, n=3, age_days=30, namespace="durable")

    mem = _MemStub(db)
    apply_retention(mem)

    remaining = {r.namespace for r in db.memories}
    assert remaining == {"durable"}
    assert len(db.memories) == 3
    other_ids = {r.id for r in other}
    assert {r.id for r in db.memories} == other_ids


def test_apply_category_filter(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="conv-30",
        category="conversation",
        older_than_days=30, action="delete",
    )
    _populate(db, n=2, age_days=60, category="conversation")
    _populate(db, n=2, age_days=60, category="preference")

    mem = _MemStub(db)
    result = apply_retention(mem)

    assert result.memories_deleted == 2
    cats = {r.category for r in db.memories}
    assert cats == {"preference"}


def test_apply_layer_filter(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="ep-90",
        layer="episodic",
        older_than_days=90, action="archive",
    )
    _populate(db, n=2, age_days=120, layer="episodic")
    _populate(db, n=2, age_days=120, layer="semantic")
    _populate(db, n=1, age_days=120)  # no layer

    mem = _MemStub(db)
    apply_retention(mem)

    archived = [r for r in db.memories if r.status == "archived"]
    assert len(archived) == 2
    assert all(r.metadata.get("_layer") == "episodic" for r in archived)


def test_apply_tag_match_any(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="tag-pii",
        tags_any=("pii", "secret"),
        older_than_days=14, action="delete",
    )
    _populate(db, n=1, age_days=30, tags=("pii",))
    _populate(db, n=1, age_days=30, tags=("secret", "other"))
    _populate(db, n=1, age_days=30, tags=("safe",))

    mem = _MemStub(db)
    result = apply_retention(mem)

    assert result.memories_deleted == 2
    assert len(db.memories) == 1
    assert db.memories[0].tags == ("safe",)


def test_apply_recent_memories_unaffected(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="age-7",
        older_than_days=7, action="archive",
    )
    _populate(db, n=3, age_days=2)

    mem = _MemStub(db)
    result = apply_retention(mem)
    assert result.memories_archived == 0
    assert all(r.status == "active" for r in db.memories)


def test_apply_writes_audit_row(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    pol = add_retention_policy(
        db, name="audit-test",
        older_than_days=10, action="archive",
    )
    _populate(db, n=2, age_days=30)

    mem = _MemStub(db)
    apply_retention(mem, initiated_by="agent-7")

    assert len(db.forget_audit) == 1
    audit = db.forget_audit[0]
    assert audit["scope"] == "policy_apply"
    assert audit["policy_id"] == pol.id
    assert audit["initiated_by"] == "agent-7"
    assert audit["doc_rows_deleted"] == 2  # archived counted as doc-affected


def test_apply_idempotent(db: _FakePostgres) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    add_retention_policy(
        db, name="idem",
        older_than_days=10, action="archive",
    )
    _populate(db, n=2, age_days=30)
    mem = _MemStub(db)

    first = apply_retention(mem)
    second = apply_retention(mem)
    assert first.memories_archived == 2
    assert second.memories_archived == 0  # already archived → skipped


# ─────────────────────────────────────────────────────────────────────
# forget_user
# ─────────────────────────────────────────────────────────────────────


def test_forget_user_deletes_doc_rows(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    other = str(uuid.uuid4())
    _populate(db, n=4, user_id=target)
    _populate(db, n=2, user_id=other)

    mem = _MemStub(db)
    result = forget_user(mem, target)

    assert result.doc_rows_deleted == 4
    remaining_uids = {r.user_id for r in db.memories}
    assert remaining_uids == {other}


def test_forget_user_deletes_vector_rows(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    _populate(db, n=2, user_id=target)
    vec = _VectorStub()
    mem = _MemStub(db, vector=vec)

    result = forget_user(mem, target)
    assert result.vector_rows_deleted == 7
    assert vec.deleted_user_ids == [target]


def test_forget_user_clears_state_lane(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    other = str(uuid.uuid4())
    db.state.append({"user_id": target, "key": "k"})
    db.state.append({"user_id": target, "key": "k2"})
    db.state.append({"user_id": other, "key": "k"})
    _populate(db, n=1, user_id=target)

    mem = _MemStub(db)
    result = forget_user(mem, target)
    assert result.state_rows_deleted == 2
    assert all(r["user_id"] == other for r in db.state)


def test_forget_user_audit_recorded(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    _populate(db, n=1, user_id=target)
    mem = _MemStub(db)
    result = forget_user(mem, target, initiated_by="agent-9")

    assert result.audit_id
    assert len(db.forget_audit) == 1
    audit = db.forget_audit[0]
    assert audit["scope"] == "user"
    assert audit["target_user_id"] == target
    assert audit["initiated_by"] == "agent-9"


def test_forget_user_dry_run(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    _populate(db, n=3, user_id=target)
    db.state.append({"user_id": target, "key": "k"})
    mem = _MemStub(db, vector=_VectorStub(), graph=_GraphStub())

    result = forget_user(mem, target, dry_run=True)
    # Counts reflect what WOULD happen
    assert result.doc_rows_deleted == 3
    assert result.state_rows_deleted == 1
    # …but state is untouched
    assert len(db.memories) == 3
    assert len(db.state) == 1
    # Audit IS NOT written in dry-run
    assert db.forget_audit == []


def test_forget_user_other_users_untouched(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    other = str(uuid.uuid4())
    _populate(db, n=2, user_id=target)
    other_rows = _populate(db, n=3, user_id=other)
    db.state.append({"user_id": other, "key": "k"})

    mem = _MemStub(db)
    forget_user(mem, target)

    assert {r.user_id for r in db.memories} == {other}
    assert len(db.memories) == 3
    assert {r.id for r in db.memories} == {r.id for r in other_rows}
    assert len(db.state) == 1


def test_forget_user_calls_graph_when_present(db: _FakePostgres) -> None:
    from attestor.compliance.retention import forget_user

    target = str(uuid.uuid4())
    _populate(db, n=1, user_id=target)
    graph = _GraphStub()
    mem = _MemStub(db, graph=graph)
    result = forget_user(mem, target)

    assert graph.deleted_user_ids == [target]
    assert result.graph_nodes_deleted == 3
    assert result.graph_edges_deleted == 5


def test_v4_uuid_user_id_handling(db: _FakePostgres) -> None:
    """User id MUST round-trip as a string-stable UUID across the audit."""
    from attestor.compliance.retention import forget_user

    raw = uuid.uuid4()
    target = str(raw)
    _populate(db, n=1, user_id=target)

    mem = _MemStub(db)
    result = forget_user(mem, target)

    assert result.user_id == target
    audit = db.forget_audit[0]
    assert audit["target_user_id"] == target
    # Round-trip through string keeps UUID validation cheap.
    assert uuid.UUID(audit["target_user_id"]) == raw


# ─────────────────────────────────────────────────────────────────────
# AgentMemory wiring (smoke — ensures the public surface lands)
# ─────────────────────────────────────────────────────────────────────


def test_agent_memory_exposes_retention_namespace() -> None:
    from attestor.compliance.retention import RetentionFacade

    facade = RetentionFacade(_MemStub(_FakePostgres()))
    # The facade exposes add / list / apply on the live stub.
    assert hasattr(facade, "add")
    assert hasattr(facade, "list")
    assert hasattr(facade, "apply")


# ─────────────────────────────────────────────────────────────────────
# Live integration — schema + end-to-end round trip
# ─────────────────────────────────────────────────────────────────────


def _live_reachable() -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def admin_conn():
    if not _live_reachable():
        pytest.skip("local Postgres unreachable")
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    from attestor.store.embeddings import clear_embedding_cache
    clear_embedding_cache()
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "1024")
    with admin_conn.cursor() as cur:
        for tbl in (
            "forget_audit", "retention_policies", "deletion_audit",
            "user_quotas", "state", "memories", "episodes",
            "sessions", "projects", "users",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.mark.live
def test_retention_policies_table_exists(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'retention_policies'"
        )
        assert cur.fetchone() is not None


@pytest.mark.live
def test_forget_audit_table_exists(admin_conn, fresh_schema) -> None:
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'forget_audit'"
        )
        assert cur.fetchone() is not None


@pytest.mark.live
def test_live_apply_archive_round_trip(admin_conn, fresh_schema) -> None:
    from attestor.compliance.retention import (
        add_retention_policy,
        apply_retention,
    )

    uid = uuid.uuid4()
    ext = f"u-ret-{uuid.uuid4().hex[:8]}"
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT set_config('attestor.current_user_id', %s, false)",
            (str(uid),),
        )
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), ext),
        )
        # Insert two old memories
        for _ in range(2):
            cur.execute(
                "INSERT INTO memories (user_id, content, category, "
                "valid_from, t_created) "
                "VALUES (%s, %s, %s, NOW() - INTERVAL '120 days', "
                "NOW() - INTERVAL '120 days')",
                (str(uid), "old", "general"),
            )

    pol = add_retention_policy(
        admin_conn,
        name="archive-90",
        older_than_days=90, action="archive",
    )
    assert pol.id

    class _LiveStore:
        def __init__(self, conn):
            self._conn = conn
            self._v4 = True

    class _LiveMem:
        def __init__(self, conn):
            self._store = _LiveStore(conn)
            self._vector_store = None
            self._graph = None
            self.state = None

    result = apply_retention(_LiveMem(admin_conn))
    assert result.memories_archived >= 2

    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM memories WHERE user_id = %s::uuid",
            (str(uid),),
        )
        statuses = {r[0] for r in cur.fetchall()}
    assert statuses == {"archived"}
