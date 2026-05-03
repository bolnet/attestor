"""StateRepo — typed profile lane on top of the v4 ``state`` table.

The state lane stores small, type-checked profile facts (preferences,
capability declarations, durable identity facts) that live next to the
retrieval pipeline but bypass it entirely. OpenAI's January 2026
``context_personalization`` cookbook calls out that retrieval is the
wrong tool for personalization — durable, type-checked profile facts
should live in a state object. This repo is that object.

Bi-temporal:
    * ``t_created`` is the transaction time start (when the row landed).
    * ``t_expired`` is set on supersession or delete so ``history()`` and
      ``as_of()`` get the full audit trail.

Supersession is enforced at the application layer, not via a unique
index, so the database stays append-only. Concurrent ``set`` writes for
the same key are serialized via a SELECT ... FOR UPDATE on the active row.

Two backends:
    * Postgres (default) — pass a psycopg2 connection.
    * In-memory — pass an ``InMemoryStateBackend`` instance. Used by the
      unit tests so they don't require a live database.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol

from attestor.state.models import StateRecord, StateValidationError
from attestor.state.schemas import validate as _schema_validate

logger = logging.getLogger("attestor.state")


# ---------------------------------------------------------------------------
# Backend protocol + in-memory implementation
# ---------------------------------------------------------------------------


class StateBackend(Protocol):
    """Storage protocol for the state lane.

    The repo handles JSON-Schema validation and key/value plumbing; the
    backend handles row storage + bi-temporal ordering. Two implementations
    ship: ``InMemoryStateBackend`` (tests) and ``PostgresStateBackend``
    (production).
    """

    def insert(self, record: StateRecord) -> StateRecord: ...

    def expire_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        when: datetime,
    ) -> int: ...

    def get_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> StateRecord | None: ...

    def list_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        prefix: str,
    ) -> list[StateRecord]: ...

    def history(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> list[StateRecord]: ...

    def as_of(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        ts: datetime,
    ) -> StateRecord | None: ...


# ---------------------------------------------------------------------------
# In-memory backend (test double)
# ---------------------------------------------------------------------------


@dataclass
class InMemoryStateBackend:
    """Thread-safe in-memory backend for unit tests.

    Stores records in a flat list and reproduces the SQL semantics by
    hand: SELECT active filters ``t_expired IS NULL``, history orders by
    ``set_at``, ``as_of`` returns the row that was active at ``ts``.

    Not intended for production — there's no persistence and no RLS.
    """

    rows: list[StateRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def insert(self, record: StateRecord) -> StateRecord:
        with self._lock:
            self.rows.append(record)
        return record

    def expire_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        when: datetime,
    ) -> int:
        n = 0
        with self._lock:
            for idx, row in enumerate(self.rows):
                if (
                    row.t_expired is None
                    and row.user_id == user_id
                    and row.project_id == project_id
                    and row.scope == scope
                    and row.key == key
                ):
                    self.rows[idx] = replace(row, t_expired=when)
                    n += 1
        return n

    def get_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> StateRecord | None:
        with self._lock:
            for row in self.rows:
                if (
                    row.t_expired is None
                    and row.user_id == user_id
                    and row.project_id == project_id
                    and row.scope == scope
                    and row.key == key
                ):
                    return row
        return None

    def list_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        prefix: str,
    ) -> list[StateRecord]:
        with self._lock:
            return [
                row for row in self.rows
                if row.t_expired is None
                and row.user_id == user_id
                and row.project_id == project_id
                and row.scope == scope
                and row.key.startswith(prefix)
            ]

    def history(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> list[StateRecord]:
        with self._lock:
            matches = [
                row for row in self.rows
                if row.user_id == user_id
                and row.project_id == project_id
                and row.scope == scope
                and row.key == key
            ]
        # Stable chronological order: rows are appended in set order, so
        # sort by t_created when present, otherwise insertion order.
        matches.sort(
            key=lambda r: (r.t_created or datetime.min.replace(tzinfo=timezone.utc))
        )
        return matches

    def as_of(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        ts: datetime,
    ) -> StateRecord | None:
        # Pick the row whose validity window [t_created, t_expired or +inf)
        # contains ``ts``. Prefer the latest qualifying row.
        candidates: list[StateRecord] = []
        for row in self.history(
            user_id=user_id, project_id=project_id, scope=scope, key=key,
        ):
            t0 = row.t_created
            t1 = row.t_expired
            if t0 is None:
                continue
            if t0 <= ts and (t1 is None or ts < t1):
                candidates.append(row)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: r.t_created or datetime.min.replace(tzinfo=timezone.utc),
        )


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------


class PostgresStateBackend:
    """Postgres-backed implementation. Uses the v4 ``state`` table.

    The connection is shared with the document store so the RLS variable
    set on memory writes also gates state reads/writes. The repo doesn't
    open transactions itself — it commits at the end of every public
    method (matches ``UserRepo`` / ``ProjectRepo`` style).
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> StateRecord:
        return StateRecord(
            key=row["key"],
            value=row["value"],
            user_id=str(row["user_id"]),
            project_id=str(row["project_id"]) if row.get("project_id") else None,
            agent_id=row.get("agent_id"),
            scope=row.get("scope") or "user",
            schema_id=row.get("schema_id"),
            set_at=row.get("set_at"),
            set_by_agent=row.get("set_by_agent"),
            t_created=row.get("t_created"),
            t_expired=row.get("t_expired"),
            signature=row.get("signature"),
        )

    def insert(self, record: StateRecord) -> StateRecord:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO state (
                    id, user_id, project_id, agent_id, scope, key, value,
                    schema_id, set_at, set_by_agent, t_created, t_expired,
                    signature
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s::jsonb,
                    %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    str(uuid.uuid4()),
                    record.user_id,
                    record.project_id,
                    record.agent_id,
                    record.scope,
                    record.key,
                    json.dumps(record.value),
                    record.schema_id,
                    record.set_at,
                    record.set_by_agent,
                    record.t_created,
                    record.t_expired,
                    record.signature,
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return self._row_to_record(dict(row))

    def expire_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        when: datetime,
    ) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE state
                SET t_expired = %s
                WHERE user_id = %s
                  AND COALESCE(project_id, '00000000-0000-0000-0000-000000000000'::uuid)
                      = COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND scope = %s
                  AND key = %s
                  AND t_expired IS NULL
                """,
                (when, user_id, project_id, scope, key),
            )
            n = cur.rowcount
        self._conn.commit()
        return n

    def get_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> StateRecord | None:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM state
                WHERE user_id = %s
                  AND COALESCE(project_id, '00000000-0000-0000-0000-000000000000'::uuid)
                      = COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND scope = %s
                  AND key = %s
                  AND t_expired IS NULL
                LIMIT 1
                """,
                (user_id, project_id, scope, key),
            )
            row = cur.fetchone()
        return self._row_to_record(dict(row)) if row else None

    def list_active(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        prefix: str,
    ) -> list[StateRecord]:
        import psycopg2.extras

        # Escape LIKE metacharacters in the user-supplied prefix —
        # otherwise ``prefix="prefs.%"`` lets a caller see keys outside
        # their declared subtree (cross-tenant key enumeration). The
        # ``ESCAPE '\'`` clause activates the literal-escape semantics.
        escaped = (
            prefix.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM state
                WHERE user_id = %s
                  AND COALESCE(project_id, '00000000-0000-0000-0000-000000000000'::uuid)
                      = COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND scope = %s
                  AND key LIKE %s ESCAPE '\\'
                  AND t_expired IS NULL
                ORDER BY key ASC
                """,
                (user_id, project_id, scope, escaped + "%"),
            )
            rows = cur.fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def history(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
    ) -> list[StateRecord]:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM state
                WHERE user_id = %s
                  AND COALESCE(project_id, '00000000-0000-0000-0000-000000000000'::uuid)
                      = COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND scope = %s
                  AND key = %s
                ORDER BY t_created ASC
                """,
                (user_id, project_id, scope, key),
            )
            rows = cur.fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def as_of(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: str,
        key: str,
        ts: datetime,
    ) -> StateRecord | None:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM state
                WHERE user_id = %s
                  AND COALESCE(project_id, '00000000-0000-0000-0000-000000000000'::uuid)
                      = COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND scope = %s
                  AND key = %s
                  AND t_created <= %s
                  AND (t_expired IS NULL OR t_expired > %s)
                ORDER BY t_created DESC
                LIMIT 1
                """,
                (user_id, project_id, scope, key, ts, ts),
            )
            row = cur.fetchone()
        return self._row_to_record(dict(row)) if row else None


# ---------------------------------------------------------------------------
# StateRepo — public surface
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """UTC-aware now. Avoids the deprecated ``datetime.utcnow``."""
    return datetime.now(timezone.utc)


def _resolve_scope(
    scope: str | None,
    project_id: str | None,
    agent_id: str | None,
) -> str:
    """Pick the default scope from the args. Caller can override with
    ``scope=`` explicitly. Order: explicit > agent > project > user."""
    if scope:
        return scope
    if agent_id:
        return "agent"
    if project_id:
        return "project"
    return "user"


class StateRepo:
    """Public API for the state lane.

    Two construction paths:

    * ``StateRepo(conn=psycopg2_connection)`` — production. Uses the
      Postgres ``state`` table.
    * ``StateRepo(backend=InMemoryStateBackend())`` — tests. No DB needed.
    """

    def __init__(
        self,
        conn: Any | None = None,
        *,
        backend: StateBackend | None = None,
        signer: Any | None = None,
    ) -> None:
        if backend is None and conn is None:
            raise ValueError("StateRepo requires either conn=... or backend=...")
        if backend is None:
            backend = PostgresStateBackend(conn)
        self._backend = backend
        self._signer = signer

    # ------------------------------------------------------------------ #
    #  Write                                                              #
    # ------------------------------------------------------------------ #

    def set(
        self,
        key: str,
        value: Any,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
        schema: str | None = None,
        set_by_agent: str | None = None,
    ) -> StateRecord:
        """Write a typed value at ``key`` for the given owner.

        If a previous active row exists for the same (user, project, scope,
        key) tuple, its ``t_expired`` is stamped first. Then the new row
        is inserted. The two operations share the same backend connection
        so a partial failure rolls back via the backend's transaction.

        Args:
            key: Dotted key, e.g. ``"preferences.theme"``.
            value: JSON-serializable value. Validated against
                ``schema`` when provided.
            user_id: Owner UUID. Required.
            project_id: Optional project anchor; isolates by project.
            agent_id: Optional agent anchor.
            scope: Override the auto-derived scope.
            schema: Name of the JSON-Schema bundled in
                ``attestor/state/schemas/`` (without the ``.json`` suffix).
                When set, the value is validated before insert. Validation
                failures raise ``StateValidationError``.
            set_by_agent: Audit-only: who actually called ``set`` (may
                differ from ``agent_id`` when an orchestrator writes on
                behalf of a sub-agent).

        Returns:
            The newly-inserted ``StateRecord`` (frozen).

        Raises:
            StateValidationError: If ``schema`` is set and validation fails.
        """
        if not key:
            raise StateValidationError("key must be non-empty")
        if not user_id:
            raise StateValidationError("user_id is required")

        if schema is not None:
            _schema_validate(value, schema)

        eff_scope = _resolve_scope(scope, project_id, agent_id)
        now = _utcnow()
        # Expire any currently-active row before the new insert. We commit
        # the expire so the new insert doesn't block on a self-lock; the
        # window between the two writes is tiny, and the listing/query
        # paths tolerate transient overlap (they all filter by
        # t_expired IS NULL and pick at most one row).
        self._backend.expire_active(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
            when=now,
        )

        record = StateRecord(
            key=key,
            value=copy.deepcopy(value),
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            scope=eff_scope,
            schema_id=schema,
            set_at=now,
            set_by_agent=set_by_agent or agent_id,
            t_created=now,
            t_expired=None,
        )
        # Optional Ed25519 signature so an audit trail can prove the row
        # wasn't tampered with after-the-fact. Mirrors the memory-write
        # signing path in AgentMemory.add().
        if self._signer is not None:
            try:
                from attestor.identity.signing import canonical_payload, sign_payload

                payload = canonical_payload(
                    memory_id=key,
                    agent_id=agent_id,
                    t_created=now,
                    content_hash=None,
                )
                sig = sign_payload(
                    payload,
                    secret_key_bytes=self._signer.secret_key_bytes,
                )
                record = replace(record, signature=sig)
            except Exception as e:  # pragma: no cover - signing is opt-in
                logger.warning("state row signing failed for key=%s: %s", key, e)

        return self._backend.insert(record)

    # ------------------------------------------------------------------ #
    #  Read                                                               #
    # ------------------------------------------------------------------ #

    def get(
        self,
        key: str,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> Any | None:
        """Return the active value at ``key`` or ``None`` if missing."""
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        rec = self._backend.get_active(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
        )
        return rec.value if rec else None

    def get_record(
        self,
        key: str,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> StateRecord | None:
        """Return the active ``StateRecord`` at ``key`` or ``None``."""
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        return self._backend.get_active(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
        )

    def list(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Return all active key/value pairs whose key starts with
        ``prefix``. Empty prefix returns every active row in the scope.
        """
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        rows = self._backend.list_active(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            prefix=prefix,
        )
        return {r.key: r.value for r in rows}

    def history(
        self,
        key: str,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> list[StateRecord]:
        """Every value this key has held, oldest first. Bi-temporal."""
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        return self._backend.history(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
        )

    def as_of(
        self,
        key: str,
        *,
        ts: datetime,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> Any | None:
        """Replay state: return the value that was active at ``ts``.

        Returns ``None`` when no row was active at that point. Equivalent
        to ``recall(as_of=...)`` on the retrieval lane, but for typed
        profile facts.
        """
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        rec = self._backend.as_of(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
            ts=ts,
        )
        return rec.value if rec else None

    # ------------------------------------------------------------------ #
    #  Delete                                                             #
    # ------------------------------------------------------------------ #

    def delete(
        self,
        key: str,
        *,
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> bool:
        """Mark the active row for ``key`` as expired.

        Returns:
            True if a row was expired, False if nothing was active.

        The history is preserved — use ``history()`` to read it.
        """
        eff_scope = _resolve_scope(scope, project_id, agent_id)
        n = self._backend.expire_active(
            user_id=user_id,
            project_id=project_id,
            scope=eff_scope,
            key=key,
            when=_utcnow(),
        )
        return n > 0


__all__ = [
    "InMemoryStateBackend",
    "PostgresStateBackend",
    "StateBackend",
    "StateRepo",
]
