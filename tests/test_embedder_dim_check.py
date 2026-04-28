"""Startup-time assertion that the embedder's output dim matches the
pgvector schema dim.

The pgvector schema column `embedding vector(N)` is fixed at table-create
time. If the embedder produces D-dim vectors and N != D, every UPDATE on
the embedding column silently no-ops because the doc path swallows
non-fatal vector errors. We've been bitten by this before — switching
from text-embedding-3-large (3072-D) to voyage-4 (1024-D) without
migrating the schema gave us an Attestor that "accepted" writes but
stored nothing.

The check fires during ``AgentMemory.__init__`` once the document store
is up. It MUST raise (not warn) on mismatch and tell the user both
numbers plus the two remediation paths (reconfigure embedder, or migrate
schema). When the table doesn't exist yet (greenfield first init), the
check is skipped — the schema will get created at the right dim by the
backend's own init path.

These tests are hermetic: a synthetic backend stub returns the schema dim
we want, the embedder's dim is forced via ``backend_configs``, and no
real Postgres / embedding provider is required.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from attestor.store.embedder_dim_check import (
    EmbedderDimMismatchError,
    assert_embedder_dim_matches_schema,
    get_schema_embedding_dim,
)


# ───────────────────────────────────────────────────────────────────────
# Stubs — mimic the parts of PostgresBackend we touch
# ───────────────────────────────────────────────────────────────────────


class _StubCursor:
    """psycopg2-style cursor that returns one canned row per execute()."""

    def __init__(self, rows_per_call: list) -> None:
        # rows_per_call is a queue: each execute() pops the next row.
        # Use None for "no row" (empty result).
        self._queue = list(rows_per_call)
        self._last_row: Optional[tuple] = None
        self.executed: list[str] = []

    def __enter__(self) -> "_StubCursor":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        self.executed.append(sql)
        if self._queue:
            self._last_row = self._queue.pop(0)
        else:
            self._last_row = None

    def fetchone(self):
        return self._last_row


class _StubConn:
    """psycopg2-style connection that hands out _StubCursors."""

    def __init__(self, rows_per_call: list) -> None:
        self._rows = rows_per_call

    def cursor(self, cursor_factory=None) -> _StubCursor:
        return _StubCursor(self._rows)


def _make_backend(schema_dim: Optional[int], embedder_dim: int):
    """Construct a PostgresBackend-shaped stub.

    schema_dim:
        - int  → the memories.embedding column has vector(schema_dim)
        - None → the memories table does not exist (or column missing)

    embedder_dim:
        - the dimension we'll claim our embedder produces.
    """
    # The introspection query is structured to return either a single-row
    # tuple of (dim,) or None when the table/column is absent.
    rows = [(schema_dim,)] if schema_dim is not None else [None]
    conn = _StubConn(rows)

    embedder = MagicMock()
    embedder.dimension = embedder_dim
    embedder.provider_name = f"stub:{embedder_dim}d"

    backend = MagicMock()
    backend._conn = conn
    backend._embedder = embedder
    backend._embedding_dim = embedder_dim

    return backend


# ───────────────────────────────────────────────────────────────────────
# get_schema_embedding_dim
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_schema_dim_returns_int_when_column_exists() -> None:
    backend = _make_backend(schema_dim=1024, embedder_dim=1024)
    assert get_schema_embedding_dim(backend._conn) == 1024


@pytest.mark.unit
def test_get_schema_dim_returns_none_when_column_missing() -> None:
    backend = _make_backend(schema_dim=None, embedder_dim=1024)
    # No row → no column / no table → None
    assert get_schema_embedding_dim(backend._conn) is None


@pytest.mark.unit
def test_get_schema_dim_returns_none_on_query_error() -> None:
    """If the introspection query raises (e.g. permissions, search path),
    treat as 'unknown' and skip rather than blowing up startup."""

    class _ExplodingConn:
        def cursor(self, cursor_factory=None):
            raise RuntimeError("permission denied for pg_attribute")

    assert get_schema_embedding_dim(_ExplodingConn()) is None


# ───────────────────────────────────────────────────────────────────────
# assert_embedder_dim_matches_schema — match path
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_match_does_not_raise() -> None:
    backend = _make_backend(schema_dim=1024, embedder_dim=1024)
    # Should be a no-op (no exception, no return value contract).
    assert_embedder_dim_matches_schema(backend)


@pytest.mark.unit
def test_match_with_3072_dim_does_not_raise() -> None:
    """Different but matching dim — still fine."""
    backend = _make_backend(schema_dim=3072, embedder_dim=3072)
    assert_embedder_dim_matches_schema(backend)


# ───────────────────────────────────────────────────────────────────────
# Mismatch path — must raise with both numbers + remediation
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mismatch_raises_with_both_dims_in_message() -> None:
    """The classic failure mode: embedder=1024, schema=1536."""
    backend = _make_backend(schema_dim=1536, embedder_dim=1024)
    with pytest.raises(
        (EmbedderDimMismatchError, ValueError, RuntimeError),
    ) as excinfo:
        assert_embedder_dim_matches_schema(backend)
    msg = str(excinfo.value)
    # Both numbers must appear so the user can immediately see the gap.
    assert "1024" in msg
    assert "1536" in msg


@pytest.mark.unit
def test_mismatch_message_mentions_remediation_paths() -> None:
    """The error has to point at the two remediation paths so the user
    isn't left guessing — reconfigure the embedder OR migrate the schema."""
    backend = _make_backend(schema_dim=1536, embedder_dim=1024)
    with pytest.raises(
        (EmbedderDimMismatchError, ValueError, RuntimeError),
    ) as excinfo:
        assert_embedder_dim_matches_schema(backend)
    msg = str(excinfo.value).lower()
    # Reconfigure path
    assert "embedder" in msg
    # Migrate path
    assert (
        "alter table" in msg
        or "migrate" in msg
        or "drop column" in msg
    )


@pytest.mark.unit
def test_mismatch_message_includes_provider_name() -> None:
    """When the embedder exposes a provider_name, surface it so the user
    knows which embedder produced the mismatch (e.g. 'voyage:voyage-4')."""
    backend = _make_backend(schema_dim=1536, embedder_dim=1024)
    backend._embedder.provider_name = "voyage:voyage-4"
    with pytest.raises(
        (EmbedderDimMismatchError, ValueError, RuntimeError),
    ) as excinfo:
        assert_embedder_dim_matches_schema(backend)
    msg = str(excinfo.value)
    assert "voyage" in msg.lower()


# ───────────────────────────────────────────────────────────────────────
# Table missing — must NOT raise (greenfield first init)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_table_missing_skips_check() -> None:
    """When memories.embedding doesn't exist yet, this is the very first
    init and the table is about to be created at the embedder's dim. Skip
    the check rather than rejecting the bootstrap."""
    backend = _make_backend(schema_dim=None, embedder_dim=1024)
    # Should not raise.
    assert_embedder_dim_matches_schema(backend)


@pytest.mark.unit
def test_query_error_skips_check() -> None:
    """When the introspection query raises (e.g. RLS denies pg_attribute,
    role lacks SELECT on system catalogs), don't break startup over a
    diagnostic check. Skip silently — recall errors will surface the real
    issue, and the doctor command can re-run the check explicitly."""

    class _ExplodingConn:
        def cursor(self, cursor_factory=None):
            raise RuntimeError("permission denied")

    embedder = MagicMock()
    embedder.dimension = 1024
    embedder.provider_name = "stub"

    backend = MagicMock()
    backend._conn = _ExplodingConn()
    backend._embedder = embedder
    backend._embedding_dim = 1024

    # Should not raise.
    assert_embedder_dim_matches_schema(backend)


# ───────────────────────────────────────────────────────────────────────
# Backend without an embedder — skip (e.g. embedding_dim was passed
# explicitly, no embedder probed)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_backend_without_embedder_skips_check() -> None:
    """If the backend has no _embedder attribute (e.g. a non-Postgres
    document store, or a Postgres backend booted with embedding_dim
    pre-set + skip_schema_init), there's nothing to compare. Skip."""
    backend = MagicMock(spec=[])  # no attributes
    # spec=[] → any attribute access raises AttributeError
    # Should not raise.
    assert_embedder_dim_matches_schema(backend)


@pytest.mark.unit
def test_backend_without_conn_skips_check() -> None:
    """Non-Postgres backends (Arango, AWS, Cosmos) don't expose _conn;
    the check only fires for the Postgres backend. Skip on absence."""
    embedder = MagicMock()
    embedder.dimension = 1024
    backend = MagicMock(spec=["_embedder"])
    backend._embedder = embedder
    # Should not raise.
    assert_embedder_dim_matches_schema(backend)


# ───────────────────────────────────────────────────────────────────────
# Wired into AgentMemory.__init__: a constructed AgentMemory whose store
# would mismatch must raise at construction time, not later at recall.
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_agent_memory_init_raises_on_dim_mismatch(tmp_path) -> None:
    """End-to-end: when AgentMemory wires a Postgres-shaped store whose
    schema dim != embedder dim, construction itself must blow up.

    We patch out the heavy bits (registry, config layering) so this stays
    a pure unit test."""
    from attestor.core import AgentMemory

    # A backend stub that satisfies the duck-typed surface AgentMemory
    # touches in __init__: stats(), close(), _v4 attribute, _conn
    # (introspection target), _embedder (dim source).
    schema_dim = 1536
    embedder_dim = 1024
    backend = _make_backend(schema_dim=schema_dim, embedder_dim=embedder_dim)
    # AgentMemory checks this attribute to gate v4 features.
    backend._v4 = False

    # Patch the registry's instantiate_backend so AgentMemory wires our stub.
    with patch(
        "attestor.core.instantiate_backend", return_value=backend,
    ), patch(
        "attestor.core.resolve_backends",
        return_value={"document": "postgres", "vector": "postgres", "graph": "postgres"},
    ), patch(
        "attestor.core.DEFAULT_BACKENDS", ["postgres"],
    ):
        with pytest.raises(
            (EmbedderDimMismatchError, ValueError, RuntimeError),
        ) as excinfo:
            AgentMemory(tmp_path / "mem")
    msg = str(excinfo.value)
    assert str(schema_dim) in msg
    assert str(embedder_dim) in msg
