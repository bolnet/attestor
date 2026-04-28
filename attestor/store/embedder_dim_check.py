"""Embedder/schema dimension assertion — fail-fast guard.

Bug class: pgvector declares ``embedding vector(N)`` at table-create time.
If the active embedder produces D-dim vectors and ``D != N``, every UPDATE
that sets the embedding column is silently no-op'd because the document
write path swallows non-fatal vector errors (`attestor/core.py::add`).
The result is an Attestor that "accepts" writes but stores nothing in the
vector lane — invisible until recall returns zero hits.

This guard runs once at ``AgentMemory.__init__``:
  - reads ``memories.embedding`` typmod from ``pg_attribute`` to learn the
    schema's declared dim,
  - compares to the embedder's ``.dimension``,
  - raises ``EmbedderDimMismatchError`` with both numbers + remediation
    when they diverge,
  - skips when the table doesn't exist yet (greenfield first init: the
    backend is about to CREATE TABLE at the embedder's dim, no schema to
    check against),
  - skips when the introspection query errors (e.g. role lacks
    ``SELECT`` on system catalogs) — diagnostic checks should never break
    a startup that was otherwise viable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("attestor")


# pgvector encodes the vector(N) dimension into ``pg_attribute.atttypmod``.
# As of pgvector 0.5+, ``atttypmod`` IS the dimension itself (no "+ 4"
# adjustment like varchar). The canonical query is to read it directly
# off the column — ``format_type(atttypid, atttypmod)`` returns e.g.
# ``"vector(1024)"`` and is brittle to parse, so we use atttypmod.
_DIM_LOOKUP_SQL = """
    SELECT atttypmod
    FROM pg_attribute
    WHERE attrelid = to_regclass('public.memories')
      AND attname  = 'embedding'
      AND NOT attisdropped
"""


class EmbedderDimMismatchError(RuntimeError):
    """Raised when the embedder's output dim doesn't match the pgvector
    schema's declared dim. Carries both numbers + the provider name so
    the user can diagnose without re-running.

    Inherits ``RuntimeError`` so legacy ``except (ValueError, RuntimeError)``
    clauses around init still see the failure as fatal."""


def get_schema_embedding_dim(conn: Any) -> Optional[int]:
    """Return the declared dim of ``memories.embedding`` in the active DB.

    Returns:
        int  — the column's vector dim
        None — when the table/column doesn't exist, OR when the
               introspection query errors (caller treats both as
               "skip the check").

    The query is read-only and runs against ``pg_attribute``; it does
    not need RLS to be set up (system catalogs are not RLS-scoped). It
    only fails on hardened deployments where the runtime role has been
    denied SELECT on ``pg_attribute`` — extremely rare; in that case
    we return None and skip rather than block startup.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(_DIM_LOOKUP_SQL)
            row = cur.fetchone()
    except Exception as e:
        # Diagnostic check; never blow up startup. Logged so it's visible
        # in operator logs without becoming a hard error.
        logger.debug("schema dim introspection failed: %s", e)
        return None

    if row is None:
        # Table or column doesn't exist yet — greenfield first init.
        return None

    # row is a 1-tuple (atttypmod,). atttypmod < 0 means "no typmod set"
    # which would be a corrupted schema; treat as unknown and skip.
    dim = row[0]
    if dim is None or dim < 0:
        return None
    return int(dim)


def assert_embedder_dim_matches_schema(backend: Any) -> None:
    """Validate that ``backend._embedder.dimension`` equals the schema's
    declared dim for ``memories.embedding``. Raises on mismatch; no-ops
    on every "can't tell" case.

    Skip conditions (no raise):
      - backend has no ``_conn`` attribute (non-Postgres store)
      - backend has no ``_embedder`` attribute (e.g. embedding_dim
        passed explicitly without probing an embedder)
      - schema dim is None (table not created yet, OR query error)
      - embedder.dimension is None / not an int (defensive)

    Raises:
        EmbedderDimMismatchError: schema_dim != embedder_dim
    """
    conn = getattr(backend, "_conn", None)
    if conn is None:
        return

    embedder = getattr(backend, "_embedder", None)
    if embedder is None:
        return

    embedder_dim = getattr(embedder, "dimension", None)
    if not isinstance(embedder_dim, int) or embedder_dim <= 0:
        return

    schema_dim = get_schema_embedding_dim(conn)
    if schema_dim is None:
        # Table missing or unable to introspect — skip the check. The
        # backend's own _init_schema path will create the column at the
        # embedder's dim.
        return

    if schema_dim == embedder_dim:
        return

    provider_name = getattr(embedder, "provider_name", "unknown")
    raise EmbedderDimMismatchError(
        f"Embedder dimension mismatch: embedder={provider_name}({embedder_dim}-D) "
        f"but schema=vector({schema_dim}). "
        f"Either reconfigure the embedder (configs/attestor.yaml -> embedder.model "
        f"to one that emits {schema_dim}-D vectors) or migrate the schema:\n"
        f"  TRUNCATE memories;\n"
        f"  ALTER TABLE memories DROP COLUMN embedding;\n"
        f"  ALTER TABLE memories ADD COLUMN embedding vector({embedder_dim});\n"
        f"  CREATE INDEX idx_memories_embedding_hnsw "
        f"ON memories USING hnsw (embedding vector_cosine_ops);"
    )
