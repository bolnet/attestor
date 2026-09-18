# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""One re-entrant lock per DBAPI connection — shared by EVERY SQL path.

psycopg2 cursors are not thread-safe, and the ``attestor worker`` runs
its sync activities on a thread pool over ONE ``AgentMemory`` (one
connection). ``PostgresBackend._execute`` already serialised its own
statements with a per-instance lock, but the raw-SQL helpers in
``compliance/`` and ``identity/`` opened cursors straight on the
connection and bypassed it.

This registry is the single source of truth: :func:`lock_for_connection`
returns the SAME :class:`threading.RLock` for the same connection object,
whether the caller holds the backend or only the raw connection. The lock
is re-entrant so a caller may hold it across a multi-statement unit (RLS
scope set → read → write) while the inner helpers lock again.
"""

from __future__ import annotations

import logging
import threading
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("attestor.store.conn_lock")

LOCK_ATTRIBUTE = "_attestor_conn_lock"

_registry_guard = threading.Lock()
_locks: weakref.WeakKeyDictionary[Any, threading.RLock] = weakref.WeakKeyDictionary()


def _from_registry(conn: Any) -> threading.RLock | None:
    """Registry lookup / insert; ``None`` when ``conn`` is not weak-referenceable."""
    try:
        existing = _locks.get(conn)
        if existing is not None:
            return existing
        created = threading.RLock()
        _locks[conn] = created
    except TypeError:
        return None
    return created


def _from_attribute(conn: Any) -> threading.RLock | None:
    """Fallback for objects without weakref support: stash the lock on them."""
    existing = getattr(conn, LOCK_ATTRIBUTE, None)
    if existing is not None:
        return existing
    created = threading.RLock()
    try:
        setattr(conn, LOCK_ATTRIBUTE, created)
    except (AttributeError, TypeError):
        return None
    return created


def lock_for_connection(conn: Any) -> threading.RLock:
    """The one lock every statement on ``conn`` must hold.

    Never raises: an object that supports neither weak references nor
    attributes (should not happen for a real driver connection) gets a
    fresh lock — serialisation is then per call, and a warning says so.
    """
    with _registry_guard:
        lock = _from_registry(conn) or _from_attribute(conn)
    if lock is None:
        logger.warning(
            "connection %s supports neither weakref nor attributes; "
            "statements on it cannot be serialised across callers",
            type(conn).__name__,
        )
        return threading.RLock()
    return lock


@contextmanager
def locked_cursor(conn: Any, **cursor_kwargs: Any) -> Iterator[Any]:
    """``conn.cursor(**cursor_kwargs)`` opened and closed under the connection lock."""
    with lock_for_connection(conn), conn.cursor(**cursor_kwargs) as cur:
        yield cur
