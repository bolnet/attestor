# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""A backend outage during ``delete_by_user`` must RAISE, never report success.

Bug under test (live e2e 2026-09-03, plan Phase 3 acceptance): with Neo4j
stopped, ``Neo4jBackend.delete_by_user`` swallowed the driver error and
returned ``(0, 0)``; the ForgetUser saga marked the graph lane ``ok`` and
COMPLETED while the user's nodes were still in the graph. The Pinecone
namespace probe had the same shape. Retry semantics live in the durable
layer; the store must surface the failure for them to apply.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from neo4j.exceptions import ServiceUnavailable

from attestor.store import pinecone_backend as pb
from attestor.store.neo4j_backend import Neo4jBackend
from attestor.store.pinecone_backend import PineconeBackend

pytestmark = pytest.mark.unit


# ── Neo4j ────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def single(self) -> dict[str, Any] | None:
        return self._row


class _Session:
    def __init__(self, *, fail: Exception | None, nodes: int, edges: int) -> None:
        self.fail = fail
        self.nodes, self.edges = nodes, edges
        self.cypher: list[str] = []

    def run(self, cypher: str, **params: Any) -> _Result:
        self.cypher.append(cypher)
        if self.fail is not None:
            raise self.fail
        if "RETURN" in cypher:
            return _Result({"nodes": self.nodes, "edges": self.edges})
        return _Result(None)

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _neo4j(session: _Session) -> Neo4jBackend:
    b = Neo4jBackend.__new__(Neo4jBackend)
    b._driver = SimpleNamespace(session=lambda database=None: session)
    b._database = "neo4j"
    return b


def test_neo4j_delete_by_user_propagates_outage() -> None:
    session = _Session(fail=ServiceUnavailable("Connection refused"), nodes=0, edges=0)
    with pytest.raises(ServiceUnavailable):
        _neo4j(session).delete_by_user("user-1")


def test_neo4j_delete_by_user_deletes_and_counts() -> None:
    session = _Session(fail=None, nodes=5, edges=3)
    assert _neo4j(session).delete_by_user("user-1") == (5, 3)
    assert any("DETACH DELETE" in c for c in session.cypher)


def test_neo4j_delete_by_user_zero_rows_skips_delete() -> None:
    session = _Session(fail=None, nodes=0, edges=0)
    assert _neo4j(session).delete_by_user("user-1") == (0, 0)
    assert not any("DETACH DELETE" in c for c in session.cypher)


# ── Pinecone ─────────────────────────────────────────────────────────────

class _DownIndex:
    def describe_index_stats(self) -> None:
        raise RuntimeError("UNAVAILABLE: connection refused")

    def delete(self, **kwargs: Any) -> None:
        raise RuntimeError("UNAVAILABLE: connection refused")


class _ControlPlane:
    def list_indexes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="mem")]

    def describe_index(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(host="h", status=SimpleNamespace(ready=True))


def test_pinecone_delete_by_user_propagates_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    down = _DownIndex()
    monkeypatch.setattr(pb, "bind_local_grpc_index", lambda *a, **k: down)
    monkeypatch.setattr(pb.PineconeBackend, "_wait_until_ready", lambda self, timeout=30.0: None)
    b = PineconeBackend.__new__(PineconeBackend)
    b._pc = _ControlPlane()
    b._index_name = "mem"
    b._index = down
    b._is_local = True
    b._api_key = "pclocal"
    b._ready = True
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        b.delete_by_user("user-1")
