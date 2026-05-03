"""Regression tests for the client/import safety bugs surfaced by the
2026-05-02 test-gap audit (post-PR #135).

Each test in this file maps 1:1 to a production bug uncovered by the audit:

  - Gap C1 — ``AgentMemory.import_json`` swallows insert errors so a
    JSON payload missing/null ``id`` fields silently drops rows.
  - Gap C2 — ``MemoryClient.add`` never accepted a ``namespace`` kwarg
    even though ``MemoryClient.recall`` does, so HTTP writes always
    landed in the ``"default"`` namespace and silently broke
    multi-tenant isolation.
  - Gap C3 — ``attestor.store.embeddings`` caches the first provider it
    builds and returns it on every subsequent call, even when the
    caller asks for a different provider. Combined with the
    skip-on-cache-hit behavior of the dim guard this re-incarnates the
    silent dim-mismatch bug.

Tests are hermetic where possible (no Postgres / Pinecone / Neo4j); the
one integration test that needs a real document store is gated on the
shared ``mem`` fixture and skips when ``POSTGRES_URL`` is unset.

See ``test_temporal_supersession_gaps.py`` for the header style this
file follows.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest

from attestor.client import MemoryClient
from attestor.models import Memory


# ──────────────────────────────────────────────────────────────────────────
# Shared fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal urllib response object compatible with the client's reader."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode())

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._buf.close()


# ──────────────────────────────────────────────────────────────────────────
# Gap C1 — import_json silently drops rows when ``id`` is missing/null
# ──────────────────────────────────────────────────────────────────────────


class _StrictPKStore:
    """Tiny fake DocumentStore that mimics a real PK constraint.

    - Rejects ``None`` / empty-string ids with ``ValueError`` (psycopg2 raises
      ``InvalidTextRepresentation`` on ``UUID NOT NULL`` columns; we surface
      the same shape with a generic exception so the
      ``except Exception: pass`` in ``import_json`` is the thing that gets
      exercised, not a UUID-library quirk).
    - Rejects duplicate ids with ``ValueError`` (PK collision).
    - Stores accepted memories in an in-memory dict keyed by id.
    """

    def __init__(self) -> None:
        self.rows: dict[str, Memory] = {}

    def insert(self, memory: Memory) -> Memory:
        if not memory.id:
            raise ValueError("id must be a non-empty string (PK NOT NULL)")
        if memory.id in self.rows:
            raise ValueError(f"duplicate id {memory.id!r} (PK violation)")
        self.rows[memory.id] = memory
        return memory


def _import_json_with_fake_store(
    items: list[dict[str, Any]],
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, _StrictPKStore]:
    """Run ``AgentMemory.import_json`` against the strict-PK fake store.

    We bypass ``AgentMemory.__init__`` (which builds Postgres + Pinecone
    + Neo4j) by attaching the fake store directly to a bare instance; the
    only thing ``import_json`` actually touches is ``self._store`` and
    ``self._content_hash``.
    """
    from attestor.core.agent_memory import AgentMemory

    mem = AgentMemory.__new__(AgentMemory)
    store = _StrictPKStore()
    mem._store = store  # type: ignore[attr-defined]

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False,
    ) as fh:
        json.dump(items, fh)
        path = fh.name
    try:
        count = mem.import_json(path)
    finally:
        os.unlink(path)
    return count, store


@pytest.mark.unit
def test_import_json_assigns_unique_id_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three JSON entries with no ``id`` field must all land in the store.

    Pre-fix: the fallback ``Memory().id`` works in CPython today but the
    contract is fragile — any change that memoised ``Memory()`` (e.g. a
    cached default-factory wrapper) would silently make every row reuse
    the same id and PK-collide. Plus a JSON payload with ``"id": null``
    or ``"id": ""`` in any single entry passes ``None``/empty straight to
    the store and the ``except Exception: pass`` swallows the error.
    """
    items = [
        {"content": "fact one"},
        {"content": "fact two"},
        {"content": "fact three"},
    ]
    count, store = _import_json_with_fake_store(items, monkeypatch=monkeypatch)
    assert count == 3, (
        f"expected all 3 id-less rows to land; only {count} did — "
        "import_json silently dropped rows"
    )
    assert len(store.rows) == 3
    # Each row got a distinct id.
    ids = {m.id for m in store.rows.values()}
    assert len(ids) == 3, f"expected 3 distinct ids, got {ids!r}"


@pytest.mark.unit
def test_import_json_handles_null_id_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry with ``"id": null`` must get a fresh id, not be dropped.

    Pre-fix: ``item.get("id", Memory().id)`` returns ``None`` when the key
    exists with a null value (``.get`` only uses the default for *missing*
    keys, not for null values). The store then rejects ``id=None`` and
    the silent-except swallows it.
    """
    items = [
        {"content": "ok one", "id": None},
        {"content": "ok two", "id": ""},
    ]
    count, store = _import_json_with_fake_store(items, monkeypatch=monkeypatch)
    assert count == 2, (
        f"expected null/empty id entries to be re-keyed and inserted; "
        f"only {count} of 2 landed"
    )
    ids = {m.id for m in store.rows.values()}
    assert all(i for i in ids), f"all rows must have a non-empty id; got {ids!r}"
    assert len(ids) == 2


@pytest.mark.unit
def test_import_json_preserves_caller_supplied_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: when ``id`` is present and non-empty, it's preserved."""
    items = [
        {"content": "with id", "id": "abc123def456"},
    ]
    count, store = _import_json_with_fake_store(items, monkeypatch=monkeypatch)
    assert count == 1
    assert "abc123def456" in store.rows


# ──────────────────────────────────────────────────────────────────────────
# Gap C2 — MemoryClient.add must accept and forward ``namespace``
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_add_forwards_namespace_in_body() -> None:
    """``add(namespace="ns_a")`` must put ``namespace`` in the POST body.

    Pre-fix: ``MemoryClient.add`` had no ``namespace`` kwarg at all. HTTP
    callers writing to a tenant-scoped namespace silently landed in the
    ``"default"`` namespace, which is a cross-tenant data-leak class
    bug, not a UX nit.
    """
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        # Return a minimal Memory.from_row()-compatible payload.
        return _FakeResponse({
            "ok": True,
            "data": {
                "id": "m1",
                "content": "hello",
                "tags": [],
                "category": "general",
                "namespace": "ns_a",
                "valid_from": "2024-01-01T00:00:00+00:00",
                "created_at": "2024-01-01T00:00:00+00:00",
                "status": "active",
                "confidence": 1.0,
                "metadata": {},
            },
        })

    client = MemoryClient("http://memory.test", agent_id="planner-01")

    with patch("attestor.client.urllib.request.urlopen", fake_urlopen):
        client.add("hello", namespace="ns_a")

    assert captured["url"] == "http://memory.test/add"
    assert captured["body"]["content"] == "hello"
    assert captured["body"]["namespace"] == "ns_a", (
        f"namespace must be forwarded in /add body, "
        f"got body keys: {sorted(captured['body'].keys())!r}"
    )


@pytest.mark.unit
def test_add_omits_namespace_when_not_set() -> None:
    """Back-compat: callers without ``namespace`` keep working.

    The server defaults to ``"default"`` when the body omits the key
    (see ``attestor/api.py::add_memory``), so the client must NOT
    inject a namespace silently — that would mask config errors.
    """
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({
            "ok": True,
            "data": {
                "id": "m1",
                "content": "hello",
                "tags": [],
                "category": "general",
                "namespace": "default",
                "valid_from": "2024-01-01T00:00:00+00:00",
                "created_at": "2024-01-01T00:00:00+00:00",
                "status": "active",
                "confidence": 1.0,
                "metadata": {},
            },
        })

    client = MemoryClient("http://memory.test")

    with patch("attestor.client.urllib.request.urlopen", fake_urlopen):
        client.add("hello")

    assert "namespace" not in captured["body"], (
        "client must not inject a namespace when caller omits it"
    )


# ──────────────────────────────────────────────────────────────────────────
# Gap C3 — get_embedding_provider returns stale cached provider
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_embedding_provider_cache_returns_stale_when_yaml_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-reading YAML must NOT return the previously-cached provider.

    Repro of the real cache-pollution bug:

      1. Code path A asks for an explicit provider via
         ``get_embedding_provider("voyage")`` — caches voyage.
      2. Code path B (e.g. the orchestrator boot path) asks for the
         YAML-configured provider via ``get_embedding_provider()`` (no
         arg). YAML has since been edited to say ``openai``.
      3. The cache short-circuit at the top of the function fires
         BEFORE the YAML re-read, so the caller silently gets voyage
         even though YAML now says openai.

    Combined with ``assert_embedder_dim_matches_schema``'s skip-on-
    introspection-error behavior, this re-incarnates the silent
    vector-dim drift bug we paid for in the 2026-04 schema migration.
    """
    from attestor.store import embeddings as emb_mod

    emb_mod.clear_embedding_cache()

    class _Stub:
        def __init__(self, name: str, dim: int) -> None:
            self.provider_name = name
            self.dimension = dim

    voyage_stub = _Stub("voyage", 1024)
    openai_stub = _Stub("openai", 1536)

    monkeypatch.setitem(
        emb_mod._PROVIDER_DISPATCH, "voyage", lambda: voyage_stub,
    )
    monkeypatch.setitem(
        emb_mod._PROVIDER_DISPATCH, "openai", lambda: openai_stub,
    )

    # Step 1 — explicit voyage request fills the cache.
    first = emb_mod.get_embedding_provider(preferred="voyage")
    assert first is voyage_stub

    # Step 2 — YAML now says "openai"; emulate get_stack() with a stub.
    class _StackStub:
        class embedder:
            provider = "openai"

    monkeypatch.setattr(
        "attestor.config.get_stack", lambda: _StackStub(),
    )

    # Step 3 — preferred=None forces YAML re-read; cache must NOT win.
    second = emb_mod.get_embedding_provider()
    assert second is openai_stub, (
        f"YAML now says openai but got {second.provider_name!r} — the "
        "single-slot cache returned a stale provider, which silently "
        "breaks dim checks downstream"
    )
