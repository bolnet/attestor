"""HTTP client / ASGI server parity tests.

Pinpoints two known drift bugs and locks the parity contract:

1. ``MemoryClient.recall(namespace=...)`` MUST forward ``namespace`` in
   the JSON body so namespace-scoped recall works over HTTP just like
   it does in the embedded ``AgentMemory``.

2. The ASGI ``/add`` endpoint MUST translate
   :class:`attestor.quotas.QuotaExceeded` to **HTTP 429** with the
   standard ``{"ok": false, "error": "quota exceeded: ..."}`` body so
   clients can distinguish quota breaches from other failures.

Both tests are hermetic — no Postgres / Neo4j required.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from attestor import api as attestor_api
from attestor.client import MemoryClient
from attestor.models import Memory
from attestor.quotas import QuotaExceeded


# ──────────────────────────────────────────────────────────────────────────
# Fix 1 — MemoryClient.recall() forwards `namespace`
# ──────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal urllib response object compatible with the client's reader."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode())

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._buf.close()


@pytest.mark.unit
def test_recall_forwards_namespace_in_body() -> None:
    """``recall(namespace="ns1")`` must put ``namespace`` in the POST body."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"ok": True, "data": []})

    client = MemoryClient("http://memory.test", agent_id="planner-01")

    with patch("attestor.client.urllib.request.urlopen", fake_urlopen):
        results: list[Any] = client.recall(query="x", namespace="ns1")

    assert results == []
    assert captured["url"] == "http://memory.test/recall"
    assert captured["body"]["query"] == "x"
    assert captured["body"]["namespace"] == "ns1"


@pytest.mark.unit
def test_recall_omits_namespace_when_not_set() -> None:
    """Back-compat: callers without ``namespace`` must keep working."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"ok": True, "data": []})

    client = MemoryClient("http://memory.test")

    with patch("attestor.client.urllib.request.urlopen", fake_urlopen):
        client.recall(query="x")

    assert "namespace" not in captured["body"]


# ──────────────────────────────────────────────────────────────────────────
# Fix 2 — QuotaExceeded → HTTP 429
# ──────────────────────────────────────────────────────────────────────────


class _FakeMem:
    """Tiny stand-in for AgentMemory that raises QuotaExceeded on add()."""

    def add(self, **kwargs: Any) -> Memory:
        raise QuotaExceeded(
            field="max_writes_per_day", limit=100, current=100,
        )


@pytest.mark.unit
def test_add_endpoint_returns_429_on_quota_exceeded() -> None:
    """/add must surface QuotaExceeded as 429 with a quota error body."""
    fake = _FakeMem()
    with patch.object(attestor_api, "_get_mem", return_value=fake):
        client = TestClient(attestor_api.app)
        resp = client.post("/add", json={"content": "hello"})

    assert resp.status_code == 429, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert "quota" in body["error"].lower()
    # Limit type should be surfaced so callers can build a useful message.
    assert "max_writes_per_day" in body["error"]


@pytest.mark.unit
def test_add_endpoint_400_when_content_missing() -> None:
    """Sanity check: pre-existing 400 on missing content still works."""
    fake = _FakeMem()
    with patch.object(attestor_api, "_get_mem", return_value=fake):
        client = TestClient(attestor_api.app)
        resp = client.post("/add", json={})

    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert "content" in body["error"]
