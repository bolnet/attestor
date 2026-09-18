# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``PineconeBackend`` self-heals when its index vanishes.

Bug under test (live e2e 2026-09-03, plan Phase 2 acceptance): Pinecone
Local is in-memory, so ``docker stop/start`` destroys the index. The
worker's cached backend kept probing readiness for an index that no
longer existed and every ``DeriveMemory`` retry timed out forever — the
repair saga could never converge. The same path bites a deleted cloud
index. Fix: on a missing index (readiness probe) or a failed data-plane
call, recreate + rebind once and retry.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from attestor.store import pinecone_backend as pb
from attestor.store.pinecone_backend import PineconeBackend

pytestmark = pytest.mark.unit


class _FakeControlPlane:
    def __init__(self, existing: set[str]) -> None:
        self.existing = set(existing)
        self.created: list[str] = []

    def list_indexes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in sorted(self.existing)]

    def create_index(self, *, name: str, dimension: int, metric: str, spec: Any) -> None:
        self.created.append(name)
        self.existing.add(name)

    def describe_index(self, name: str) -> SimpleNamespace:
        if name not in self.existing:
            raise RuntimeError(f"index {name} not found")
        return SimpleNamespace(host=f"localhost:5081/{name}", status=SimpleNamespace(ready=True))

    def Index(self, name: str) -> _FakeIndex:  # noqa: N802 - SDK name
        return _FakeIndex(fail_first=0)


class _FakeIndex:
    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.upserts: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("UNAVAILABLE: index host gone")
        self.upserts.append(kwargs)


def _backend(
    monkeypatch: pytest.MonkeyPatch, *, existing: set[str], is_local: bool,
) -> tuple[PineconeBackend, _FakeControlPlane, list[_FakeIndex]]:
    pc = _FakeControlPlane(existing)
    bound: list[_FakeIndex] = []

    def fake_bind(pc_arg: Any, *, index_name: str, host: str, api_key: str) -> _FakeIndex:
        idx = _FakeIndex(fail_first=0)
        bound.append(idx)
        return idx

    monkeypatch.setattr(pb, "bind_local_grpc_index", fake_bind)
    monkeypatch.setattr(pb.PineconeBackend, "_wait_until_ready", lambda self, timeout=30.0: None)
    b = PineconeBackend.__new__(PineconeBackend)
    b._pc = pc
    b._index_name = "mem"
    b._dimension = 4
    b._metric = "cosine"
    b._cloud = "aws"
    b._region = "us-east-1"
    b._serverless_spec_cls = lambda **kw: kw
    b._api_key = "pclocal"
    b._is_local = is_local
    b._index = _FakeIndex(fail_first=0)
    b._ready = False
    b._embedder = SimpleNamespace(embed=lambda text: [0.1, 0.2, 0.3, 0.4])
    return b, pc, bound


def test_ensure_ready_recreates_missing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    b, pc, bound = _backend(monkeypatch, existing=set(), is_local=True)

    b._ensure_ready()

    assert pc.created == ["mem"]
    assert len(bound) == 1
    assert b._index is bound[0]
    assert b._ready is True


def test_ensure_ready_skips_create_when_index_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    b, pc, bound = _backend(monkeypatch, existing={"mem"}, is_local=True)
    original = b._index

    b._ensure_ready()

    assert pc.created == []
    assert b._index is original


def test_add_heals_once_after_data_plane_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index vanished after the readiness flag was cached (emulator restart)."""
    b, pc, bound = _backend(monkeypatch, existing=set(), is_local=True)
    b._ready = True
    b._index = _FakeIndex(fail_first=1)

    b.add("m1", "hello", namespace="tenant-a")

    assert pc.created == ["mem"]
    assert len(bound) == 1
    (call,) = bound[0].upserts
    assert call["namespace"] == "tenant-a"
    assert call["vectors"][0]["id"] == "m1"


def test_add_raises_when_heal_does_not_help(monkeypatch: pytest.MonkeyPatch) -> None:
    b, pc, bound = _backend(monkeypatch, existing={"mem"}, is_local=True)
    b._ready = True
    always_broken = _FakeIndex(fail_first=99)
    b._index = always_broken
    monkeypatch.setattr(pb, "bind_local_grpc_index", lambda *a, **k: always_broken)

    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        b.add("m1", "hello")


def test_cloud_binding_uses_control_plane_index(monkeypatch: pytest.MonkeyPatch) -> None:
    b, pc, bound = _backend(monkeypatch, existing=set(), is_local=False)

    b._ensure_ready()

    assert pc.created == ["mem"]
    assert bound == []  # cloud never goes through the local gRPC binder
    assert isinstance(b._index, _FakeIndex)
