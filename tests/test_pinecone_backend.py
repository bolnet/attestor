# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for ``PineconeBackend.delete`` / ``delete_by_user`` namespace scoping.

Bug under test (found in code review): ``delete(memory_id)`` and
``delete_by_user``'s metadata-filter path both hardcoded
``namespace="default"``. A forget_user / retention delete on a memory
that lives in any non-default Pinecone namespace silently left its
vector behind while reporting success — breaking the audited-forget
guarantee.

These tests construct ``PineconeBackend`` via ``__new__`` (bypassing
``__init__``, which talks to a real/local Pinecone control plane) and
wire in a fake index double so the namespace-scoping logic can be
exercised without network access or a running Pinecone Local emulator.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from attestor.store.pinecone_backend import PineconeBackend

pytestmark = pytest.mark.unit


class _FakeIndex:
    """Records every ``delete`` call; simulates ``describe_index_stats``.

    ``filter_supported=False`` simulates Pinecone Local / Free-tier
    plans that reject the ``filter=`` kwarg on delete (raises, same as
    the real SDK does against those tiers).
    """

    def __init__(
        self,
        namespaces: dict[str, int] | None = None,
        filter_supported: bool = True,
    ) -> None:
        self.namespaces = namespaces or {"default": 0}
        self.filter_supported = filter_supported
        self.delete_calls: list[dict[str, Any]] = []

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)
        if kwargs.get("filter") is not None and not self.filter_supported:
            raise RuntimeError("filter delete not supported on this plan")

    def describe_index_stats(self) -> SimpleNamespace:
        ns = {
            name: SimpleNamespace(vector_count=count)
            for name, count in self.namespaces.items()
        }
        return SimpleNamespace(
            namespaces=ns,
            total_vector_count=sum(self.namespaces.values()),
        )


def _make_backend(index: _FakeIndex) -> PineconeBackend:
    """Build a PineconeBackend without running its network-bound __init__."""
    backend = PineconeBackend.__new__(PineconeBackend)
    backend._index = index
    backend._ready = True  # skip _wait_until_ready polling
    return backend


# ── delete(memory_id, namespace=...) ────────────────────────────────────


@pytest.mark.unit
def test_delete_hits_memory_namespace() -> None:
    """Per-id delete must target the memory's own namespace, not 'default'."""
    index = _FakeIndex()
    backend = _make_backend(index)

    ok = backend.delete("mem-1", namespace="tenant-a")

    assert ok is True
    assert index.delete_calls == [{"ids": ["mem-1"], "namespace": "tenant-a"}]


@pytest.mark.unit
def test_delete_default_namespace_unchanged() -> None:
    """Omitting namespace keeps the pre-fix default-namespace behavior."""
    index = _FakeIndex()
    backend = _make_backend(index)

    ok = backend.delete("mem-2")

    assert ok is True
    assert index.delete_calls == [{"ids": ["mem-2"], "namespace": "default"}]


@pytest.mark.unit
def test_delete_returns_false_on_backend_error() -> None:
    class _RaisingIndex(_FakeIndex):
        def delete(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    backend = _make_backend(_RaisingIndex())
    assert backend.delete("mem-3", namespace="tenant-b") is False


# ── delete_by_user(user_id) — must cross every namespace ────────────────


@pytest.mark.unit
def test_delete_by_user_applies_filter_to_every_namespace() -> None:
    """A user's vectors can live in any namespace; all must be purged."""
    index = _FakeIndex(namespaces={"default": 1, "tenant-a": 3, "tenant-b": 2})
    backend = _make_backend(index)

    backend.delete_by_user("user-42")

    namespaces_hit = {c["namespace"] for c in index.delete_calls}
    assert namespaces_hit == {"default", "tenant-a", "tenant-b"}
    for call in index.delete_calls:
        assert call["filter"] == {"user_id": {"$eq": "user-42"}}


@pytest.mark.unit
def test_delete_by_user_default_namespace_only_unchanged() -> None:
    """Single-namespace (default-only) deployments keep working as before."""
    index = _FakeIndex(namespaces={"default": 5})
    backend = _make_backend(index)

    backend.delete_by_user("user-1")

    assert len(index.delete_calls) == 1
    assert index.delete_calls[0]["namespace"] == "default"
    assert index.delete_calls[0]["filter"] == {"user_id": {"$eq": "user-1"}}


@pytest.mark.unit
def test_delete_by_user_falls_back_per_namespace_when_filter_unsupported() -> None:
    """Free/Local tiers reject filter=; fall back to delete_all scoped to
    a namespace that IS the user_id (single-tenant-per-namespace deploys),
    never wiping a *shared* namespace's other users."""
    index = _FakeIndex(
        namespaces={"default": 1, "user-7": 4},
        filter_supported=False,
    )
    backend = _make_backend(index)

    backend.delete_by_user("user-7")

    # Every namespace got a (failed) filter attempt...
    filter_attempts = [c for c in index.delete_calls if c.get("filter") is not None]
    assert {c["namespace"] for c in filter_attempts} == {"default", "user-7"}
    # ...and the fallback only nukes the namespace matching the user id,
    # never the shared "default" namespace.
    fallback_calls = [c for c in index.delete_calls if c.get("delete_all")]
    assert fallback_calls == [{"delete_all": True, "namespace": "user-7"}]


@pytest.mark.unit
def test_delete_by_user_no_namespaces_defaults_to_default() -> None:
    """describe_index_stats() reporting no namespaces still tries 'default'."""
    index = _FakeIndex(namespaces={})
    backend = _make_backend(index)

    backend.delete_by_user("user-9")

    assert index.delete_calls == [
        {"filter": {"user_id": {"$eq": "user-9"}}, "namespace": "default"},
    ]


@pytest.mark.unit
def test_delete_by_user_describe_stats_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing namespace probe must raise, never silently no-op.

    Contract change (live e2e 2026-09-03): an outage during forget used to
    degrade to ``['default']`` and report success with the user's vectors
    still present. The durable saga owns retries; the store must surface
    the failure.
    """
    class _StatsDown(_FakeIndex):
        def describe_index_stats(self) -> SimpleNamespace:
            raise RuntimeError("stats unavailable")

    idx = _StatsDown()
    backend = _make_backend(idx)
    # The one-shot heal probes readiness; keep the unit test off the 30 s poll.
    monkeypatch.setattr(PineconeBackend, "_wait_until_ready", lambda self, timeout=30.0: None)
    with pytest.raises(RuntimeError, match="stats unavailable"):
        backend.delete_by_user("user-1")
    assert idx.delete_calls == []
