# SPDX-License-Identifier: MIT
"""Unit tests for hook tenant resolution and per-project hook wiring.

All tests use a fake AgentMemory so no live Postgres/Pinecone/Neo4j is needed.
"""

from pathlib import Path
from types import SimpleNamespace

from attestor._project import (
    project_external_id,
    project_namespace,
    resolve_project_root,
)
from attestor.hooks._tenant import resolve_tenant


class _FakeMem:
    """Records ensure_user calls and returns a deterministic user id."""

    def __init__(self):
        self.ensured: list[str] = []

    def ensure_user(self, external_id: str, **_):
        self.ensured.append(external_id)
        return SimpleNamespace(id=f"uid::{external_id}")


def test_resolve_tenant_keys_user_to_project_root(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    mem = _FakeMem()

    user_id, namespace = resolve_tenant(mem, str(repo))

    root = resolve_project_root(str(repo))
    assert user_id == f"uid::{project_external_id(root)}"
    assert namespace == project_namespace(root)
    assert mem.ensured == [project_external_id(root)]


def test_resolve_tenant_distinct_dirs_distinct_users(tmp_path: Path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    mem = _FakeMem()

    ua, _ = resolve_tenant(mem, str(a))
    ub, _ = resolve_tenant(mem, str(b))

    assert ua != ub


def test_resolve_tenant_degrades_when_ensure_user_missing(tmp_path: Path):
    """A store without ensure_user (v3) yields user_id=None but keeps ns."""
    user_id, namespace = resolve_tenant(object(), str(tmp_path))
    assert user_id is None
    assert namespace == project_namespace(resolve_project_root(str(tmp_path)))


def test_resolve_tenant_degrades_when_ensure_user_raises(tmp_path: Path):
    class _Raises:
        def ensure_user(self, **_):
            raise RuntimeError("postgres down")

    user_id, namespace = resolve_tenant(_Raises(), str(tmp_path))
    assert user_id is None
    assert namespace == project_namespace(resolve_project_root(str(tmp_path)))


def test_session_start_recall_is_tenant_scoped(monkeypatch, tmp_path):
    """session_start must pass the project user_id + namespace to recall()."""
    import attestor._paths
    import attestor.core
    import attestor.hooks.session_start as ss

    captured = {}

    class _Mem:
        def __init__(self, *a, **k):
            pass

        def ensure_user(self, external_id, **_):
            return SimpleNamespace(id=f"uid::{external_id}")

        def recall(self, query, budget=None, user_id=None, namespace=None, **_):
            captured["user_id"] = user_id
            captured["namespace"] = namespace
            return []

        def pagerank(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(attestor.core, "AgentMemory", _Mem)
    monkeypatch.setattr(attestor._paths, "resolve_store_path", lambda *a, **k: str(tmp_path))

    out = ss.handle({"cwd": str(tmp_path)})

    root = resolve_project_root(str(tmp_path))
    assert captured["user_id"] == f"uid::{project_external_id(root)}"
    assert captured["namespace"] == project_namespace(root)
    assert "additionalContext" in out


def test_post_tool_use_add_is_tenant_scoped(monkeypatch, tmp_path):
    import attestor._paths
    import attestor.core
    import attestor.hooks.post_tool_use as ptu

    captured = {}

    class _Mem:
        def __init__(self, *a, **k):
            pass

        def ensure_user(self, external_id, **_):
            return SimpleNamespace(id=f"uid::{external_id}")

        def add(self, content, tags=None, category="general",
                user_id=None, namespace=None, scope="user", **_):
            captured.update(
                user_id=user_id, namespace=namespace, scope=scope,
                content=content,
            )

        def close(self):
            pass

    monkeypatch.setattr(attestor.core, "AgentMemory", _Mem)
    monkeypatch.setattr(attestor._paths, "resolve_store_path", lambda *a, **k: str(tmp_path))

    ptu.handle({
        "cwd": str(tmp_path),
        "tool_name": "Write",
        "tool_input": {"file_path": "foo.py"},
    })

    root = resolve_project_root(str(tmp_path))
    assert captured["user_id"] == f"uid::{project_external_id(root)}"
    assert captured["namespace"] == project_namespace(root)
    assert captured["scope"] == "project"


def test_stop_summary_reads_and_writes_are_tenant_scoped(monkeypatch, tmp_path):
    import attestor._paths
    import attestor.core
    import attestor.hooks.stop as st

    captured = {}

    class _Mem:
        def __init__(self, *a, **k):
            self._store = SimpleNamespace(_v4=True)
            self.resolved: list[str] = []

        def ensure_user(self, external_id, **_):
            return SimpleNamespace(id=f"uid::{external_id}")

        def _resolve(self, user_id=None, autostart=True, **_):
            self.resolved.append(user_id)
            return SimpleNamespace(
                user=SimpleNamespace(id=user_id),
                project=SimpleNamespace(id="p"),
                session=None,
            )

        def search(self, query=None, after=None, before=None, limit=10,
                   namespace=None, **_):
            captured["search_ns"] = namespace
            captured["resolved"] = list(self.resolved)
            return [SimpleNamespace(
                category="project", tags=["file-change"],
                content="Edited file foo.py",
            )]

        def add(self, content, tags=None, category="general",
                user_id=None, namespace=None, scope="user", **_):
            captured.update(user_id=user_id, namespace=namespace, scope=scope)

        def close(self):
            pass

    monkeypatch.setattr(attestor.core, "AgentMemory", _Mem)
    monkeypatch.setattr(attestor._paths, "resolve_store_path", lambda *a, **k: str(tmp_path))

    st.handle({"cwd": str(tmp_path)})

    root = resolve_project_root(str(tmp_path))
    expected_uid = f"uid::{project_external_id(root)}"
    expected_ns = project_namespace(root)
    # RLS switched to the project user BEFORE the time-window read.
    assert captured["resolved"] == [expected_uid]
    assert captured["search_ns"] == expected_ns
    # Summary write is project-scoped.
    assert captured["user_id"] == expected_uid
    assert captured["namespace"] == expected_ns
    assert captured["scope"] == "project"
