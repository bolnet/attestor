# SPDX-License-Identifier: MIT
"""Unit tests for the MCP server's per-project default tenant."""

from pathlib import Path
from types import SimpleNamespace

from attestor._project import (
    project_external_id,
    project_namespace,
    resolve_project_root,
)
from attestor.mcp import server as mcp_server


def test_default_tenant_derives_from_cwd(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)

    external_id, namespace = mcp_server._default_tenant_for_cwd()

    root = resolve_project_root(str(repo))
    assert namespace == project_namespace(root)
    assert external_id == project_external_id(root)


def test_handle_tool_add_applies_default_tenant():
    captured = {}

    class _Mem:
        def add(self, content, tags=None, category="general", entity=None,
                namespace=None, event_date=None, confidence=1.0,
                user_id=None, scope="user", **_):
            captured.update(namespace=namespace, user_id=user_id, scope=scope)
            return SimpleNamespace(id="m1", content=content)

    mcp_server._handle_tool(
        _Mem(), "memory_add", {"content": "hi"},
        default_user_id="u1", default_namespace="project:/x",
    )

    assert captured["namespace"] == "project:/x"
    assert captured["user_id"] == "u1"
    assert captured["scope"] == "project"


def test_handle_tool_explicit_namespace_overrides_default():
    captured = {}

    class _Mem:
        def add(self, content, tags=None, category="general", entity=None,
                namespace=None, event_date=None, confidence=1.0,
                user_id=None, scope="user", **_):
            captured.update(namespace=namespace)
            return SimpleNamespace(id="m1", content=content)

    mcp_server._handle_tool(
        _Mem(), "memory_add", {"content": "hi", "namespace": "custom"},
        default_user_id="u1", default_namespace="project:/x",
    )

    assert captured["namespace"] == "custom"


def test_handle_tool_recall_applies_default_tenant():
    captured = {}

    class _Mem:
        def recall(self, query, budget=2000, namespace=None, user_id=None, **_):
            captured.update(namespace=namespace, user_id=user_id)
            return []

    mcp_server._handle_tool(
        _Mem(), "memory_recall", {"query": "q"},
        default_user_id="u1", default_namespace="project:/x",
    )

    assert captured["namespace"] == "project:/x"
    assert captured["user_id"] == "u1"
