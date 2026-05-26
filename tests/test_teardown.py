# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for ``attestor teardown`` removal logic (MCP entry + hooks)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from attestor.cli.commands.teardown import _remove_hooks, _remove_mcp_entry

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_remove_mcp_entry_drops_attestor_keeps_others(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mcp = tmp_path / ".mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {
        "attestor": {"command": "bash"},
        "memory": {"command": "bash"},      # legacy name — also dropped
        "other-tool": {"command": "node"},  # must be preserved
    }}))

    _remove_mcp_entry(dry_run=False)

    servers = json.loads(mcp.read_text())["mcpServers"]
    assert "attestor" not in servers
    assert "memory" not in servers
    assert "other-tool" in servers  # never touch another tool's entry


def test_remove_mcp_entry_dry_run_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mcp = tmp_path / ".mcp.json"
    original = json.dumps({"mcpServers": {"attestor": {"command": "bash"}}})
    mcp.write_text(original)

    _remove_mcp_entry(dry_run=True)

    assert mcp.read_text() == original  # untouched


def test_remove_hooks_content_matched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings = settings_dir / "settings.json"
    settings.write_text(json.dumps({"hooks": {
        "PostToolUse": [
            {"hooks": [
                {"type": "command", "command": "bash -c '... attestor hook post-tool-use'"},
                {"type": "command", "command": "prettier --write"},  # another tool — keep
            ]},
        ],
        "Stop": [
            {"hooks": [{"type": "command", "command": "bash -c '... attestor hook stop'"}]},
        ],
    }}))

    _remove_hooks(dry_run=False)

    hooks = json.loads(settings.read_text())["hooks"]
    # The Stop event had only Attestor's hook -> event dropped entirely.
    assert "Stop" not in hooks
    # PostToolUse keeps the other tool's hook, drops Attestor's.
    commands = [h["command"] for entry in hooks["PostToolUse"] for h in entry["hooks"]]
    assert all("attestor hook" not in c for c in commands)
    assert any("prettier" in c for c in commands)


def test_remove_hooks_no_attestor_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings = settings_dir / "settings.json"
    original = json.dumps({"hooks": {"PostToolUse": [
        {"hooks": [{"type": "command", "command": "eslint ."}]},
    ]}})
    settings.write_text(original)

    _remove_hooks(dry_run=False)

    assert json.loads(settings.read_text()) == json.loads(original)  # unchanged
