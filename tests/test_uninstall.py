# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Tests for the complete-uninstall settings surgery.

The critical guarantee: removing Attestor's Claude Code wiring must strip ONLY
Attestor's own MCP entry + hooks and never touch another tool's hooks (ECC,
continuous-learning, etc.). _strip_attestor is pure, so we test it directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "attestor_uninstall",
    Path(__file__).resolve().parent.parent / "scripts" / "attestor_uninstall.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_strip_attestor = _mod._strip_attestor


def _attestor_hook(event_cmd: str) -> dict:
    return {"hooks": [{"type": "command", "command": event_cmd}]}


def test_removes_attestor_mcp_keys_only():
    settings = {"mcpServers": {"attestor": {"command": "attestor"},
                               "memory": {"command": "attestor"},
                               "playwright": {"command": "npx"}}}
    new, changes = _strip_attestor(settings)
    assert set(new["mcpServers"]) == {"playwright"}
    assert any("attestor" in c for c in changes)
    # input not mutated
    assert "attestor" in settings["mcpServers"]


def test_preserves_other_tools_hooks_under_same_event():
    settings = {"hooks": {"SessionStart": [
        _attestor_hook("bash -c 'set -a; . ~/.attestor/.env; set +a; attestor hook session-start'"),
        _attestor_hook("ecc-session-start --foo"),
    ]}}
    new, changes = _strip_attestor(settings)
    remaining = new["hooks"]["SessionStart"]
    assert len(remaining) == 1
    assert "ecc-session-start" in remaining[0]["hooks"][0]["command"]
    assert changes  # something was removed


def test_drops_event_when_only_attestor():
    settings = {"hooks": {"Stop": [_attestor_hook("attestor hook stop")]}}
    new, _ = _strip_attestor(settings)
    assert "Stop" not in new.get("hooks", {})


def test_noop_when_no_attestor_entries():
    settings = {"mcpServers": {"other": {}},
                "hooks": {"PreToolUse": [_attestor_hook("some-other-tool run")]}}
    new, changes = _strip_attestor(settings)
    assert changes == []
    assert new == settings


def test_real_project_settings_only_loses_attestor_hooks():
    """Sanity: a settings file with attestor + foreign hooks keeps the foreign ones."""
    settings = {
        "permissions": {"defaultMode": "bypassPermissions"},
        "hooks": {
            "PostToolUse": [
                _attestor_hook("bash -c 'set -a; . \"$HOME/.attestor/.env\"; set +a; attestor hook post-tool-use'"),
                _attestor_hook("plankton-format"),
            ],
        },
    }
    new, _ = _strip_attestor(settings)
    cmds = [h["command"] for e in new["hooks"]["PostToolUse"] for h in e["hooks"]]
    assert cmds == ["plankton-format"]
    # unrelated top-level keys survive
    assert new["permissions"]["defaultMode"] == "bypassPermissions"
