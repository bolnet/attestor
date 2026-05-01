"""Claude Code / Cursor MCP setup helpers.

Shared between the ``init`` and ``setup-claude-code`` command handlers.
Extracted from the legacy ``attestor/cli.py`` byte-for-byte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _print_mcp_config(tool: str, binary: str, store_path: str):
    """Print MCP config for a specific tool."""
    entry = _mcp_entry(binary, store_path)
    config = {"mcpServers": {"memory": entry}}
    if tool == "claude-code":
        print("\nClaude Code -- add to .claude/settings.json:")
    elif tool == "cursor":
        print("\nCursor -- add to .cursor/mcp.json:")
    else:
        return
    print(json.dumps(config, indent=2))


def _mcp_entry(binary: str, store_path: str) -> dict:
    """Canonical shape of the attestor MCP server entry."""
    return {"command": binary, "args": ["mcp", "--path", store_path]}


def _load_claude_settings(settings_path: Path) -> dict:
    """Load ~/.claude/settings.json, backing up and warning on parse failure."""
    if not settings_path.exists():
        return {}
    raw = settings_path.read_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        backup = settings_path.with_suffix(".json.bak")
        backup.write_text(raw)
        print(
            f"WARNING: could not parse {settings_path} ({exc}); "
            f"backed up to {backup} and starting with an empty config.",
            file=sys.stderr,
        )
        return {}


def _configure_claude_mcp(binary: str, store_path: str) -> None:
    """Write the attestor MCP server entry into ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_claude_settings(settings_path)
    mcp_servers = settings.setdefault("mcpServers", {})
    mcp_servers["memory"] = _mcp_entry(binary, store_path)

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nClaude Code MCP server 'memory' configured in {settings_path}")


def _configure_claude_hooks(binary: str):
    """Write Claude Code lifecycle hooks to ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_claude_settings(settings_path)
    hooks = settings.setdefault("hooks", {})

    hook_defs = {
        "SessionStart": {"command": f"{binary} hook session-start"},
        "PostToolUse": {"command": f"{binary} hook post-tool-use"},
        "Stop": {"command": f"{binary} hook stop"},
    }

    for event, hook_cfg in hook_defs.items():
        event_hooks = hooks.setdefault(event, [])
        # Check if already configured
        already = any(
            h.get("command") == hook_cfg["command"]
            for entry in event_hooks
            for h in (entry.get("hooks", []) if isinstance(entry, dict) else [])
        )
        if not already:
            event_hooks.append({"hooks": [{"type": "command", **hook_cfg}]})

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print("\nClaude Code hooks configured in ~/.claude/settings.json")
    print("  - SessionStart: injects relevant memories into context")
    print("  - PostToolUse: auto-captures file changes and commands")
    print("  - Stop: generates session summary")
