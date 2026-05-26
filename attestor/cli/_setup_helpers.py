# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Claude Code / Cursor MCP setup helpers.

Shared between the ``init`` and ``setup-claude-code`` command handlers.
Extracted from the legacy ``attestor/cli.py`` byte-for-byte.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


def _print_mcp_config(tool: str, binary: str, store_path: str):
    """Print MCP config for a specific tool."""
    entry = _mcp_entry(binary, store_path)
    config = {"mcpServers": {"attestor": entry}}
    if tool == "claude-code":
        print("\nClaude Code -- add to ./.mcp.json (project) or ~/.claude.json mcpServers (global):")
    elif tool == "cursor":
        print("\nCursor -- add to .cursor/mcp.json:")
    else:
        return
    print(json.dumps(config, indent=2))


def _mcp_entry(binary: str, store_path: str) -> dict:
    """A `.env`-sourcing bash-wrapper MCP entry.

    Claude Code spawns the MCP server with a **minimal environment**, so it
    needs `ATTESTOR_CONFIG` + the DB/embedder secrets + `PATH` (`~/.local/bin`
    for the pipx shim). An `env:{}` block can't source a file and would have to
    inline secrets; instead we source `~/.attestor/.env` (the same pattern the
    hooks use), which gives the server everything and keeps secrets out of the
    committed config. `set -a` is required so un-`export`ed `.env` lines reach
    the child.
    """
    return {
        "command": "bash",
        "args": [
            "-c",
            # Claude Code spawns the MCP server with a minimal, non-interactive
            # PATH that excludes ~/.local/bin (the pipx shim), so a bare
            # `attestor` is "command not found". Prepend it explicitly first.
            'export PATH="$HOME/.local/bin:$PATH"; '
            'set -a; [ -f "$HOME/.attestor/.env" ] && . "$HOME/.attestor/.env"; '
            f"set +a; exec attestor mcp --path {shlex.quote(str(store_path))}",
        ],
    }


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
    """Write the attestor MCP server entry into the project ``./.mcp.json``.

    Target choice matters: Claude Code reliably loads **project ``./.mcp.json``**
    and **global ``~/.claude.json`` ``mcpServers``** — it does NOT read
    ``~/.claude/.mcp.json`` and (per the install-test) servers written to
    ``~/.claude/settings.json`` don't get picked up either. We default to the
    project file: scoped, reliably loaded, and we never have to rewrite the
    user's large global ``~/.claude.json``. (For a global install, add the same
    entry to ``~/.claude.json`` ``mcpServers``.)
    """
    mcp_path = Path.cwd() / ".mcp.json"

    settings = _load_claude_settings(mcp_path)
    mcp_servers = settings.setdefault("mcpServers", {})
    mcp_servers["attestor"] = _mcp_entry(binary, store_path)
    # Drop the pre-2026-05 server name so a re-run doesn't leave a stale
    # duplicate 'memory' entry alongside the canonical 'attestor' one.
    mcp_servers.pop("memory", None)

    mcp_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nClaude Code MCP server 'attestor' configured in {mcp_path}")


def _print_active_config() -> None:
    """Show which single config file Attestor uses + the resolved stack.

    Attestor reads every stack / model / embedder / retrieval setting from
    one YAML file. Surfacing it during install makes the contract obvious:
    to change any setting, edit this file and re-run — there is no second
    place to look. Never crashes the install; if the file is missing it
    prints a pointer instead.
    """
    from attestor.config import (
        get_stack,
        print_stack_banner,
        resolved_config_path,
    )

    print("\nAttestor reads every setting from a single config file.")
    try:
        stack = get_stack()
    except SystemExit as exc:
        print(f"  {exc}")
        print(
            "  Create configs/attestor.yaml (or set ATTESTOR_CONFIG) and re-run — "
            "it is the single source of truth."
        )
        return
    path = resolved_config_path()
    if path is not None:
        print(f"  Config file in use: {path}")
    print("\nThis is how your current config looks like:")
    print_stack_banner(stack, run_label="setup")
    print(
        "To change any setting (embedder, models, backends, retrieval budget), "
        "edit that file and re-run setup — never hand-edit config elsewhere.\n"
    )


def _hook_command(binary: str, subcommand: str) -> str:
    """Build a hook command that loads the user's env before invoking attestor.

    Claude Code spawns hooks without the interactive shell's environment, so
    ``ATTESTOR_CONFIG`` and the embedding-provider keys are absent — the hook
    then falls back to the bundled default config, the embedder fails to
    initialize, and (because the handler logs but returns an empty response)
    the hook silently saves nothing.

    ``set -a`` auto-exports everything sourced from ``~/.attestor/.env`` so the
    ``attestor`` subprocess actually inherits it. A bare ``source`` is NOT
    enough: ``KEY=value`` lines without ``export`` stay shell-local and never
    reach the child process. The file is optional (``[ -f ]``) so installs that
    bind config purely via the MCP ``env`` block still work.

    If a config was passed from outside (``--config`` / ``$ATTESTOR_CONFIG``) we
    bake it into the command *after* sourcing ``.env`` so the explicit choice
    wins over anything the env file sets — the hook resolves the same single
    source of truth as the CLI and the MCP server.
    """
    cfg = os.environ.get("ATTESTOR_CONFIG")
    cfg_prefix = f'ATTESTOR_CONFIG="{cfg}" ' if cfg else ""
    # ~/.local/bin (pipx shim) is not on the hook subprocess's minimal PATH, so
    # a bare `attestor` is "command not found" and the hook silently saves
    # nothing. Prepend it before anything else.
    return (
        'bash -c \'export PATH="$HOME/.local/bin:$PATH"; set -a; '
        '[ -f "$HOME/.attestor/.env" ] && . "$HOME/.attestor/.env"; set +a; '
        f"{cfg_prefix}{binary} hook {subcommand}'"
    )


def _configure_claude_hooks(binary: str):
    """Write Claude Code lifecycle hooks to ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_claude_settings(settings_path)
    hooks = settings.setdefault("hooks", {})

    hook_defs = {
        "SessionStart": {"command": _hook_command(binary, "session-start")},
        "PostToolUse": {"command": _hook_command(binary, "post-tool-use")},
        "Stop": {"command": _hook_command(binary, "stop")},
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
