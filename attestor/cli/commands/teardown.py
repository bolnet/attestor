# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``attestor teardown`` — zero-question reverse of ``attestor quickstart``.

Removes exactly what ``quickstart`` created, in one command, printing every
step:

  1. the local Docker backends (postgres + neo4j + pinecone),
  2. the store config dir (``~/.attestor``),
  3. the MCP server entry from the project ``./.mcp.json``,
  4. the Attestor lifecycle hooks from ``~/.claude/settings.json``
     (content-matched on ``attestor hook`` — never touches other tools' hooks).

**Data-safe by default:** Docker named volumes (your actual memories) are KEPT,
so a later ``attestor quickstart`` reconnects to the same data. Pass ``--purge``
to also delete the volumes (wipes all stored memories). ``--dry-run`` previews
without changing anything.

It does NOT remove the pipx package or the Claude Code plugin (separate
surfaces, like ``quickstart`` doesn't install them) — it prints those commands
for you to run.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from attestor.cli.commands.quickstart import (
    DEFAULT_STORE,
    _compose_file,
    _docker_available,
)

if TYPE_CHECKING:
    import argparse

HOOK_MARKER = "attestor hook"  # content-match: only Attestor's own hooks
MCP_KEYS = ("attestor", "memory")  # current + pre-2026-05 server names


def _compose_down(*, purge: bool, dry_run: bool) -> None:
    compose = _compose_file()
    cmd = ["docker", "compose", "-f", str(compose), "down"]
    if purge:
        cmd.append("--volumes")
        label = "remove containers + VOLUMES (wipes memories)"
    else:
        label = "remove containers (keep data volumes)"
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}   # {label}")
        return
    if not _docker_available():
        print(f"  Docker not available — run yourself: {' '.join(cmd)}")
        return
    print(f"  {' '.join(cmd)}   # {label}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  docker compose down failed: {exc}")
        return
    if proc.returncode != 0:
        print(f"  docker compose down returned {proc.returncode}: {proc.stderr.strip()[-400:]}")


def _remove_store(store: Path, *, purge: bool, dry_run: bool) -> None:
    """Remove the store config dir. Always removable (regenerable by quickstart)."""
    if not store.exists():
        print(f"  {store} — not present")
        return
    if dry_run:
        print(f"  [dry-run] would remove {store} (config; data lives in Docker volumes)")
        return
    import shutil

    shutil.rmtree(store)
    print(f"  removed {store}")


def _remove_mcp_entry(*, dry_run: bool) -> None:
    mcp_path = Path.cwd() / ".mcp.json"
    if not mcp_path.exists():
        print(f"  {mcp_path} — not present")
        return
    try:
        data = json.loads(mcp_path.read_text())
    except json.JSONDecodeError:
        print(f"  {mcp_path} — unparseable, leaving untouched")
        return
    servers = data.get("mcpServers", {})
    removed = [k for k in MCP_KEYS if k in servers]
    if not removed:
        print(f"  {mcp_path} — no attestor entry")
        return
    if dry_run:
        print(f"  [dry-run] would remove mcpServers {removed} from {mcp_path}")
        return
    for k in removed:
        servers.pop(k, None)
    mcp_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  removed mcpServers {removed} from {mcp_path}")


def _remove_hooks(*, dry_run: bool) -> None:
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print(f"  {settings_path} — not present")
        return
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        print(f"  {settings_path} — unparseable, leaving untouched")
        return
    hooks = settings.get("hooks", {})
    removed = 0
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        kept_entries = []
        for entry in entries:
            inner = entry.get("hooks", []) if isinstance(entry, dict) else []
            kept_inner = [h for h in inner if HOOK_MARKER not in str(h.get("command", ""))]
            removed += len(inner) - len(kept_inner)
            if kept_inner:
                kept_entries.append({**entry, "hooks": kept_inner})
            # entries whose only hooks were attestor's are dropped entirely
        if kept_entries:
            hooks[event] = kept_entries
        else:
            hooks.pop(event, None)
    if removed == 0:
        print(f"  {settings_path} — no attestor hooks")
        return
    if dry_run:
        print(f"  [dry-run] would remove {removed} attestor hook(s) from {settings_path}")
        return
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"  removed {removed} attestor hook(s) from {settings_path}")


def _cmd_teardown(args: argparse.Namespace) -> None:
    store = Path(getattr(args, "path", None) or DEFAULT_STORE).expanduser()
    purge = bool(getattr(args, "purge", False))
    dry_run = bool(getattr(args, "dry_run", False))

    if dry_run:
        mode = "DRY-RUN (no changes)"
    elif purge:
        mode = "PURGE (also deletes stored memories)"
    else:
        mode = "default (keeps data volumes)"
    print("Attestor Teardown — reverse of `attestor quickstart` (zero questions)")
    print("=" * 66)
    print(f"  store ............... {store}")
    print(f"  mode ................ {mode}")
    print()

    print("[1/4] Docker backends (postgres + neo4j + pinecone)")
    _compose_down(purge=purge, dry_run=dry_run)

    print("\n[2/4] Store config (~/.attestor)")
    _remove_store(store, purge=purge, dry_run=dry_run)

    print("\n[3/4] MCP server entry (./.mcp.json)")
    _remove_mcp_entry(dry_run=dry_run)

    print("\n[4/4] Lifecycle hooks (~/.claude/settings.json)")
    _remove_hooks(dry_run=dry_run)

    print("\nDone. Not removed (separate surfaces — run yourself if you want them gone):")
    print("  • package:  cd ~ && pipx uninstall attestor   (run from $HOME, not the repo)")
    print("  • plugin:   /plugin uninstall attestor   (in Claude Code)")
    if not purge:
        print("\nData volumes KEPT — `attestor quickstart` reconnects to the same memories.")
        print("To wipe everything including stored memories, re-run with --purge.")
