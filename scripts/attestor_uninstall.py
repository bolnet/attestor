#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Completely uninstall Attestor — package, config, Claude Code wiring, and containers.

NOTE — TEST/REFERENCE ARTIFACT. Attestor is prompt-first: the canonical
uninstall is the prompt at ``commands/uninstall-attestor.md``, which Claude
Code executes directly (scanning + adapting to the machine's actual state).
This script encodes the same procedure for local testing and CI — it is not
the primary path; keep it in sync with the prompt.

DRY-RUN BY DEFAULT: prints exactly what it would do and changes nothing. Pass
``--yes`` to execute. This is the symmetric counterpart to ``attestor init
--install`` / ``setup-claude-code``.

Footprint it removes (the inverse of a full install):
  1. The ``attestor`` package (pipx, else pip).
  2. ``~/.attestor/`` — config.json, attestor.yaml, .env.
  3. The ``attestor`` MCP server entry + Attestor's own hooks from any Claude
     Code settings file (global ~/.claude + the project .claude/.mcp.json).
     Hooks are matched by command content (``attestor hook ...``) so OTHER
     tools' hooks are never touched.
  4. Docker backend containers (``attestor-*``) and their named volumes
     (only with ``--containers``; this DELETES all stored memory).
  5. Stray run artifacts in the current repo (.cc_attestor_probe_store,
     config.json, logs/).

What it can't do for you: ``/plugin uninstall attestor`` (Claude Code plugin
removal is interactive) — it prints the reminder instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# A hook entry belongs to Attestor iff one of its commands invokes the CLI's
# hook subcommand. Matching on this string is safe: it never matches another
# tool's hooks (ECC, continuous-learning, etc.).
_HOOK_MARKER = "attestor hook"
_MCP_KEYS = ("attestor", "memory")  # "memory" = pre-2026-05 server name


def _run(cmd: list[str], *, dry: bool, cwd: str | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return
    subprocess.run(cmd, check=False, cwd=cwd)


def _settings_files() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.json",
        Path.cwd() / ".mcp.json",
    ]
    return [p for p in candidates if p.is_file()]


def _strip_attestor(settings: dict) -> tuple[dict, list[str]]:
    """Return (new_settings, changes). Pure — does not mutate the input."""
    changes: list[str] = []
    new = json.loads(json.dumps(settings))  # deep copy

    mcp = new.get("mcpServers")
    if isinstance(mcp, dict):
        for key in _MCP_KEYS:
            if key in mcp:
                del mcp[key]
                changes.append(f"mcpServers['{key}']")

    hooks = new.get("hooks")
    if isinstance(hooks, dict):
        for event, entries in list(hooks.items()):
            if not isinstance(entries, list):
                continue
            kept = []
            for entry in entries:
                inner = entry.get("hooks", []) if isinstance(entry, dict) else []
                is_attestor = any(
                    _HOOK_MARKER in (h.get("command", "") or "")
                    for h in inner
                    if isinstance(h, dict)
                )
                if is_attestor:
                    changes.append(f"hooks['{event}'] entry")
                else:
                    kept.append(entry)
            if kept:
                hooks[event] = kept
            else:
                del hooks[event]
    return new, changes


def _clean_settings(*, dry: bool) -> None:
    print("\n[3] Claude Code wiring (MCP entry + Attestor hooks)")
    any_change = False
    for path in _settings_files():
        try:
            settings = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skip {path}: {exc}")
            continue
        new, changes = _strip_attestor(settings)
        if not changes:
            print(f"  - {path}: nothing to remove")
            continue
        any_change = True
        print(f"  - {path}: remove {', '.join(changes)}")
        if not dry:
            path.with_suffix(path.suffix + ".bak").write_text(path.read_text())
            path.write_text(json.dumps(new, indent=2) + "\n")
    if not any_change:
        print("  (no Attestor entries found in any settings file)")


def _uninstall_package(*, dry: bool) -> None:
    print("\n[1] Package")
    # Run from $HOME, never the repo: if cwd has an ``attestor/`` subdir, pipx
    # reads "attestor" as a path and refuses ("looks like a path") instead of
    # treating it as the package name.
    neutral = str(Path.home())
    if shutil.which("pipx") and _pipx_has_attestor():
        _run(["pipx", "uninstall", "attestor"], dry=dry, cwd=neutral)
    else:
        _run([sys.executable, "-m", "pip", "uninstall", "-y", "attestor"],
             dry=dry, cwd=neutral)


def _pipx_has_attestor() -> bool:
    try:
        out = subprocess.run(
            ["pipx", "list"], capture_output=True, text=True, check=False
        )
        return "attestor" in out.stdout
    except OSError:
        return False


def _remove_home(*, dry: bool) -> None:
    print("\n[2] ~/.attestor (config + .env)")
    home = Path.home() / ".attestor"
    if home.exists():
        _run(["rm", "-rf", str(home)], dry=dry)
    else:
        print("  (not present)")


def _remove_containers(*, dry: bool) -> None:
    print("\n[4] Docker containers + volumes  (DELETES STORED MEMORY)")
    if not shutil.which("docker"):
        print("  (docker not available)")
        return
    names = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=attestor-"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    if names:
        _run(["docker", "rm", "-f", *names], dry=dry)
    else:
        print("  (no attestor- containers)")
    vols = subprocess.run(
        ["docker", "volume", "ls", "-q", "--filter", "name=attestor"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    if vols:
        _run(["docker", "volume", "rm", *vols], dry=dry)
    else:
        print("  (no attestor volumes)")


def _remove_artifacts(*, dry: bool) -> None:
    print("\n[5] Stray run artifacts in cwd")
    for name in (".cc_attestor_probe_store", "config.json", "logs"):
        p = Path.cwd() / name
        if p.exists():
            _run(["rm", "-rf", str(p)], dry=dry)
        else:
            print(f"  ({name} not present)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Completely uninstall Attestor.")
    ap.add_argument("--yes", action="store_true", help="Actually execute (default: dry-run).")
    ap.add_argument("--containers", action="store_true",
                    help="Also remove Docker containers + volumes (DELETES stored memory).")
    ap.add_argument("--artifacts", action="store_true",
                    help="Also remove stray run artifacts in the current repo.")
    args = ap.parse_args()
    dry = not args.yes

    banner = "DRY-RUN (nothing will change — pass --yes to execute)" if dry else "EXECUTING"
    print("=" * 72)
    print(f"Attestor uninstall — {banner}")
    print("=" * 72)

    _uninstall_package(dry=dry)
    _remove_home(dry=dry)
    _clean_settings(dry=dry)
    if args.containers:
        _remove_containers(dry=dry)
    else:
        print("\n[4] Docker containers/volumes — skipped (pass --containers to remove; DELETES memory)")
    if args.artifacts:
        _remove_artifacts(dry=dry)
    else:
        print("\n[5] Stray repo artifacts — skipped (pass --artifacts to remove)")

    print("\n[6] Claude Code plugin (if installed via /plugin)")
    print("  Run inside Claude Code:  /plugin uninstall attestor")

    print("\n" + "=" * 72)
    if dry:
        print("Dry-run only. Re-run with --yes (and optionally --containers --artifacts).")
    else:
        print("Done. Restart Claude Code so it drops the MCP server + hooks.")
    print("=" * 72)


if __name__ == "__main__":
    main()
