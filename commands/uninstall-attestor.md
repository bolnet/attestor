---
description: Completely uninstall Attestor from Claude Code — reverse every install surface
argument-hint: "(no args — will scan, then confirm destructive steps)"
allowed-tools: Bash, Read, Write, Edit, Glob
---

# Uninstall Attestor from Claude Code

You are completely uninstalling **Attestor** (PyPI: `attestor`) from this machine. This prompt is the **source of truth** — drive the whole uninstall yourself by following it. (`scripts/attestor_uninstall.py` exists only as a test/reference of the same procedure; you don't need it.)

A full install touches **six surfaces** — reverse each, in this order. Scan first, report, then act. Destructive steps (data deletion) need explicit confirmation; config/wiring removal does not.

Rules:
- **Never** remove another tool's hooks. Match Attestor hooks by command content `attestor hook` only.
- **Back up** every JSON settings file to `*.bak` before editing; read → parse → mutate in memory → write atomically. Never `jq -e` mid-edit.
- Confirm before deleting **data** (Docker volumes, `~/.attestor`). Wiring removal (MCP entry, hooks, package) is safe to do without a prompt.
- If a JSON file won't parse, stop and show the user — don't guess-repair.

---

## Step 1 — Scan all six surfaces (report before changing anything)

```bash
echo "[1] package:"   ; (command -v attestor && pipx list 2>/dev/null | grep attestor) || echo "  none"
echo "[2] ~/.attestor:"; ls -la ~/.attestor 2>/dev/null || echo "  none"
echo "[3] wiring:"     ; for f in ~/.claude/settings.json ~/.claude/.mcp.json ~/.claude.json ./.claude/settings.json ./.mcp.json; do echo "  $f"; grep -o '"attestor"\|attestor hook' "$f" 2>/dev/null | sort -u | sed 's/^/    /'; done
echo "[4] docker:"     ; docker ps -a --filter name=attestor- --format '  {{.Names}}' 2>/dev/null; docker volume ls -q --filter name=attestor 2>/dev/null | sed 's/^/  vol /'
echo "[5] artifacts:"  ; ls -d .cc_attestor_probe_store config.json logs 2>/dev/null | sed 's/^/  /' || echo "  none"
echo "[6] plugin:"     ; grep -l "bolnet/attestor" ~/.claude.json 2>/dev/null && echo "  plugin ref present" || echo "  none"
```

If nothing is found, stop: "Attestor is not installed — nothing to remove."

---

## Step 2 — [1] Uninstall the package

```bash
( cd "$HOME" && pipx uninstall attestor ) || pip uninstall -y attestor
```

**Run it from `$HOME`, not the repo.** If cwd contains an `attestor/` directory (the source tree), pipx reads `attestor` as a path and fails with *"'attestor' looks like a path"*. `cd "$HOME"` first.

---

## Step 3 — [3] Remove the MCP entry + Attestor's hooks (every settings file)

For each of `~/.claude/settings.json`, `~/.claude/.mcp.json`, `~/.claude.json`, `./.claude/settings.json`, `./.mcp.json` that exists:

1. Back up to `<file>.bak`.
2. `mcpServers`: delete keys `attestor` **and** `memory` (the pre-2026-05 name). Preserve all other servers; leave `mcpServers` as `{}` if it empties.
3. `hooks`: in every event, drop only entries whose inner `hooks[].command` contains `attestor hook`. Keep every other tool's entry (ECC, continuous-learning, etc. — they share the same event arrays). Drop an event key only if it becomes empty.

The **global** `~/.claude/settings.json` usually has only *other* tools' hooks — expect "nothing to remove" there; Attestor's hooks normally live in the **project** `./.claude/settings.json`.

---

## Step 4 — [2] Remove `~/.attestor/` (config + .env)

This holds `config.json`, `attestor.yaml`, `.env` — config only, no memory data. Safe to remove without a data-loss prompt:

```bash
rm -rf ~/.attestor
```

---

## Step 5 — [4] Docker backends + volumes — **confirm first (DELETES ALL MEMORY)**

Ask via `AskUserQuestion`: "Remove the Attestor Docker containers and volumes? This permanently deletes all stored memories, vectors, and graph state."

On yes:

```bash
docker rm -f $(docker ps -aq --filter name=attestor-) 2>/dev/null
docker volume rm $(docker volume ls -q --filter name=attestor) 2>/dev/null
```

(These also clean up stale `arango_*` / duplicate volumes from older setups.) On no: leave them and say so.

---

## Step 6 — [5] Stray repo artifacts (optional, confirm)

`AgentMemory` run from the repo root leaves `.cc_attestor_probe_store/`, a root `config.json`, and `logs/`. Offer to remove them: `rm -rf .cc_attestor_probe_store config.json logs`.

---

## Step 7 — [6] Plugin

If Step 1 found a plugin ref, tell the user to run (you can't do this for them): **`/plugin uninstall attestor`** inside Claude Code.

---

## Step 8 — Verify + report

```bash
command -v attestor || echo "binary gone"
pipx list 2>/dev/null | grep -c attestor
[ -e ~/.attestor ] && echo "HOME present" || echo "HOME gone"
for f in ~/.claude/settings.json ~/.claude/.mcp.json ~/.claude.json ./.claude/settings.json ./.mcp.json; do grep -c "attestor hook\|\"attestor\"" "$f" 2>/dev/null; done
docker ps -aq --filter name=attestor- | wc -l
```

Report a ≤8-line summary of what was **removed** vs **preserved** (and which `*.bak` backups were left). Remind the user to **restart Claude Code** so it drops the now-orphaned MCP server process + hooks.
