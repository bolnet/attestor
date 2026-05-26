---
description: Install Attestor for Claude Code — interactive configuration
argument-hint: "(no args — will ask)"
allowed-tools: Bash, Read, Write, Edit, Glob
---

# Install Attestor for Claude Code

You are installing **Attestor** (PyPI: `attestor`, import: `attestor`) — a memory layer for agent teams — into the user's Claude Code setup.

**This prompt is the source of truth — drive the whole install yourself by following it.** Attestor is prompt-first for Claude Code + local use: you do the install/setup by reading and acting on this prompt, not by deferring to a script. The Python helpers (`attestor setup-claude-code` / `scripts/`) exist as a test/reference of the same steps; prefer doing each step here directly so you can adapt to the machine's actual state.

This command is **interactive**. The wizard is the single source of truth for every install input — scope, paths, backends, credentials, secrets, hook wiring, verification. **Do NOT silently apply defaults when state already exists. Do NOT collect secrets out-of-band. If the install needs a value, the wizard asks for it.**

> **Plugin mode:** When Attestor is installed as a Claude Code plugin (`/plugin install attestor`), the MCP server (`.mcp.json`) and the session-start / post-tool-use / stop hooks (`hooks/hooks.json`) are **auto-wired by the plugin** — Claude Code v2.1+ loads them by convention. In that case this wizard only needs to: (a) `pip`/`pipx install attestor` so the `attestor` binary is on PATH, (b) collect backend connection details (Postgres + Pinecone + Neo4j; local Docker or cloud), (c) collect the embedding provider, and (d) run `attestor doctor`. **Skip the `settings.json` MCP + hook writing steps in plugin mode** — they only apply to manual, non-plugin installs and would duplicate the plugin's wiring. Memory is automatically isolated per project: each working directory (git root, else cwd) becomes its own RLS tenant, so projects never share memory.

Rules:
- Ask **one** question at a time via `AskUserQuestion` — never batch.
- Ask Q0 (pre-existing state) questions first — only ask the ones whose preconditions were detected in Step 1.
- Skip sub-questions that don't apply to the chosen branch (e.g. skip Q-embed-provider when cloud backend selected).
- Never write secrets into config files — always reference env vars. The wizard asks **where** to persist each secret (gitignored `.env` / shell profile / this-session-only).
- Merge JSON configs; never clobber existing `mcpServers` or `hooks` entries without an explicit user choice.
- If a step fails, stop and report — do not retry silently.

---

## Step 1 — Detect current state (before asking anything)

Run these checks in parallel:

```bash
command -v attestor || echo "NOT_INSTALLED"
command -v pipx || echo "NO_PIPX"
python3 --version
ls ~/.attestor >/dev/null 2>&1 && echo "STORE_EXISTS" || echo "STORE_NEW"
ls ~/.claude/.mcp.json >/dev/null 2>&1 && echo "GLOBAL_MCP_EXISTS" || echo "GLOBAL_MCP_NONE"
ls .mcp.json >/dev/null 2>&1 && echo "PROJECT_MCP_EXISTS" || echo "PROJECT_MCP_NONE"
grep -l "attestor" ~/.claude/settings.json 2>/dev/null && echo "HOOKS_WIRED" || echo "HOOKS_NONE"
```

Report a one-line summary and move to Step 2.

---

## Step 2 — Interview (one question per turn, use AskUserQuestion)

### Q0 — Pre-existing state handling (ask ONLY the ones that apply)

**Q0a. Existing MCP entry** (if any `mcpServers` entry named `memory` or `attestor` already exists)
- `Update in place` *(Recommended)* — overwrite with wizard settings.
- `Keep as-is` — skip MCP changes.
- `Add alongside` — keep existing, add a second entry under a different key.

**Q0b. Existing hooks** (if `settings.json` hooks already reference `attestor`)
- `Keep as-is` *(Recommended)*
- `Replace with wizard selection`
- `Remove all Attestor hooks`

**Q0c. Existing store** (if `STORE_PATH` already exists)
- `Reuse` *(Recommended)* — keep memories and settings, apply new config only.
- `Pick a new path` — create a fresh store elsewhere.
- `Reset (destructive)` — wipe and start clean. Requires explicit re-confirm.

### Q1. Scope
- `Global (~/.claude/.mcp.json)` *(Recommended)*
- `Project (./.mcp.json)`

### Q2. Store location
- `Default (~/.attestor/)` *(Recommended)*
- `Custom path` — follow-up free-text for absolute path.

### Q3. Backend topology (Local vs Cloud — the same three roles either way)

The canonical stack is **Postgres (document) + Pinecone (vector) + Neo4j (graph)**; only the connection details differ. (The single-DB pgvector bundle and the Arango / AWS-native / Cosmos / AlloyDB backends were removed on 2026-05-02 — do not offer them.)

- `Local Docker` *(Recommended)* — three containers on this machine: `attestor-postgres` (`pgvector/pgvector:pg16`), `attestor-pinecone` (Pinecone Local emulator), `attestor-neo4j` (`neo4j:5.24-community` + GDS). Full walkthrough: `docs/LOCAL_DOCKER_SETUP.md`.
- `Cloud-managed` — managed Postgres (Neon / RDS / Cloud SQL / AlloyDB-as-PG), Pinecone Cloud, Neo4j AuraDB.

### Q-backend-creds — Credentials (collect after Q3, before install)

Non-secret settings live in `configs/attestor.yaml`; only secrets vary by environment. Required keys:

| Topology | Required env vars |
|---|---|
| Local Docker | `PINECONE_API_KEY` (Pinecone Inference embedder — cloud-only), `NEO4J_PASSWORD`, `OPENROUTER_API_KEY` (LLM calls). Pinecone **Local** + Postgres need no key. |
| Cloud-managed | `POSTGRES_URL`, `NEO4J_URI`, `NEO4J_PASSWORD`, `PINECONE_API_KEY`, `OPENROUTER_API_KEY` |

If you change the embedder in `configs/attestor.yaml` (Q4), swap `PINECONE_API_KEY` for that provider's key.

**Q-backend-creds.1 — How to collect:**
- `Use existing env vars` *(Recommended if already exported)* — detect and confirm.
- `Paste them now` — prompt free-text per key.
- `Skip` — write config with placeholder; user fills in before restart.

**Q-backend-creds.2 — Where to persist** (only if user pasted values):
- `Gitignored .env file (~/.attestor/.env, chmod 600)` *(Recommended)*
- `Shell profile (~/.zshrc or ~/.bashrc)`
- `This session only` — hold in memory; user must re-export before restart.

### Q4. Embedder (the `configs/attestor.yaml` default — only change if asked)

Whichever is chosen, set it under `stack.embedder` in `configs/attestor.yaml` (the single source of truth) and put the key in `.env`:
- `Pinecone Inference llama-text-embed-v2 (1024-D)` *(Recommended — the default)* — needs `PINECONE_API_KEY` (cloud-only; pairs with the Pinecone vector role).
- `Voyage voyage-4 (1024-D)` — needs `VOYAGE_API_KEY`.
- `OpenAI text-embedding-3-* (→1024-D)` — needs `OPENAI_API_KEY`.
- `Ollama bge-m3 (local, free)` — provider `openai` + `OPENAI_BASE_URL` pointed at the local Ollama endpoint.

### Q5. Claude Code hooks (multi-select)
- `session-start` *(Recommended)* — inject this project's relevant memories at session start.
- `post-tool-use` *(Recommended)* — auto-capture file changes + commands into this project's memory.
- `stop` — write a project-scoped session summary on exit.
- `none` — MCP only, no hooks.

> No namespace to configure — memory is **automatically isolated per project**: each working directory (git root, else cwd) is its own RLS tenant, so projects never share memory.

### Q6. Default token budget for `recall()`
- `2000`
- `5000`
- `10000` *(Recommended for multi-agent)*
- `Custom`

### Q7. Verification & restart preferences
- `Run doctor + print MCP config automatically` *(Recommended)*
- `Skip verification`

Always end with a printed restart reminder. No auto-restart.

---

## Step 3 — Install the package

If attestor is not on PATH:
```bash
pipx install attestor || python3 -m pip install --user attestor
```
If already installed:
```bash
pipx upgrade attestor || python3 -m pip install --user -U attestor
```
Confirm with `attestor --help | head -5`.

---

## Step 4 — Provision the store (respects Q0c answer)

- If Q0c = Reuse → skip creation, just run doctor.
- If Q0c = Pick new path → `mkdir -p "$STORE_PATH"`.
- If Q0c = Reset → re-confirm, then `rm -rf "$STORE_PATH"` and recreate.

Then: `attestor doctor "$STORE_PATH"` must report OK for Document / Vector / Graph / Retrieval.

---

## Step 5 — Persist secrets (respects Q-backend-creds.2 / Q-embed-creds)

- **Gitignored `.env`**: write `KEY=value` lines to `~/.attestor/.env`, `chmod 600`, ensure `.env` is in `.gitignore`.
- **Shell profile**: append `export KEY=value` to `~/.zshrc` (or `~/.bashrc`). Tell the user to `source` it.
- **Session-only**: hold in memory for the install run; print a reminder to re-export before restart.

Never inline secrets into `~/.claude/.mcp.json` or `settings.json`.

---

## Step 6 — Write MCP config (respects Q0a + Q1 + Q3)

Merge into the chosen MCP config file. Read first, then update the chosen `mcpServers` entry:

```json
{
  "mcpServers": {
    "attestor": {
      "command": "attestor",
      "args": ["mcp", "--path", "<STORE_PATH>"],
      "env": {
        "ATTESTOR_CONFIG": "<ABSOLUTE_PATH_TO_attestor.yaml>"
      }
    }
  }
}
```

`ATTESTOR_CONFIG` is the single source of truth — pin it to the **absolute** path of the `attestor.yaml` this install uses, so the long-running MCP server resolves the same config the CLI did (not a cwd-relative guess). Use the resolved path that `attestor setup-claude-code` prints under "Config file in use". Backend secrets are never inlined — they come from the shell / `~/.attestor/.env`.

---

## Step 7 — Wire hooks (respects Q0b + Q5)

Edit `~/.claude/settings.json` (or `.claude/settings.json` for project scope). Only emit the hooks the user selected in Q5.

**Critical — the hook command MUST load the user env before calling `attestor`.** Claude Code spawns hooks without the interactive shell's environment, so a bare `attestor hook ...` runs with no `ATTESTOR_CONFIG` / provider keys, the embedder fails to init, and the hook silently saves nothing. Wrap every hook in `set -a; source ~/.attestor/.env; set +a` — and `set -a` is required, because un-`export`ed `KEY=value` lines in `.env` otherwise stay shell-local and never reach the subprocess:

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook session-start'" }] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command", "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook post-tool-use'" }] }
    ],
    "Stop": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook stop'" }] }
    ]
  }
}
```

If a non-default config is in play, bake it in too: `… set +a; ATTESTOR_CONFIG="<abs path>" attestor hook stop`. If Q0b = Replace, strip existing Attestor hook entries (any command containing `attestor hook`) before merging.

---

## Step 8 — Verify (if Q7 = Run doctor)

```bash
attestor doctor "$STORE_PATH" && echo "--- MCP config ---" && cat "$MCP_CONFIG_FILE"
```

Then tell the user (≤6 lines):
- What was installed and where
- Which MCP config file was touched
- Which hooks were wired
- Which secrets were written and where
- The sanity-check command: `attestor doctor ~/.attestor`
- That they must **restart Claude Code** for the MCP server to attach
