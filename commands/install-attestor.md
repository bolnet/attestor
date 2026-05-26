---
description: Install Attestor for Claude Code — one default profile, zero questions
argument-hint: "(no args — installs the local default, asks nothing)"
allowed-tools: Bash, Read
---

# Install Attestor for Claude Code

You are installing **Attestor** (PyPI: `attestor`, import: `attestor`) — a memory layer for agent teams — into the user's Claude Code setup.

**This install is ZERO-QUESTION and ONE-PERMISSION by design.** There is ONE default profile. You do NOT interview the user, you do NOT call `AskUserQuestion`, and you do NOT pause for per-setting choices. You **announce** the defaults, **run one command**, and **report**. Print everything; ask nothing.

> **Why no questions:** the single command `attestor quickstart` writes the entire config from the bundled local default, brings up the backends, wires the MCP server + hooks, and runs the health check — non-interactively. Every value is fixed and printed below; there is nothing to ask. (If the user explicitly wants a cloud/custom stack, point them to `docs/INSTALL.md` — but the default path asks nothing.)

## The single default profile (fixed — printed, never asked)

| Setting | Default |
|---|---|
| Store path | `~/.attestor` |
| Document | Postgres 16 (local Docker) |
| Vector | Pinecone Local emulator @ `localhost:5080` (local Docker) |
| Graph | Neo4j 5 + GDS (local Docker) |
| Embedder | Ollama `bge-m3` @1024-D — **local, zero cloud key** |
| LLM keys | none required (add/recall work fully local) |
| Passwords | `attestor` (localhost-only dev default); Pinecone key `local` |
| Token budget | 10000 |
| MCP server | written to project `./.mcp.json` as a `.env`-sourcing wrapper |
| Hooks | SessionStart + PostToolUse + Stop, merged into `~/.claude/settings.json` |

This is the canonical **three-role** stack — Postgres + Pinecone + Neo4j — running locally; `quickstart` brings up all three containers (and skips the standalone `attestor-api` container, unused for the Claude Code path).

Memory is automatically isolated per project (git root, else cwd) — no namespace to set.

---

## Step 1 — Ensure the binary (one command)

```bash
command -v attestor >/dev/null 2>&1 || pipx install attestor || python3 -m pip install --user attestor
```

(If already present, optionally `pipx upgrade attestor`.) Confirm with `attestor --help | head -3`.

---

## Step 2 — Run the zero-question installer (the one command)

```bash
attestor quickstart
```

That single command does **all** of it, non-interactively, printing each step:

1. writes `~/.attestor/attestor.yaml` (stack) + `~/.attestor/config.toml` (connection, `$PGPASSWORD`/`$NEO4J_PASSWORD` env refs, `v4=true`),
2. writes `~/.attestor/.env` (local passwords + the Ollama embedder route) — idempotent, `chmod 600`, never clobbers existing values,
3. brings up the local Docker backends (Postgres + Neo4j + Pinecone Local) and waits for PG/Neo4j health,
4. wires the Claude Code MCP server (`./.mcp.json`) + the 3 lifecycle hooks (`~/.claude/settings.json`), both `.env`-sourcing — it does **not** trust the plugin's bundled `.mcp.json` (which hard-codes a dev path),
5. runs `attestor doctor` and prints the resolved config.

It **bypasses** `init`'s fresh-store gate by force-writing `config.toml`, so a background MCP server's tuning-only `config.json` cannot block the connection config.

**Do not second-guess it with questions.** If a sub-step fails (e.g. Docker not running, Ollama/`bge-m3` not pulled), the command prints the exact remedy — relay that, don't start an interview.

Useful flags (only if the user explicitly asks): `--no-docker`, `--no-wire`, `--no-verify`, or a custom store path as the positional arg.

---

## Step 3 — Report (≤6 lines) and remind to restart

After `attestor quickstart` returns, tell the user:
- ✅ Installed `attestor`, wrote `~/.attestor/{config.toml,attestor.yaml,.env}`, brought up Postgres + Neo4j.
- ✅ MCP server `attestor` → `./.mcp.json`; hooks → `~/.claude/settings.json`.
- The `doctor` result (note: **Vector Store may show "Not initialized"** — that's cosmetic; pgvector serves the vector role and recall returns `source: vector`).
- To prove it: after restart, `/mcp` shows `attestor` (8 tools); ask it to remember something, then recall it.
- **You must restart Claude Code** so the MCP server + hooks attach (they load at session start).
- To change ANY setting later: edit `~/.attestor/attestor.yaml` (the single source of truth) and re-run `attestor quickstart`.

---

## Prerequisites (state them, don't ask about them)

The fully-local default needs **Docker** (for Postgres + Neo4j + Pinecone Local) and **Ollama** serving `bge-m3` (`ollama pull bge-m3`). `attestor quickstart` runs a preflight that scans these ports/tools and prints their state first. If something's missing, it still writes all config and prints what to start — report that and let the user start it, then re-run `attestor quickstart` (it's idempotent).
