---
description: Install Attestor for Claude Code — interactive configuration
argument-hint: "(no args — will ask)"
allowed-tools: Bash, Read, Write, Edit, Glob
---

# Install Attestor for Claude Code

You are installing **Attestor** (PyPI: `attestor`, import: `attestor`) — a memory layer for agent teams — into the user's Claude Code setup.

This command is **interactive**. The wizard is the single source of truth for every install input — scope, paths, backends, credentials, secrets, hook wiring, verification. **Do NOT silently apply defaults when state already exists. Do NOT collect secrets out-of-band. If the install needs a value, the wizard asks for it.**

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

### Q3. Backend type (top-level Local vs Cloud split)
- `Local` *(Recommended)* — runs on your machine.
- `Cloud-managed` — managed service.

### Q3a. Local stack (only if Q3 = Local)
- `Zero-config embedded` *(Recommended)* — SQLite + ChromaDB + NetworkX. No external services.
- `Local PostgreSQL` — self-hosted Postgres 16 with pgvector + Apache AGE.
- `Local ArangoDB` — self-hosted ArangoDB (doc + vector + graph in one).

### Q3b. Cloud platform (only if Q3 = Cloud-managed) — grouped by provider
- `AWS`
- `Azure`
- `GCP`

### Q3b.i. AWS backend variant (only if Q3b = AWS)
- `ArangoDB Oasis on AWS` *(Recommended)* — single managed service for all three roles.
- `DynamoDB + OpenSearch Serverless + Neptune` — all-AWS native stack.

### Q3b.ii. Azure backend
- `Cosmos DB with DiskANN` — doc + vector + graph in Cosmos.

### Q3b.iii. GCP backend
- `AlloyDB` — Postgres-compatible, pgvector + AGE + ScaNN.

### Q-backend-creds — Credential collection (after backend chosen, before install)

For **every** cloud backend and any local variant that needs credentials (Local Postgres / Local Arango), the wizard collects the required secrets. Required keys per backend:

| Backend | Required env vars |
|---|---|
| Local zero-config | *(none)* |
| Local PostgreSQL | `DATABASE_URL` |
| Local ArangoDB | `ARANGO_URL`, `ARANGO_USER`, `ARANGO_PASSWORD` |
| AWS + ArangoDB Oasis | `ARANGO_URL`, `ARANGO_USER`, `ARANGO_PASSWORD` |
| AWS native | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (or confirm `~/.aws/credentials`) |
| Azure Cosmos | `COSMOS_CONNECTION_STRING` |
| GCP AlloyDB | `DATABASE_URL` |

**Q-backend-creds.1 — How to collect:**
- `Use existing env vars` *(Recommended if already exported)* — wizard detects and confirms.
- `Paste them now` — prompts free-text per key.
- `Skip` — writes config with placeholder; user fills in before restart.

**Q-backend-creds.2 — Where to persist** (only if user pasted values):
- `Gitignored .env file (~/.attestor/.env, chmod 600)` *(Recommended)*
- `Shell profile (~/.zshrc or ~/.bashrc)`
- `This session only` — hold in memory; user must re-export before restart.

### Q4. Embedding provider (only if Q3a = Zero-config embedded; cloud backends bring their own)
- `Local (all-MiniLM-L6-v2, 384D)` *(Recommended)* — no API key, ~90MB download on first use.
- `OpenAI (text-embedding-3-small)` — needs `OPENAI_API_KEY`.
- `OpenRouter` — needs `OPENROUTER_API_KEY`.

### Q-embed-creds (only if OpenAI or OpenRouter chosen)
Same collect/persist sub-questions as Q-backend-creds.

### Q5. Claude Code hooks (multi-select)
- `session-start` *(Recommended for multi-agent)* — every agent boots with shared team context (20K tokens).
- `post-tool-use` *(Recommended for multi-agent)* — auto-capture observations so memory grows across the team.
- `stop` — session summary on exit; useful for planner→executor→reviewer handoffs.
- `none` — MCP only, no hooks.

### Q6. Namespace (only if hooks enabled)
- `Default "user"` *(Recommended)* — single-user laptop.
- `Custom` — follow-up free-text for project/team slug.

### Q7. Default token budget for `recall()`
- `2000`
- `5000`
- `10000` *(Recommended for multi-agent)*
- `Custom`

### Q8. Verification & restart preferences
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
        "ATTESTOR_NAMESPACE": "<NAMESPACE>",
        "ATTESTOR_TOKEN_BUDGET": "<BUDGET>",
        "ATTESTOR_BACKEND": "<local|postgres|arango|aws|azure|gcp>"
      }
    }
  }
}
```

Backend-specific env var **names** go in `env` (the values come from the shell/`.env`, never inline).

---

## Step 7 — Wire hooks (respects Q0b + Q5)

Edit `~/.claude/settings.json` (or `.claude/settings.json` for project scope). Only emit the hooks the user selected in Q5.

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "attestor hook session-start" }] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command", "command": "attestor hook post-tool-use" }] }
    ],
    "Stop": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "attestor hook stop" }] }
    ]
  }
}
```

If Q0b = Replace, strip existing Attestor hook entries before merging.

---

## Step 8 — Verify (if Q8 = Run doctor)

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
