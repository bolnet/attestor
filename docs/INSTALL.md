# Attestor Install Guide

A step-by-step guide to installing and verifying Attestor across different topologies. Each chapter is a self-contained setup you can follow from scratch.

**Chapters**

| # | Topology | Backend |
|---|----------|---------|
| [00](#chapter-00--install-via-claude-code-recommended) | **Install via Claude Code** (one prompt · cold start) | Postgres + Pinecone + Neo4j |
| [01](#chapter-01--local-stack-with-docker-compose) | Local stack (Docker Compose) | Postgres + Pinecone + Neo4j |
| [02](#chapter-02--sidecar-rest-api) | Sidecar REST API | Same stack, exposed over HTTP |
| [03](#chapter-03--cloud-managed) | Cloud managed | Managed Postgres + Pinecone + Neo4j |

---

## Chapter 00 — Install via Claude Code (recommended)

**One install path — the guided wizard** ([`../commands/install-attestor.md`](../commands/install-attestor.md)). Whatever you type, Claude reads this repo and runs the wizard end-to-end: it **scans your machine first, looks up current docs via Context7**, then installs the `attestor` package, brings up the three backend containers, wires the MCP server + hooks, and verifies. It assumes you start with **nothing installed**. Four ways to launch it — type any of:

| # | Way | Type this |
|---|-----|-----------|
| 1 | Plugin (recommended) | `/plugin install attestor` |
| 2 | Command | `/install-attestor` |
| 3 | Repo URL | `github.com/bolnet/attestor` |
| 4 | Natural language | `install attestor` |

First-time plugin use needs `/plugin marketplace add bolnet/attestor` once. The detailed cold-start prompt below is what the wizard runs — paste it directly only if you want to drive a bare session by hand.

> Every chapter targets the same **canonical stack: Postgres (document) + Pinecone (vector) + Neo4j (graph)**, with the embedder, models, and retrieval budget coming from the single source of truth, [`configs/attestor.yaml`](../configs/attestor.yaml). Chapter 00 is the fastest path; 01–03 are the manual local / sidecar / cloud setups.

### The three backends — one clearly-named Docker instance per storage role

Each storage role is its own container, all `attestor-`prefixed and labeled by type:

| Container | Type | Storage role | Image | Ports |
|-----------|------|--------------|-------|-------|
| `attestor-postgres` | **Postgres 16 + pgvector** | **Document** — source of truth (content, tags, entity, ts, provenance, confidence) | `pgvector/pgvector:pg16` | `5432` |
| `attestor-pinecone` | **Pinecone Local** | **Vector** — dense embeddings, per-namespace cosine search | `ghcr.io/pinecone-io/pinecone-local:latest` | `5080-5090` |
| `attestor-neo4j` | **Neo4j 5 + GDS** | **Graph** — entity nodes + typed edges, PageRank / BFS | `neo4j:5.24-community` | `7474`, `7687` |

### The install prompt — paste into a fresh Claude Code session

```text
Install Attestor as my agent-memory layer and wire it natively into Claude Code.
Assume a COLD machine — I may have nothing installed. Work in this order and ask
me only when you hit a real decision or need a secret.

PHASE 1 — SCAN (install nothing yet)
Detect and show me a table of present vs missing:
- OS + arch; Python (need >=3.10); pip / pipx; Docker + Docker Compose (running?).
- Existing Attestor: `attestor --version`, ~/.attestor, an MCP entry in
  ~/.claude or ./.mcp.json, attestor hooks in settings.json.
- Running backend containers (`docker ps`) and listeners on 5432 / 5080 / 7687.
Then tell me what you'll INSTALL, REUSE, and SKIP before doing anything.

PHASE 2 — LOOK UP CURRENT DOCS WITH CONTEXT7 (before every install step)
Before you install or run ANY tool, fetch its current docs via the Context7 MCP
(resolve-library-id -> query-docs): pipx, pgvector, Pinecone Local, Neo4j GDS,
the `attestor` PyPI package, and the Claude Code plugin / MCP / hooks format.
Use the commands and versions Context7 returns — not ones from memory. If
Context7 is unavailable, tell me and pause.

PHASE 3 — BACKENDS: three clearly-named Docker instances (attestor- prefix), one per role
Start exactly these and confirm each reports healthy:
- attestor-postgres  Postgres 16 + pgvector   DOCUMENT store   :5432         pgvector/pgvector:pg16
- attestor-pinecone  Pinecone Local           VECTOR store     :5080-5090    ghcr.io/pinecone-io/pinecone-local:latest
- attestor-neo4j     Neo4j 5 + GDS            GRAPH store      :7474/:7687   neo4j:5.24-community (NEO4J_PLUGINS=["graph-data-science"])
Use a single password (default `attestor`) and put every secret in a gitignored
.env — never hardcode. Persist data in named volumes.

PHASE 4 — INSTALL THE ATTESTOR PACKAGE
`pipx install attestor` (fallback `pip install --user attestor`). Confirm
`attestor --version`. The `attestor` binary must be on PATH — the plugin's hooks
and MCP server call it directly.

PHASE 5 — WIRE INTO CLAUDE CODE (prefer the plugin)
Ask my scope: global (~/.claude) or this project only. Then:
- Plugin (recommended): `/plugin marketplace add bolnet/attestor` then
  `/plugin install attestor` — this auto-wires the MCP server (.mcp.json) and the
  SessionStart / PostToolUse / Stop hooks by convention.
- If I decline the plugin: merge the attestor MCP server into .mcp.json and the
  three hooks into settings.json yourself, without clobbering existing entries.
Ask my embedding provider (Pinecone Inference default, else Voyage / OpenAI /
Ollama) and store its API key in .env.

PHASE 6 — VERIFY
Run `attestor doctor <store-path>` (expect Document / Vector / Graph / Retrieval
all healthy) and call the `memory_health` MCP tool. Then prove per-project
isolation: add a memory here, and confirm a different project directory does NOT
see it. Show me the results.

NOTES
- Memory is automatically isolated per project — each git-root (else cwd) is its
  own tenant. There is no namespace to configure.
- If any phase fails, STOP and tell me exactly what failed. Do not continue silently.
```

After it finishes you can use the `memory_*` tools immediately, and every project you open gets its own hard-isolated memory automatically.

> Hooks load the user environment before calling `attestor`. The wired hook command is
> `bash -c 'set -a; [ -f "$HOME/.attestor/.env" ] && . "$HOME/.attestor/.env"; set +a; attestor hook <event>'`.
> The `set -a` matters: a bare `source` of a `.env` with un-exported `KEY=value` lines leaves
> those vars shell-local, so the `attestor` subprocess never sees `ATTESTOR_CONFIG` / provider keys
> and the hook silently saves nothing.

---

## Chapter 01 — Local stack with Docker Compose

Attestor's canonical stack is three services: **Postgres** (document role, source of truth), **Pinecone Local** (vector role, the free `:5080` Docker emulator), and **Neo4j + GDS** (graph role). The bundled Compose stack in `attestor/infra/local/` brings all three up on a laptop.

> The detailed, copy-paste-ready walkthrough — including the four health checks — lives in
> **[`docs/LOCAL_DOCKER_SETUP.md`](LOCAL_DOCKER_SETUP.md)** (and its agent-driven variant
> **[`docs/CLAUDE_LOCAL_SETUP_PROMPT.md`](CLAUDE_LOCAL_SETUP_PROMPT.md)**). This chapter is the short version.

### Prerequisites

- Python 3.10 or later; `pip`, `pipx`, or `poetry`
- Docker + Docker Compose (v2). The Pinecone Local image is `linux/amd64`; Apple Silicon runs it under emulation automatically.
- A repo-root `.env` (gitignored) with the keys the configured stack needs. With the default `configs/attestor.yaml` embedder (Pinecone Inference `llama-text-embed-v2`, 1024-D) that is:

  ```bash
  PINECONE_API_KEY=...      # Pinecone Inference embedder — cloud-only, app.pinecone.io (matches configs/attestor.yaml)
  NEO4J_PASSWORD=attestor
  OPENROUTER_API_KEY=...    # answer/judge model calls (optional for plain add/recall)
  ```

  The vector *store* can stay on Pinecone Local (no key); the Inference *embedder* is cloud-only, hence `PINECONE_API_KEY`. Swap the embedder by editing `configs/attestor.yaml` (Voyage / OpenAI / Ollama) — the `.env` keys follow whatever provider you choose.

### Step 1 — Start the three backends

```bash
cd attestor/infra/local
cp .env.example .env            # fill in the keys above
docker compose up -d postgres neo4j pinecone
```

This brings up three containers:

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `attestor_postgres_document_db` | `attestor/db-postgres:16` (pgvector) | `5432` | Document |
| `attestor_pinecone_vector_db` | `ghcr.io/pinecone-io/pinecone-local:latest` | `5080-5090` | Vector |
| `attestor_neo4j_graph_db` | `neo4j:5.24-community` (+ GDS plugin) | `7474`, `7687` | Graph |

Wait for all three to report healthy:

```bash
docker ps --filter name=attestor- --format '{{.Names}}\t{{.Status}}'
```

### Step 2 — Install the CLI

```bash
pipx install attestor      # isolated CLI (recommended)
# or: pip install attestor / poetry add attestor
attestor --help
```

### Step 3 — Point Attestor at the stack

Connection details come from `configs/attestor.yaml` (the source of truth). For a one-off override, env vars win:

```bash
export POSTGRES_URL="postgresql://postgres:attestor@localhost:5432/attestor"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="attestor"
# Pinecone Local needs no key; Pinecone Cloud uses PINECONE_API_KEY.
```

### Step 4 — Write your first memory

```python
from attestor import AgentMemory

mem = AgentMemory()   # reads configs/attestor.yaml / env

mem.add(
    "The order service uses event sourcing with a 30-day retention policy",
    entity="order-service",
    tags=["architecture", "decision"],
)
```

Every memory is persisted across all three roles:
- **Document** (Postgres) — content, tags, entity, timestamp, confidence, provenance
- **Vector** (Pinecone) — the dense embedding for cosine search
- **Graph** (Neo4j) — entity node `order-service` + typed edges

### Step 5 — Recall

```python
results = mem.recall("how is the order service structured?", budget=2000)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

The deterministic **6-step retrieval pipeline** runs (no LLM in the hot path):
1. **Vector top-K** — Pinecone cosine on the query embedding (optional HyDE v2 lane)
2. **BM25 lane** (optional) — Postgres FTS
3. **RRF blend** — reciprocal-rank fusion (k=60) merges vector + BM25
4. **Graph narrow** — Neo4j BFS depth=2 affinity bonus + synthetic-triple injection
5. **MMR diversity** (λ=0.7) + confidence decay
6. **Token-budget pack** — greedy fit under the recall budget

### Step 6 — Verify

```bash
attestor doctor <store-path>
```

Expect Document (Postgres), Vector (Pinecone), Graph (Neo4j), and the Retrieval pipeline all healthy. If the vector or graph role fails, retrieval degrades gracefully — the document store is the only hard dependency.

```bash
attestor add "API rate limit is 1000 req/min" --tags api,limits
attestor recall "what are the rate limits?"
attestor stats
```

### Degradation

Attestor's retrieval pipeline tolerates partial outages:

- **Vector down** — falls back to tag match + graph expansion
- **Graph down** — falls back to tag match + vector search
- **Document store** is the only hard dependency

Non-fatal errors in the vector or graph layers are caught and logged; the document path never breaks.

### Claude Code integration

The fastest path is Chapter 00. To wire it manually, add the MCP server to `.mcp.json` (project) or `~/.claude/settings.json` (global):

```json
{
  "mcpServers": {
    "attestor": {
      "command": "attestor",
      "args": ["mcp"],
      "env": {
        "ATTESTOR_CONFIG": "/absolute/path/to/configs/attestor.yaml"
      }
    }
  }
}
```

The MCP server inherits the `env` block above. The lifecycle hooks (SessionStart / PostToolUse / Stop) run as separate subprocesses that do **not** inherit your interactive shell, so they load `~/.attestor/.env` themselves:

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command",
      "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook session-start'" }] }],
    "PostToolUse": [{ "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command",
      "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook post-tool-use'" }] }],
    "Stop": [{ "hooks": [{ "type": "command",
      "command": "bash -c 'set -a; [ -f \"$HOME/.attestor/.env\" ] && . \"$HOME/.attestor/.env\"; set +a; attestor hook stop'" }] }]
  }
}
```

`attestor setup-claude-code` writes exactly this wiring for you. The `set -a` is required — without it, un-exported `.env` vars never reach the hook subprocess and hooks save nothing.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `attestor: command not found` | Not on PATH | `pipx install attestor`, or check `pip show attestor` |
| `connection refused` on 5432 / 5080 / 7687 | Containers not healthy yet | `docker ps`; wait for all three to report `healthy` |
| Neo4j auth error | Password mismatch | Align `NEO4J_PASSWORD` in `.env` with the container's `NEO4J_AUTH` |
| Embedder fails to initialize | Provider key missing for the `configs/attestor.yaml` embedder | Set the matching key in `.env` (e.g. `PINECONE_API_KEY` for Pinecone Inference) |
| Hooks save nothing | `.env` vars not exported to the subprocess | Use the `set -a; . "$HOME/.attestor/.env"; set +a; …` form above; check hook stderr for the error envelope |
| Doctor: 0 vectors but N memories | Embed dim ≠ schema `vector(N)` | Keep the embedder dim and the schema dim locked together |

---

## Chapter 02 — Sidecar REST API

Bring up the same three backends plus the API container, exposing the full `AgentMemory` surface over HTTP so non-Python agents can read/write memory:

```bash
cd attestor/infra/local
docker compose up -d            # postgres + pinecone + neo4j + attestor-api
curl localhost:8080/health      # {"ok": true, "data": {"healthy": true, ...}}
```

The API container (`attestor_api`) serves the same routes as the library — `/add`, `/recall`, `/search`, `/timeline`, `/forget`, `/memory/{id}`, `/health`, `/stats` (see [`attestor/api.py`](../attestor/api.py)). Any language can drive it via `MemoryClient` or raw REST. Backend config resolves from env (`POSTGRES_URL` / `NEO4J_URI` + `PINECONE_*`) and otherwise from `configs/attestor.yaml`; the vector (Pinecone) role is always preserved.

---

## Chapter 03 — Cloud Managed

The stack is the same; only connection strings change. Swap the local Compose services for managed equivalents and bind secrets via env:

| Role | Local | Managed options |
|------|-------|-----------------|
| Document | Postgres (Compose) | Neon · RDS · Cloud SQL · AlloyDB-as-PG · Cosmos PG flex |
| Vector | Pinecone Local | Pinecone Cloud (free Starter tier, or Standard from $50/mo) |
| Graph | Neo4j (Compose) | Neo4j AuraDB (or self-hosted Neo4j 5 + GDS) |

```bash
export POSTGRES_URL="postgresql://user:pass@managed-pg-host:5432/attestor"
export NEO4J_URI="neo4j+s://<auradb-id>.databases.neo4j.io"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="..."
export PINECONE_API_KEY="pcsk_..."   # Pinecone Cloud — index settings from configs/attestor.yaml
```

Run the API container (or your own image) with those env vars; `configs/attestor.yaml` remains the source of truth for the embedder, models, and retrieval budget. Validated reference deploys (App Runner / Cloud Run / Container Apps in front of managed Postgres + Pinecone + Neo4j) follow the same pattern — only DB hostnames and secrets differ.

> **Operational notes** (from cloud-deploy validation): Neo4j needs ≥512 MB RAM even idle (the JVM + GDS plugin OOM in 0.5 GB containers — use the next size up). Don't put Neo4j behind HTTP-only compute (`bolt://` is TCP/7687 — use a small VM in the same VPC, or a TCP-capable platform). Keep the embedder dim and the schema `vector(N)` locked together. Tighten ingress (5432 / 7687) to your compute's egress range before production.

---

## Uninstall

Attestor is **prompt-first**: tell Claude Code **"uninstall attestor"** (or run **`/uninstall-attestor`**) and it reverses all six install surfaces itself — following `commands/uninstall-attestor.md`:

1. **Package** — `pipx uninstall attestor` (run from `$HOME`, not the repo: a local `attestor/` dir makes pipx read the name as a path).
2. **`~/.attestor/`** — config + `.env` (no memory data lives here).
3. **Claude Code wiring** — the `attestor` MCP entry + Attestor's own hooks, content-matched on `attestor hook` so other tools' hooks are never touched (project `.claude/settings.json` / `.mcp.json`; the global file usually has none).
4. **Docker backends** — `attestor-*` containers + volumes (**confirm — this deletes all stored memory**).
5. **Plugin** — `/plugin uninstall attestor` (interactive).
6. **Stray repo artifacts** — `.cc_attestor_probe_store`, root `config.json`, `logs/`.

For local testing / CI there's a dry-run-by-default script that encodes the same procedure: `python scripts/attestor_uninstall.py` (add `--yes --containers --artifacts` to execute). It is a test/reference of the prompt above, not the primary path. Restart Claude Code afterward so it drops the orphaned MCP server + hooks.
