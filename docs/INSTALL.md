# Attestor Install Guide

A step-by-step guide to installing and verifying Attestor across different topologies. Each chapter is a self-contained setup you can follow from scratch.

**Chapters**

| # | Topology | Backend |
|---|----------|---------|
| [00](#chapter-00--install-via-claude-code-recommended) | **Install via Claude Code** (one prompt · cold start) | Postgres + Pinecone + Neo4j |
| [01](#chapter-01--local-stack-with-docker-compose) | Local stack (Docker Compose) | Postgres + Pinecone + Neo4j |
| [02](#chapter-02--sidecar-rest-api) | Sidecar REST API | Same stack, exposed over HTTP |
| [03](#chapter-03--cloud-managed) | Cloud managed | Managed Postgres + Pinecone + Neo4j |
| [04](#chapter-04--governance-jobs-temporal) | Governance jobs (Temporal) — opt-in | Same stack + a Temporal server (shares Postgres) |

---

## Chapter 00 — Install via Claude Code (recommended)

**The one install:** `pipx install attestor && attestor quickstart` — zero questions, one default profile. It brings up the local backends (Postgres + Pinecone Local + Neo4j), uses a local **Ollama `bge-m3`** embedder (no cloud key), wires the Claude Code MCP server (`./.mcp.json`) + lifecycle hooks, and runs `attestor doctor`.

**Prerequisites:** Docker running, and Ollama serving `bge-m3` (`ollama pull bge-m3`). `quickstart` runs a preflight that scans the ports/tools and tells you if anything is missing — it never prompts.

Alternatively, **drive it from inside Claude Code via the plugin** (optional):
1. `/plugin marketplace add bolnet/attestor` (one-time)
2. `/plugin install attestor` (then ENABLE it in the `/plugin` → Installed menu)
3. `/attestor:install-attestor` (namespaced command — runs `attestor quickstart` for you)

> **Note:** Plugin commands are namespaced — the command is `/attestor:install-attestor`, not bare `/install-attestor`. A freshly-installed plugin can be disabled; enable it in the `/plugin` → Installed menu and `/reload-plugins`, or the command won't resolve.

> Every chapter targets the same **canonical stack: Postgres (document) + Pinecone (vector) + Neo4j (graph)**, with the embedder, models, and retrieval budget coming from the single source of truth, [`configs/attestor.yaml`](../configs/attestor.yaml). Chapter 00 is the fastest path; 01–03 are the manual local / sidecar / cloud setups.

### The three backends — three Docker containers (attestor_ prefix), one per role

Each storage role is its own container, all `attestor_*` prefixed:

| Container | Type | Storage role | Ports |
|-----------|------|--------------|-------|
| `attestor_postgres_document_db` | **Postgres 16 + pgvector** | **Document** — source of truth (content, tags, entity, ts, provenance, confidence) | `5432` |
| `attestor_pinecone_vector_db` | **Pinecone Local** | **Vector** — dense embeddings, per-namespace cosine search | `5080-5089` |
| `attestor_neo4j_graph_db` | **Neo4j 5 + GDS** | **Graph** — entity nodes + typed edges, PageRank / BFS | `7687` |

After `attestor quickstart` finishes, you can use the `memory_*` MCP tools immediately, and every project you open gets its own hard-isolated memory automatically.

> Hooks load the user environment before calling `attestor`. The wired hook command is
> `bash -c 'set -a; [ -f "$HOME/.attestor/.env" ] && . "$HOME/.attestor/.env"; set +a; attestor hook <event>'`.
> The `set -a` matters: a bare `source` of a `.env` with un-exported `KEY=value` lines leaves
> those vars shell-local, so the `attestor` subprocess never sees `ATTESTOR_CONFIG` / provider keys
> and the hook silently saves nothing.

---

## Chapter 01 — Manual local stack (Docker Compose only)

**Recommended: use `attestor quickstart` (Chapter 00) instead.** This chapter is for advanced users who want to bring up the backends manually without the preflight checks and auto-wiring.

Attestor's canonical stack is three services: **Postgres** (document role, source of truth), **Pinecone Local** (vector role, the free `:5080` Docker emulator), and **Neo4j + GDS** (graph role). The bundled Compose stack in `attestor/infra/local/` brings all three up on a laptop.

> The detailed, copy-paste-ready walkthrough — including the four health checks — lives in
> **[`docs/LOCAL_DOCKER_SETUP.md`](LOCAL_DOCKER_SETUP.md)**. This chapter is the short version.

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
docker compose up -d
```

This brings up three containers:

| Container | Port | Role |
|-----------|------|------|
| `attestor_postgres_document_db` | `5432` | Document |
| `attestor_pinecone_vector_db` | `5080-5090` | Vector |
| `attestor_neo4j_graph_db` | `7474`, `7687` | Graph |

Wait for all three to report healthy:

```bash
docker compose ps
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

The fastest path is Chapter 00 (`attestor quickstart`). To wire MCP + hooks manually afterward, add to `.mcp.json` (project) or `~/.claude/settings.json` (global):

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

The lifecycle hooks (SessionStart / PostToolUse / Stop) run as separate subprocesses that do **not** inherit your interactive shell, so they load `~/.attestor/.env` themselves:

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

The `set -a` is required — without it, un-exported `.env` vars never reach the hook subprocess and hooks save nothing.

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

The API container (`attestor_api`) serves the same routes as the library — `/add`, `/recall`, `/search`, `/timeline`, `/forget`, `/memory/{id}`, `/health`, `/stats` (see [`attestor/api.py`](../attestor/api.py)). Any language can drive it via `MemoryClient` or raw REST. With the opt-in durable profile ([Chapter 04](#chapter-04--governance-jobs-temporal)), run `attestor worker` next to the API container so governance jobs have a runner. Backend config resolves from env (`POSTGRES_URL` / `NEO4J_URI` + `PINECONE_*`) and otherwise from `configs/attestor.yaml`; the vector (Pinecone) role is always preserved.

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

## Chapter 04 — Governance jobs (Temporal)

**Opt-in.** Attestor's governance *jobs* — episode consolidation, derived-state repair and rebuild (Pinecone vectors + Neo4j graph from Postgres), the audit-first forget-user saga, and the retention / session sweeps — can run as durable workflows on [Temporal](https://temporal.io) (durable execution; unrelated to `attestor/temporal/`, which is temporal *reasoning*). Every job then has retry policies, deterministic idempotent ids, and an operator UI that shows exactly which lane is stuck or drifted. The design is in [`docs/plans/temporal-integration.md`](plans/temporal-integration.md).

Three hard rules never change:

- **Recall never touches Temporal.** The 6-step read path stays in-process and deterministic (`tests/test_durable_isolation.py` fails the build if `attestor/retrieval/**` or `attestor/hooks/**` ever imports `attestor.durable`).
- **Hooks never wait on Temporal.** Claude Code hooks keep today's in-process behaviour.
- **Postgres stays the source of truth.** Temporal holds job state only; workflow payloads carry ids, never memory content.

Nothing in this chapter runs unless you opt in: the default `attestor quickstart` and `configs/attestor.yaml` (`durable.enabled: false`) are unchanged.

### Step 1 — Install the extra

```bash
pipx install "attestor[durable]"          # or: pip install "attestor[durable]"
# in a checkout: .venv/bin/pip install "temporalio>=1.32,<2"
```

The extra adds the `temporalio` SDK (Python ≥ 3.10). With `durable.enabled: true` and the SDK missing, `attestor worker` and every `attestor durable …` command fail loudly rather than degrading.

### Step 2 — Start a Temporal server (compose profile `durable`)

The bundled compose file carries the server + UI behind a profile, so they never start by accident:

```bash
attestor quickstart --durable            # the zero-question install + the durable profile
# equivalent by hand:
cd attestor/infra/local
docker compose --profile durable up -d postgres neo4j pinecone temporal temporal-ui
```

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `attestor_temporal_server` | `temporalio/auto-setup` | `7233` | Temporal frontend (gRPC) — `durable.address` |
| `attestor_temporal_ui` | `temporalio/ui` | `8233` | Web UI — every job, retry, and failure |

`auto-setup` reuses the existing Postgres container and credentials (`DB=postgres12`, `POSTGRES_SEEDS=postgres`), creates its own `temporal` + `temporal_visibility` databases, and creates the `attestor` namespace so it matches `durable.namespace`. Memory data stays in the `attestor` database. Without Docker: `temporal server start-dev --db-filename ~/.attestor/temporal.db` (creates only the `default` namespace — set `durable.namespace: default` or `TEMPORAL_NAMESPACE=default`).

### Step 3 — Enable it in YAML

`configs/attestor.yaml` (or the copy in your store, `~/.attestor/attestor.yaml`) is authoritative:

```yaml
durable:
  enabled: true                  # opt-in; false → every durable.* call is an in-process no-op
  address: "localhost:7233"
  namespace: "attestor"
  task_queue: "attestor-governance"
  tls: false
  schedules:
    retention_sweep: "0 3 * * *"       # daily 03:00
    session_sweep: "*/10 * * * *"      # every 10 min
```

`TEMPORAL_ADDRESS` / `TEMPORAL_NAMESPACE` / `TEMPORAL_TASK_QUEUE` / `TEMPORAL_TLS` override these at runtime (managed Temporal Cloud: point `address` at your namespace endpoint and set `tls: true`).

### Step 4 — Run the worker

```bash
attestor worker                                   # foreground; one worker on durable.task_queue
attestor worker --task-queue attestor-governance  # overrides: --address, --namespace
```

The worker registers every governance workflow + activity on one task queue and fails at start if the server is unreachable. Run it wherever the API / MCP server runs (Mode B sidecar: alongside `attestor api`; Mode C shared service: one or more worker containers next to the service, same Postgres + Pinecone + Neo4j). It is the only process that talks to Temporal on the hot path — `add()` merely *starts* a repair workflow (fire-and-forget) when a vector or graph write fails.

### Step 5 — Apply the schedules and verify

```bash
attestor durable schedules apply      # create-or-update RetentionSweep + SessionSweep (idempotent)
attestor durable schedules list       # presence / paused / next run
attestor durable status               # config + server reachability
attestor durable status forget-<id>   # follow one workflow (e.g. a forget-user saga)
attestor durable rebuild --user <id> [--since ISO] [--namespace NS] [--wait]
                                      # rebuild vectors + graph from Postgres for one tenant
```

Open `http://localhost:8233` to watch jobs. Workflow ids are deterministic (`derive-{memory_id}`, `consolidate-{episode_id}`, `forget-{user_id}-{audit_id}`, `retention-sweep-…`, `session-sweep-…`), so a retry or a concurrent rebuild never runs the same job twice.

### Teardown

`attestor teardown` removes the Temporal containers along with the rest of the stack (it passes `--profile durable` to `docker compose down`); `--purge` also drops the volumes. `pipx uninstall attestor` removes the extra with the package.

---

## Uninstall

**The one command:** `attestor teardown` — zero-question reverse of `attestor quickstart`. It removes Docker backends + volumes (confirm — deletes all memory), MCP entry, hooks, plugin, and stray artifacts. Keeps `~/.attestor/` by default; add `--purge` to also wipe config + `.env`.

```bash
attestor teardown                    # preview only
attestor teardown --yes              # execute (keeps data volumes)
attestor teardown --yes --purge      # also wipe data volumes + ~/.attestor
```

You can also drive this from inside Claude Code: `/uninstall-attestor` runs the same uninstall. Restart Claude Code afterward so it drops the orphaned MCP server + hooks.
