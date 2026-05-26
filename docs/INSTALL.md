# Attestor Install Guide

A step-by-step guide to installing and verifying Attestor across different topologies. Each chapter is a self-contained setup you can follow from scratch.

**Chapters**

| # | Topology | Backend |
|---|----------|---------|
| [00](#chapter-00--install-via-claude-code-recommended) | **Install via Claude Code** (one prompt · cold start) | Postgres + Pinecone + Neo4j |
| [01](#chapter-01--local-stack-with-docker-compose) | Local stack (Docker Compose) | Postgres + pgvector, Neo4j + GDS |
| [02](#chapter-02--sidecar-rest-api) | Sidecar REST API | Same stack, exposed over HTTP |
| [03](#chapter-03--cloud-managed) | Cloud managed | Managed Postgres + Neo4j / Arango / AWS / Azure / GCP |

---

## Chapter 00 — Install via Claude Code (recommended)

**The one instruction.** Tell Claude Code **"install attestor to claude code"** (or run `/install-attestor`, or paste the prompt below). Claude **scans your machine first, looks up the current docs for every tool via Context7, then installs** — it brings up the three backend containers, installs the `attestor` package, wires the MCP server + hooks, and verifies. It assumes you start with **nothing installed**.

> Chapters 01–03 describe the lower-level / older pgvector topologies. Chapter 00 reflects the current **canonical stack: Postgres (document) + Pinecone (vector) + Neo4j (graph)**.

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

---

## Chapter 01 — Local stack with Docker Compose

Attestor needs two services: a Postgres with pgvector (document + vector roles) and a Neo4j with GDS (graph role). The easiest way to run both on a laptop is the included Docker Compose stack in `attestor/infra/local/`.

### Prerequisites

- Python 3.10 or later
- `pip`, `pipx`, or `poetry`
- Docker + Docker Compose
- An embedding provider — default is OpenAI `text-embedding-3-large` (1536 dims). Set `OPENAI_API_KEY` in `.env`.

### Step 1 — Start Postgres + Neo4j

```bash
cd attestor/infra/local
cp .env.example .env            # add OPENAI_API_KEY (and optionally OPENROUTER_API_KEY)
docker compose up -d postgres neo4j
```

This brings up two healthy containers:

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| `attestor-pg-local` | `attestor/db-postgres:16` (pgvector) | `5432` | Document + vector |
| `attestor-neo4j-local` | `neo4j:5.24-community` (+ GDS plugin) | `7474`, `7687` | Graph + PageRank / Leiden |

Wait for both to report healthy:

```bash
docker compose ps
```

### Step 2 — Install the CLI

```bash
# Via pip
pip install attestor

# Via pipx (isolated CLI)
pipx install attestor

# Via poetry (project dependency)
poetry add attestor
```

Verify:

```bash
attestor --help
```

### Step 3 — Point Attestor at the stack

Export connection info (matches the Compose defaults):

```bash
export POSTGRES_URL="postgresql://postgres:attestor@localhost:5432/attestor"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="attestor"
export OPENAI_API_KEY="sk-..."
```

Or put the same values in a `config.toml` under your store path.

### Step 4 — Write your first memory

```python
from attestor import AgentMemory

mem = AgentMemory()   # reads env / config.toml

mem.add(
    "The order service uses event sourcing with a 30-day retention policy",
    entity="order-service",
    tags=["architecture", "decision"],
)
```

This writes across roles:
- **Document + vector** (Postgres) — content, tags, entity, timestamp, confidence, pgvector embedding
- **Graph** (Neo4j) — entity node `order-service` + typed edges

### Step 5 — Recall

```python
results = mem.recall("how is the order service structured?", budget=2000)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

The 5-layer retrieval pipeline runs:
1. **Tag match** — Postgres FTS / trigram finds memories tagged with relevant terms
2. **Graph expansion** — Neo4j BFS depth=2 finds related entities
3. **Vector search** — pgvector cosine similarity on the query embedding
4. **Fusion + rank** — RRF (k=60) merges all candidates, applies PageRank and confidence decay
5. **Diversity + fit** — MMR (λ=0.7) removes near-duplicates, greedy packs under the token budget

### Step 6 — Run the doctor

```bash
attestor doctor
```

Expected:

```
Attestor Doctor
==================================================

Overall: ALL HEALTHY

  [OK] PostgresBackend (document + vector)
  [OK] Neo4jBackend (graph)
  [OK] Retrieval Pipeline
```

All checks must show `[OK]`. If the vector or graph role shows `[FAIL]`, retrieval degrades: the document path is the only hard dependency.

### Step 7 — Verify from the CLI

```bash
attestor add "API rate limit is 1000 req/min" --tags api,limits
attestor recall "what are the rate limits?"
attestor timeline --limit 10
attestor stats
```

### What's on disk

The Docker volumes hold all state:

| Volume | Purpose |
|--------|---------|
| `postgres_data` | Postgres data directory (documents, pgvector index) |
| `neo4j_data` | Neo4j data directory (graph) |
| `neo4j_logs` | Neo4j logs |
| `neo4j_plugins` | GDS plugin |

Wipe everything with `docker compose down -v`.

### Degradation

Attestor's retrieval pipeline tolerates partial outages:

- **Vector down** — falls back to tag match + graph expansion
- **Graph down** — falls back to tag match + vector search
- **Document store** is the only hard dependency

Non-fatal errors in vector or graph layers are caught and logged; the document path never breaks.

### Claude Code integration

The fastest path for Claude Code users:

```
> install attestor
```

The interactive wizard configures MCP scope, Postgres + Neo4j connections, hooks, namespace, and token budget.

Or add manually to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "attestor",
      "args": ["mcp"],
      "env": {
        "POSTGRES_URL": "postgresql://postgres:attestor@localhost:5432/attestor",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "attestor",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `attestor: command not found` | Not on PATH | Use `pipx install attestor` or check `pip show attestor` |
| `connection refused` on 5432/7687 | Containers not healthy yet | `docker compose ps`; wait for both to report `healthy` |
| Neo4j auth error | Password mismatch between env and container | Align `NEO4J_PASSWORD` in `.env` with `NEO4J_AUTH` |
| pgvector extension missing | Stock Postgres image without pgvector | Use the bundled `postgres.Dockerfile` (pgvector preinstalled) |
| OpenAI 401 on add/recall | Missing `OPENAI_API_KEY` | Export it or put it in `.env` before `docker compose up` |
| Doctor: 0 vectors but N memories | Embedding failed for some memories | Re-add affected memories; check `OPENAI_API_KEY` and network egress |

### Next steps

- **Chapter 02** — Run the REST API as a sidecar for non-Python agents
- **Chapter 03** — Switch to managed cloud backends for shared multi-agent deployments

---

## Chapter 02 — Sidecar REST API

Bring up the same stack plus the API container:

```bash
cd attestor/infra/local
docker compose up -d                 # postgres + neo4j + attestor-api
curl localhost:8080/health
```

The API container (`attestor-api-local`) exposes the same surface as `AgentMemory` over HTTP, so any language can read/write memory via `MemoryClient` or raw REST. See `attestor/api.py` for the route list.

---

## Chapter 03 — Cloud Managed

Swap the local Compose services for managed equivalents. Connection config is the only change.

| Provider | Document + vector | Graph |
|----------|-------------------|-------|
| Neon / RDS / Cloud SQL / AlloyDB | Postgres + pgvector | Neo4j AuraDB |
| ArangoDB Oasis | ArangoDB (doc + vector + graph in one) | — |
| AWS | DynamoDB + OpenSearch Serverless | Neptune |
| Azure | Cosmos DB DiskANN | NetworkX in-process (Azure extra ships `networkx`) |
| GCP | AlloyDB (pgvector + ScaNN) | Apache AGE on AlloyDB |

Reference Terraform lives under `attestor/infra/` for AWS, Azure, and GCP. Pick the backend via `config.toml`:

```toml
backend = "postgres"     # or "arangodb", "aws", "azure", "gcp"
```

Per-provider credentials are read from environment variables — never commit them. See each backend module (`attestor/store/*_backend.py`) for the exact variables expected.
