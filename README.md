<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo-dark.svg">
    <img alt="Memwright" src="docs/logo.svg" width="400">
  </picture>
</p>

<p align="center">
  <em>Zero-config memory for Claude Code. No Docker. No API keys. Just install and go.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/v/memwright?color=C15F3C&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/pyversions/memwright?style=flat-square" alt="Python"></a>
  <a href="https://github.com/bolnet/agent-memory/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bolnet/agent-memory?style=flat-square" alt="License"></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.bolnet/memwright"><img src="https://img.shields.io/badge/MCP-Registry-C15F3C?style=flat-square" alt="MCP Registry"></a>
</p>

---

## The Problem

Claude Code forgets everything between sessions. Every new conversation starts from zero — no memory of what you built yesterday, what decisions you made, or what your project even does.

Claude Code's built-in auto-memory stores flat markdown files. No search. No ranking. No contradiction handling. As your project grows, those files become a wall of text that burns tokens without helping.

## What Memwright Does

Memwright gives Claude Code persistent, searchable memory that actually saves tokens:

- **Ranked retrieval** — 3-layer search (tags + entity graph + vector similarity) returns only the most relevant memories, not everything
- **Token budgets** — You set a ceiling (e.g. 2,000 tokens). Memwright fits the highest-scored memories within that budget. No overflow
- **Contradiction handling** — "User works at Google" automatically supersedes "User works at Meta". Stale facts don't pollute context
- **Zero config** — `poetry add memwright`, add one JSON block, restart Claude Code. Done

## Setup (1 minute)

### One-liner (Claude Code)

```bash
poetry add memwright
claude mcp add memory -- memwright mcp
```

Restart Claude Code. Approve the server once. Done — Claude now has 8 memory tools.

### Or configure manually

Add to `~/.claude/.mcp.json` (global) or `.mcp.json` (per-project):

```json
{
  "mcpServers": {
    "memory": {
      "command": "memwright",
      "args": ["mcp"]
    }
  }
}
```

### Verify

```bash
memwright doctor ~/.memwright
```

Or ask Claude to call `memory_health`. All 4 components should be healthy: SQLite, ChromaDB, NetworkX Graph, Retrieval Pipeline.

## Run as a Cloud Service

Deploy memwright as an HTTP API on any cloud. One script, one parameter.

```bash
./scripts/deploy.sh aws        # Lambda + API Gateway (serverless, pay-per-request)
./scripts/deploy.sh gcp        # Cloud Run (auto-scale 0–3, 2 CPU / 4GB)
./scripts/deploy.sh azure      # Container Apps (scale-to-zero, 2 CPU / 4GB)

./scripts/deploy.sh aws --teardown   # destroy everything
```

**Prerequisites**: Docker, Terraform, cloud CLI (`aws`/`gcloud`/`az`), ArangoDB credentials in `.env`.

Each command provisions all infrastructure via Terraform, builds a slim Docker image, deploys the Starlette ASGI API, and runs a health check.

| Cloud | What Terraform creates | Infra file |
|-------|----------------------|------------|
| AWS | ECR + Lambda (2GB, 120s) + API Gateway HTTP | `agent_memory/infra/lambda/main.tf` |
| GCP | Artifact Registry + Cloud Run (2 CPU, 4GB) | `agent_memory/infra/cloudrun/main.tf` |
| Azure | ACR + Log Analytics + Container Apps (2 CPU, 4GB) | `agent_memory/infra/containerapp/main.tf` |

## How It Works (And Why Tokens Don't Get Exhausted)

### Memory lives outside the context window

This is the key difference. Claude Code's built-in auto-memory loads `MEMORY.md` directly into the context window — every line, every message, always. Memwright stores memories in a separate process (SQLite + ChromaDB + NetworkX on disk). Claude's context window never sees them until it explicitly asks.

```
Claude Code auto-memory:                 Memwright:

┌──────────────────────────┐             ┌──────────────────────────┐
│  Context Window          │             │  Context Window          │
│                          │             │                          │
│  System prompt           │             │  System prompt           │
│  CLAUDE.md               │             │  CLAUDE.md               │
│  MEMORY.md ← ALL of it, │             │  User message            │
│    every message, grows  │             │  memory_recall → 2K max  │
│    forever               │             │                          │
│  User message            │             └──────────────────────────┘
│                          │
└──────────────────────────┘             ┌──────────────────────────┐
                                         │  Memwright (on disk)     │
                                         │  10,000 memories         │
                                         │  10,000 vectors          │
                                         │  500 entities            │
                                         │  ← never in context      │
                                         └──────────────────────────┘
```

### The cost stays flat as memory grows

```
MEMORY.md approach:
  Month 1:   2K tokens loaded every message
  Month 3:   8K tokens loaded every message
  Month 6:  15K tokens loaded every message  ← context getting crowded

Memwright approach:
  Month 1:   2K tokens max when recalled
  Month 3:   2K tokens max when recalled     (now ranking from 1,000 memories)
  Month 6:   2K tokens max when recalled     (now ranking from 5,000 memories)
                                              ← same cost, better results
```

More stored memories actually makes Memwright *better* — more candidates to rank from — while the context cost stays the same.

### How a recall actually works

When Claude calls `memory_recall("deployment setup", budget=2000)`:

1. **3 layers search in parallel** — tag match (SQLite), entity graph traversal (NetworkX), semantic vector search (ChromaDB)
2. **RRF fusion** — memories found by multiple layers score higher
3. **Temporal + entity boosts** — recent and entity-relevant memories rank higher
4. **Budget fitting** — takes results in score order until the budget is full

```
Store has 5,000 memories. Claude asks about "deployment setup".

  Tag search finds:     15 memories tagged "deployment"
  Graph search finds:    8 memories linked to "AWS", "Docker" entities
  Vector search finds:  20 semantically similar memories

  After dedup + RRF:    30 unique candidates, scored and ranked

  Budget fitting (2,000 tokens):
    Memory A (score 0.95):  500 tokens → in   (total: 500)
    Memory B (score 0.90):  600 tokens → in   (total: 1,100)
    Memory C (score 0.88):  400 tokens → in   (total: 1,500)
    Memory D (score 0.85):  300 tokens → in   (total: 1,800)
    Memory E (score 0.80):  400 tokens → SKIP (would exceed 2,000)

  Result: 4 memories, 1,800 tokens. The other 4,996 memories never entered context.
```

### Side-by-side comparison

| | Claude Code auto-memory | Memwright |
|---|---|---|
| Where memory lives | In context window (MEMORY.md) | On disk (separate process) |
| When it's loaded | Every message, always | Only when Claude calls `memory_recall` |
| Token cost | Grows with project history | Fixed ceiling you choose (2K, 4K, 20K) |
| Retrieval | None — full file dump | 3-layer ranked search, returns top results |
| Contradiction handling | Manual — you edit files | Automatic — new facts supersede old ones |
| After 6 months | 15K+ tokens of unranked context every message | 2K of the most relevant tokens, on demand |

## MCP Tools (What Claude Can Do)

Once the MCP server is running, Claude has these tools:

| Tool | What it does |
|------|-------------|
| `memory_add` | Store a fact with tags, category, entity, and confidence |
| `memory_recall` | Smart retrieval — set `budget` to control token usage (default: 2,000) |
| `memory_search` | Filter by category, entity, status, date range |
| `memory_get` | Fetch one memory by ID |
| `memory_forget` | Archive a memory (never deleted, just superseded) |
| `memory_timeline` | Chronological history of an entity |
| `memory_stats` | Store size, memory count, vector count, graph nodes |
| `memory_health` | Health check across all 4 components |

### MCP Resources & Prompts

- **@-mentions**: `@memwright:entity://python` — pull entity context into conversation
- **Prompts**: `/mcp__memwright__recall`, `/mcp__memwright__timeline`

## How Retrieval Works

```
Query
  ├─ Layer 1: Tag Match (SQLite)         → exact/partial tag hits
  ├─ Layer 2: Graph Expansion (NetworkX) → related entities via BFS
  └─ Layer 3: Vector Search (ChromaDB)   → semantic similarity
                    │
              RRF Fusion (k=60)
        memories in multiple layers score higher
                    │
            Temporal Boost (+0.2 max, decays over 90 days)
            Entity Boost (+0.15 to +0.30)
                    │
           fit_to_budget(results, token_budget)
        greedy selection by score, respects ceiling
                    │
             Ranked Results
```

Querying "Python" also finds memories about "FastAPI" if they're connected in the entity graph. Multi-hop reasoning through relationship traversal.

## Architecture

```
AgentMemory
├── Local (zero-config)
│   ├── SQLite           — Core document storage, ACID, always available
│   ├── ChromaDB         — Semantic vector search (local sentence-transformers)
│   └── NetworkX         — Entity graph, multi-hop BFS, JSON persistence
├── Cloud backends (single service fills all roles)
│   ├── PostgreSQL       — pgvector + Apache AGE (Neon, Cloud SQL, self-hosted)
│   ├── ArangoDB         — Native document + vector + graph (ArangoGraph Cloud)
│   ├── AWS              — ArangoDB Oasis on AWS (native document + vector + graph)
│   ├── Azure            — Cosmos DB (DiskANN vectors) + NetworkX graph
│   └── GCP              — AlloyDB (pgvector + AGE) + Vertex AI embeddings
├── Retrieval            — 3-layer cascade with RRF fusion + temporal/entity boosts
├── Temporal             — Contradiction detection, supersession, validity windows
├── Embeddings           — Auto-detected: cloud-native → OpenAI → local fallback
├── MCP Server           — 8 tools + resources + prompts (Claude Code integration)
└── CLI + Doctor         — Health check, benchmarks, export/import
```

The default local stack (SQLite + ChromaDB + NetworkX) works with zero config — no servers, no containers, no API keys. Cloud backends are opt-in and each fills all three roles (document, vector, graph) in a single service. If any optional component fails, the system degrades gracefully to SQLite-only.

## Cloud Backends

Each cloud backend fills all three roles (document store, vector search, entity graph) in a single service. Configure with `--backend` and `--backend-config` on any CLI command, or pass a `config` dict to the Python API.

### PostgreSQL (Neon, Cloud SQL, self-hosted)

Uses pgvector for vector search and Apache AGE for the entity graph. AGE is optional — on platforms without it (like Neon free tier), graph operations gracefully degrade and the backend still works for document + vector.

```python
mem = AgentMemory("./store", config={
    "backends": ["postgres"],
    "postgres": {"url": "postgresql://user:pass@host:5432/memwright"}
})
```

Tested on: **Neon** (free tier, $0), **GCP Cloud SQL** (PostgreSQL 15), **Docker** (PG16 + pgvector + AGE).

### ArangoDB (ArangoGraph Cloud, self-hosted)

Native document, vector, and graph support in one database. Supports both local Docker and ArangoGraph Cloud with TLS.

```python
mem = AgentMemory("./store", config={
    "backends": ["arangodb"],
    "arangodb": {"url": "https://your-instance.arangodb.cloud:8529", "database": "memwright"}
})
```

### AWS (ArangoDB Oasis)

ArangoDB Oasis deployed in AWS us-east-1. Native document, vector, and graph support — all three roles, no degradation. Uses the same `ArangoBackend` as ArangoGraph Cloud.

```python
mem = AgentMemory("./store", config={
    "backends": ["arangodb"],
    "arangodb": {
        "url": "https://your-oasis-instance.arangodb.cloud:8529",
        "database": "memwright",
        "auth": {"username": "root", "password": "$ARANGO_PASSWORD"},
    }
})
```

Setup: `bash agent_memory/infra/arango-oasis-setup.sh` (free tier, $0).
Teardown: `bash agent_memory/infra/arango-oasis-teardown.sh <deployment-id> <org-id>`.

### Azure (Cosmos DB)

Cosmos DB with DiskANN vector indexing for documents and vectors. Graph uses NetworkX in-memory with persistence to Cosmos containers. Supports API key or DefaultAzureCredential (managed identity).

```python
mem = AgentMemory("./store", config={
    "backends": ["azure"],
    "azure": {"cosmos_endpoint": "https://your-account.documents.azure.com:443/"}
})
```

Infrastructure: `agent_memory/infra/azure.tf` (Terraform).

### GCP (AlloyDB)

Extends the PostgreSQL backend with AlloyDB Connector (IAM auth via ADC) and Vertex AI embeddings (text-embedding-005, 768D). Falls back to psycopg2 via Auth Proxy if the connector isn't available.

```python
mem = AgentMemory("./store", config={
    "backends": ["gcp"],
    "gcp": {"project_id": "my-project", "cluster": "memwright", "instance": "primary"}
})
```

Infrastructure: `agent_memory/infra/gcp.tf` (Terraform).

### Backend CLI Usage

```bash
# Benchmarks with a cloud backend
agent-memory locomo --backend postgres --backend-config '{"url": "postgresql://..."}'
agent-memory mab --backend arangodb --backend-config '{"url": "https://..."}'

# Performance benchmark
python bench_perf.py --backend arangodb --backend-config '{"url": "http://localhost:8530"}'
```

## Embedding Providers

Memwright auto-detects the best available embedding provider:

| Priority | Provider | Model | Dimensions | Trigger |
|----------|----------|-------|------------|---------|
| 1 | Cloud-native | Bedrock Titan / Azure OpenAI / Vertex AI | 768-1536 | Cloud backend configured |
| 2 | OpenAI / OpenRouter | text-embedding-3-small | 1536 | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` set |
| 3 | Local (default) | all-MiniLM-L6-v2 | 384 | Always available, no API key needed |

The local fallback downloads ~90MB on first use. All providers implement the same interface — switching is transparent.

## Connection Configuration

A 3-layer config system resolves backend settings:

1. **Engine defaults** — sensible defaults per backend (ports, databases, TLS)
2. **User config** — `config.json` in store path or programmatic dict
3. **CLI overrides** — `--backend-config` arguments

Credentials support `$ENV_VAR` syntax for environment variable injection. TLS certificates can be provided as file paths or base64-encoded strings.

## Testing

### Running Tests

```bash
# All unit tests — no Docker, no API keys required
poetry run pytest tests/ -v

# Live PostgreSQL tests (Neon or any PostgreSQL with pgvector)
NEON_DATABASE_URL='postgresql://...' poetry run pytest tests/test_postgres_live.py -v

# Live Azure Cosmos DB tests
AZURE_COSMOS_ENDPOINT='https://...' AZURE_COSMOS_KEY='...' poetry run pytest tests/test_azure_live.py -v

# Performance benchmark
poetry run python bench_perf.py

# Accuracy benchmarks (need OPENROUTER_API_KEY)
poetry run agent-memory locomo --max-conversations 1 --max-questions 5 --verbose
poetry run agent-memory mab --max-examples 1 --max-questions 5 --verbose
```

### Test Coverage

- **392+ unit tests** covering all backends, config resolution, embeddings, retrieval, and CLI
- **14 live integration tests** against Neon PostgreSQL (free tier)
- **Mock tests** for every cloud backend (no cloud account needed)
- All unit tests run without Docker or API keys

### Test Files

| File | Tests | What it covers |
|------|-------|---------------|
| `test_connection.py` | 46 | 3-layer config, URL parsing, env resolution, auth/TLS |
| `test_embeddings.py` | 26 | All embedding providers, fallback chain |
| `test_registry.py` | — | Backend mapping, role conflicts |
| `test_postgres_backend.py` | — | Schema, CRUD, vector/graph ops |
| `test_postgres_live.py` | 14 | Live tests against real PostgreSQL |
| `test_arango_backend.py` | — | Document, vector, graph operations |
| `test_arango_live.py` | 14 | Live tests against ArangoDB (Oasis/local) |
| `test_azure_backend.py` | — | Cosmos DB containers, vector indexing |
| `test_aws_backend.py` | — | DynamoDB, OpenSearch, Neptune (legacy) |
| `test_gcp_backend.py` | — | AlloyDB connector, fallback logic |

## CLI

Both `memwright` and `agent-memory` work as entry points:

```bash
memwright mcp                          # Start MCP server (zero-config, uses ~/.memwright)
memwright mcp --path /custom/path      # Start with custom store location
memwright doctor ~/.memwright           # Health check

agent-memory add ./store "text" --tags "t1,t2" --category general
agent-memory recall ./store "query" --budget 4000
agent-memory search ./store --category project --entity Python
agent-memory list ./store
agent-memory timeline ./store --entity Python
agent-memory stats ./store
agent-memory export ./store -o backup.json
agent-memory import ./store backup.json
agent-memory compact ./store           # Remove archived memories
```

## Memory Store Location

The MCP server stores everything at `~/.memwright/` by default (configurable with `--path`). The store is a directory containing:

```
~/.memwright/
├── memory.db        — SQLite database (core storage)
├── config.json      — Token budget, min results config
├── graph.json       — NetworkX entity graph
└── chroma/          — ChromaDB vector store + embeddings
```

You can inspect memories directly: `sqlite3 ~/.memwright/memory.db "SELECT * FROM memories LIMIT 10;"`

## Configuration

Stored in `{store_path}/config.json`:

```json
{
  "default_token_budget": 2000,
  "min_results": 3
}
```

All fields are optional. Defaults apply if the file doesn't exist. The store auto-provisions on first use.

## Also Works With

While Claude Code is the primary target, the MCP server works with any MCP client:

| Client | Config file |
|--------|-------------|
| Claude Code | `.mcp.json` (project) or `~/.claude/.mcp.json` (global) |
| Cursor | `.cursor/mcp.json` |
| Windsurf | MCP config in settings |
| Any MCP client | Standard MCP stdio transport |

Same `memwright mcp` command. Same zero-config setup.

## Python API

For custom agents or scripts:

```python
from agent_memory import AgentMemory

mem = AgentMemory("./my-agent")  # auto-provisions all backends

# Store
mem.add("User prefers Python over Java",
        tags=["preference", "coding"], category="preference")

# Recall with token budget
results = mem.recall("what language?", budget=2000)

# Get formatted context string for prompt injection
context = mem.recall_as_context("user background", budget=4000)

# Contradiction handling — automatic
mem.add("User works at Google", tags=["career"], category="career", entity="Google")
mem.add("User works at Meta", tags=["career"], category="career", entity="Meta")
# ^ Google memory auto-superseded
```

## Benchmarks

### LOCOMO (Long Conversation Memory)

| System | Accuracy |
|--------|----------|
| MemMachine | 84.9% |
| **Memwright** | **81.2%** |
| Zep | ~75% |
| Letta | 74.0% |
| Mem0 (Graph) | 66.9% |
| OpenAI Memory | 52.9% |

*Scores are self-reported across vendors. [Methodology is disputed](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/).*

Retrieval is fully local — tag matching, graph traversal, vector search with RRF fusion. No LLM re-ranking. Embeddings are local (sentence-transformers). Only benchmark answer synthesis uses an LLM.

## Uninstall

### 1. Remove the MCP server config

**Global** — delete the `memory` entry from `~/.claude/.mcp.json`

**Per-project** — delete the `memory` entry from `.mcp.json` in your project root

### 2. Uninstall the package

```bash
poetry remove memwright
```

### 3. Delete stored memories (optional)

```bash
# Export first if you want a backup
agent-memory export ~/.memwright -o memwright-backup.json

# Then delete
rm -rf ~/.memwright
```

If you used per-project stores, also remove any `.memwright/` directories in your project roots.

## License

Apache 2.0

---

<sub>mcp-name: io.github.bolnet/memwright</sub>
