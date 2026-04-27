# Attestor

**The memory layer for agent teams.** Self-hosted, deterministic retrieval, zero LLM in the critical path.

```
pip install attestor          # Python library + CLI + REST API + MCP server
```

- **PyPI:** `attestor`
- **Import:** `attestor`
- **Status:** v4.0.0 alpha (greenfield; no v3 migration path)
- **License:** see `LICENSE`

---

## What it is

Attestor is a memory store for multi-agent systems where the same memories are persisted across **three storage roles** — document, vector, and graph — and read back through a **deterministic 6-step retrieval pipeline** that never calls an LLM on the hot path.

It is designed for:

- agent teams that need a shared, tenant-isolated memory store with RBAC and audit trails
- products that must answer *"what did we know on 2026-03-01?"* (bi-temporal replay)
- environments where latency, cost, and reproducibility matter more than peak relevance

It is **not** a general vector database, a RAG framework, or an LLM agent runtime. It plugs into your agent stack as a backend.

---

## Quick start

### 1. Install

```bash
pip install attestor
# or
pipx install attestor
```

### 2. Bring up a local Postgres + Neo4j

```bash
attestor setup local      # writes docker-compose.yml under attestor/infra/local
docker compose -f attestor/infra/local/docker-compose.yml up -d
```

Postgres ships with `pgvector`. Neo4j ships with GDS for PageRank / BFS / Leiden.

### 3. Pull the default embedder

```bash
ollama pull bge-m3        # 1024-D, 8K context, local default
```

### 4. Verify everything is wired up

```bash
attestor doctor
```

You should see green for **Document Store (Postgres)**, **Vector Store (pgvector)**, **Graph Store (Neo4j)**, and **Retrieval Pipeline**.

### 5. Use it

```python
from attestor import AgentMemory, AgentContext, AgentRole

mem = AgentMemory()  # picks up env / ~/.attestor.toml automatically

ctx = AgentContext(
    agent_id="researcher-1",
    role=AgentRole.RESEARCHER,
    namespace="acme-prod",
)

mem.add(
    content="Alice is the engineering manager",
    entity="alice",
    category="role",
    context=ctx,
)

results = mem.recall(query="who runs engineering?", context=ctx)
for r in results:
    print(r.score, r.memory.content)
```

---

## Architecture

### Three storage roles

Every memory is persisted across the required backends in your topology:

| Role | Purpose | Default | Alternatives |
|------|---------|---------|--------------|
| **Document** | Source of truth (content, tags, entity, ts, provenance, confidence) | Postgres | AlloyDB, ArangoDB, DynamoDB, Cosmos DB |
| **Vector** | Dense embedding per memory | Postgres + pgvector | AlloyDB ScaNN, ArangoDB, OpenSearch Serverless, Cosmos DiskANN |
| **Graph** | Entity nodes + typed edges (`uses`, `authored-by`, `supersedes`) | Neo4j (GDS) | AGE on AlloyDB, ArangoDB, Neptune, NetworkX (Azure) |

Postgres is always the source of truth. Neo4j (or whichever graph store you pick) is **derived state, rebuildable from Postgres**. If the vector or graph store goes down, the document path keeps serving — degradation is explicit and tiered.

### Retrieval — semantic-first, no LLM in the loop

The orchestrator (`attestor/retrieval/orchestrator.py`) runs the same 6 steps for every query:

1. **Vector semantic search** — top-K = 50 nearest by cosine
2. **Graph narrow** — soft boost by hop-distance (BFS depth ≤ 2) from each candidate's entity to the question entities
3. **Triples injection** — typed-edge facts injected as synthetic memories so the consumer can reason over relations, not just text
4. **MMR rerank** — λ = 0.7 diversity vs. relevance trade-off
5. **Confidence decay + temporal boost** — recency / decay applied per memory
6. **Budget fit** — greedy pack into the agent's token budget

Optional **BM25 / FTS lane** (`bm25_lane=...` on the orchestrator) runs a `tsvector` keyword search in parallel and **fuses with the vector lane via Reciprocal Rank Fusion (RRF, k = 60)**. Useful for acronyms, IDs, and rare proper nouns where embeddings under-recall. Gracefully no-ops on backends that don't have the `content_tsv` column.

Every call writes a JSONL trace to `logs/attestor_trace.jsonl` (disable with `ATTESTOR_TRACE=0`).

### Multi-agent primitives

- **Six RBAC roles** (`AgentRole` enum): `ORCHESTRATOR`, `PLANNER`, `EXECUTOR`, `RESEARCHER`, `REVIEWER`, `MONITOR`
- **Namespace isolation** via Postgres Row-Level Security on every tenant table (`tenant_isolation_*` policies on the `attestor.current_user_id` session var)
- **Provenance** on every memory (who wrote it, which session, which episode)
- **Per-agent write quotas and token budgets**
- **Recall cache** per `AgentContext` session — repeated queries are deduplicated transparently
- **`AgentContext.scratchpad: Dict[str, Any]`** — typed lane for planner → executor handoffs without rounding through the store

### Bi-temporal — nothing is deleted

Every memory has two time axes:

- **Event time** — `valid_from` / `valid_until` (when the fact is true in the world)
- **Transaction time** — `t_created` / `t_expired` (when the row landed in the store)

Plus a `superseded_by` chain. Old facts are never deleted — they remain queryable forever.

```python
# What did we believe on March 1?
mem.recall(query="who runs engineering?", as_of="2026-03-01T00:00:00Z", context=ctx)

# Show me everything we knew about Alice between Feb and Apr
mem.recall(query="alice", time_window=("2026-02-01", "2026-04-01"), context=ctx)
```

**Auto-supersession on write.** `add()` runs `TemporalManager.check_contradictions()` against the new memory before insert; any active row with the same `(entity, category, namespace)` and different content is automatically marked superseded by the new id (`status="superseded"`, `valid_until=now`, `superseded_by=<new_id>`). Detection is rule-based string-equality today — soft / similarity-based contradiction is on the roadmap.

### Chain-of-Note reading

```python
pack = mem.recall_as_pack(query="who runs engineering?", context=ctx)
# pack.memories: list of {id, content, validity_window, confidence, source_episode_id}
# pack.prompt: default Chain-of-Note prompt (NOTES → SYNTHESIS → CITE → ABSTAIN → CONFLICT)
```

The default prompt has explicit ABSTAIN and CONFLICT clauses — every frontier model defaults to confabulation otherwise.

---

## Runtime topologies

Same API across all three. Only configuration changes.

| Mode | Shape | When to use |
|------|-------|-------------|
| **A — Embedded library** | `AgentMemory(config)` in-process, talks directly to Postgres + Neo4j | Single-process agents, scripts, notebooks |
| **B — Sidecar** | `attestor api` on `localhost:8080`, language-agnostic HTTP client shares the same Postgres + Neo4j | Polyglot agents on one box (Python + TS + Go) |
| **C — Shared service** | One Attestor service in front of an agent mesh (App Runner / Cloud Run / Container Apps) backed by managed Postgres + Neo4j | Production multi-agent platforms |

```bash
attestor api    --port 8080         # Mode B / C — Starlette ASGI REST API
attestor serve  --transport stdio   # MCP stdio server
attestor mcp    --transport http    # MCP HTTP server
```

---

## Backends

| Backend | Document | Vector | Graph | Status |
|---------|:--------:|:------:|:-----:|--------|
| **Postgres + Neo4j** *(default)* | ✓ | pgvector | Neo4j + GDS | Production-ready |
| **ArangoDB** | ✓ | ✓ | ✓ | Production-ready (single backend covers all 3 roles) |
| **AWS** | DynamoDB | OpenSearch Serverless | Neptune | Backend code + Terraform shipped |
| **Azure** | Cosmos DB | Cosmos DiskANN | NetworkX (in-process) | Backend code shipped, Terraform forthcoming |
| **GCP** | AlloyDB | AlloyDB ScaNN | AGE on AlloyDB | Backend code shipped, Terraform forthcoming |

Override the default via config:

```toml
# ~/.attestor.toml
backend = "postgres+neo4j"   # or "arangodb" | "aws" | "azure" | "gcp"
```

Reference Terraform lives under `attestor/infra/`.

---

## Embeddings

Provider auto-detect (`attestor/store/embeddings.py`), in this order:

1. **Local Ollama `bge-m3`** — 1024-D, 8K context — used when `http://localhost:11434` is reachable
2. **Cloud-native** — Bedrock Titan / Vertex / Azure OpenAI when their SDK + creds are present
3. **OpenAI `text-embedding-3-large`** (3072-D) — when `OPENAI_API_KEY` is set
4. **OpenRouter** — for federated runs

Local models are the default. Cloud APIs are fallbacks. Override via env:

```bash
export ATTESTOR_EMBEDDING_PROVIDER=openai
export ATTESTOR_EMBEDDING_MODEL=text-embedding-3-large
```

---

## CLI

`attestor --help` lists all commands. The most useful ones:

| Command | Purpose |
|---------|---------|
| `attestor init` | Create a starter config |
| `attestor setup local` | Generate local Docker Compose for Postgres + Neo4j |
| `attestor doctor` | Health-check every store + the retrieval pipeline |
| `attestor add` | Add a memory |
| `attestor recall` | Recall relevant memories |
| `attestor search` | Search with filters (entity / category / namespace / time window) |
| `attestor list` | List memories |
| `attestor timeline` | Show entity timeline |
| `attestor stats` | Store statistics |
| `attestor export` / `import` | JSON export and import |
| `attestor compact` | Remove archived memories |
| `attestor update` / `forget` | Mutate a memory's content / archive it |
| `attestor inspect` | Inspect the raw database |
| `attestor api` | Start the Starlette REST API |
| `attestor serve` | Start MCP server (stdio) |
| `attestor mcp` | Start MCP server (HTTP) |
| `attestor ui` | Browser UI for the store |
| `attestor hook {session-start,post-tool-use,stop}` | Run a Claude Code lifecycle hook |
| `attestor locomo` / `attestor mab` / `attestor lme` | Built-in benchmark runners (LoCoMo, MultiAgentBench, LongMemEval) |

---

## MCP server

`attestor serve` exposes an MCP server with these tools:

| Tool | Purpose |
|------|---------|
| `memory_add` | Write a memory with provenance |
| `memory_get` | Fetch one memory by id |
| `memory_recall` | Run the full retrieval pipeline |
| `memory_search` | Filtered list (entity / category / time / namespace) |
| `memory_forget` | Archive a memory by id |
| `memory_timeline` | Chronology for an entity |
| `memory_stats` | Store statistics |
| `memory_health` | Per-role health snapshot — call this first when integrating |

Plus MCP **resources** (memory listings) and **prompts** (canned recall prompts for IDE assistants).

---

## Hooks (Claude Code)

Three lifecycle hooks ship in `attestor/hooks/`:

- **`session_start`** — injects relevant memories into the session context based on cwd / repo
- **`post_tool_use`** — auto-captures useful artifacts from `Write` / `Edit` / `Bash`
- **`stop`** — writes a session summary on exit

Wire them up via the installer (next section) or by hand in `~/.claude/settings.json`.

---

## Install for Claude Code

Single instruction users can give Claude Code:

```
install attestor
```

(Or run `/install-attestor`.) The installer (`commands/install-attestor.md`) interviews you on:

1. **Scope** — global (`~/.claude/.mcp.json`) vs project (`.mcp.json`)
2. **Postgres connection** — local Docker, Neon, RDS, etc.
3. **Neo4j connection** — local Docker, AuraDB, etc.
4. **Backend override** — default `postgres+neo4j`, or `arangodb` / `aws` / `azure` / `gcp`
5. **Embedding provider** — local Ollama (default), OpenAI, or cloud-native
6. **Hooks** — whether to wire `session-start` / `post-tool-use` / `stop`
7. **Namespace + default token budget**

Then it installs `attestor` via pipx, writes the MCP config, optionally writes `settings.json` hooks, and runs `attestor doctor` to verify.

---

## Project layout

```
attestor/
  core.py                -- AgentMemory (main public API)
  client.py              -- MemoryClient (HTTP drop-in for remote Attestor)
  context.py             -- AgentContext (identity, role, namespace, budget, scratchpad)
  models.py              -- Memory, RetrievalResult, ContextPack
  cli.py                 -- attestor CLI entry point
  api.py                 -- Starlette ASGI REST API
  longmemeval.py         -- LongMemEval benchmark runner
  store/
    base.py              -- DocumentStore / VectorStore / GraphStore protocols
    registry.py          -- Backend selection
    connection.py        -- Config layering / env resolution
    embeddings.py        -- Provider auto-detect (Ollama / OpenAI / Bedrock / Vertex / Azure)
    postgres_backend.py  -- pgvector (document + vector roles)
    neo4j_backend.py     -- Neo4j + GDS (graph role)
    arango_backend.py    -- All 3 roles in one
    aws_backend.py       -- DynamoDB + OpenSearch Serverless + Neptune
    azure_backend.py     -- Cosmos DB DiskANN + NetworkX
    gcp_backend.py       -- AlloyDB pgvector + AGE + ScaNN
    schema.sql           -- v4 Postgres schema (RLS, bi-temporal columns)
  graph/
    extractor.py         -- Deterministic entity / relation extraction
  retrieval/
    orchestrator.py      -- 6-step semantic-first pipeline
    tag_matcher.py
    scorer.py            -- MMR, confidence decay, entity boost, fit
    trace.py             -- JSONL trace writer
  temporal/
    manager.py           -- Timelines, supersession, contradiction detection, as_of replay
  extraction/             -- Rule-based + optional LLM memory extraction
  mcp/
    server.py            -- MCP server (tools, resources, prompts)
  hooks/
    session_start.py
    post_tool_use.py
    stop.py
  utils/
    config.py, tokens.py
  infra/
    local/                -- Docker Compose (Postgres + Neo4j)
    aws_arango/           -- Reference Terraform
tests/                    -- Unit tests; live cloud tests env-gated
evals/                    -- LongMemEval / LoCoMo / MultiAgentBench harnesses
docs/                     -- Architecture notes, ADRs
commands/                 -- /install-attestor, etc.
```

---

## Development

```bash
poetry install
poetry run pytest tests/ -q                          # unit tests, no external services needed
ATTESTOR_LIVE_PG=1 poetry run pytest tests/live -q   # live integration (env-gated)
```

Style:

- `black` for formatting, `isort` for imports, `ruff` for lint, `mypy` for types
- PEP 8, type-annotated signatures, dataclasses for DTOs
- Many small files (200–400 lines typical, 800 max)

Conventions worth knowing before you contribute:

- Postgres is the source of truth. Neo4j is derived; rebuild it from Postgres if it drifts.
- Non-fatal errors in vector / graph paths are caught and logged. The document path never silently breaks.
- Configuration layering: env vars → `~/.attestor.toml` → in-code overrides.

---

## Health check

Always call this first when integrating:

```bash
attestor doctor                  # CLI
```

```python
mem = AgentMemory()
print(mem.health())              # Python API
```

```jsonc
// MCP
{ "tool": "memory_health" }
```

It probes Document Store (Postgres), Vector Store (pgvector), Graph Store (Neo4j), and the retrieval pipeline. If any role is degraded, `recall()` continues against the surviving roles and the response includes the degradation in its trace.

---

## Status & versioning

- **Version:** 4.0.0 (alpha)
- **v3 → v4:** greenfield rebuild. v3 was alpha-only with no production users; **there is no automated migration**. Drop your v3 DB and reinstall.
- See [`CHANGELOG.md`](./CHANGELOG.md) for the full track-by-track changelog.

---

## License

See [`LICENSE`](./LICENSE).
