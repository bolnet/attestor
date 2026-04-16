# Memwright (agent-memory)

The memory layer for agent teams. Self-hosted, deterministic retrieval, zero LLM in the critical path.

PyPI: `memwright` -- Python import: `agent_memory`.

## Project Structure

```
agent_memory/
  core.py              -- AgentMemory class (main public API)
  client.py            -- MemoryClient (HTTP drop-in for remote Memwright)
  context.py           -- AgentContext (identity, role, namespace, token budget, provenance)
  models.py            -- Memory, RetrievalResult dataclasses
  cli.py               -- CLI entry point (22 subcommands)
  api.py               -- Starlette ASGI REST API (8 routes)
  store/
    base.py            -- DocumentStore / VectorStore / GraphStore interfaces
    registry.py        -- Backend selection by config
    embeddings.py      -- Provider auto-detect (local / OpenAI / Bedrock / Vertex / Azure)
    sqlite_store.py    -- SQLite document store (WAL, 17 cols, 6 indexes)
    chroma_store.py    -- ChromaDB vector store (local all-MiniLM-L6-v2)
    postgres_backend.py-- pgvector + Apache AGE (Neon/any Postgres 16)
    arango_backend.py  -- ArangoDB (doc + vector + graph in one)
    aws_backend.py     -- DynamoDB + OpenSearch Serverless + Neptune
    azure_backend.py   -- Cosmos DB DiskANN
    gcp_backend.py     -- AlloyDB (pgvector + AGE + ScaNN)
    schema.sql
  graph/
    networkx_graph.py  -- MultiDiGraph, PageRank, multi-hop BFS, JSON persistence
    extractor.py       -- Entity/relation extraction
  retrieval/
    orchestrator.py    -- 5-layer cascade with RRF fusion
    tag_matcher.py
    scorer.py          -- Temporal + entity + PageRank boosts, confidence decay, MMR
  temporal/
    manager.py         -- Contradiction detection, supersession, timeline, as_of replay
  extraction/          -- Rule-based + optional LLM memory extraction
  mcp/
    server.py          -- MCP server (8 tools, 2 resources, 2 prompts)
  hooks/
    session_start.py   -- Context injection at session start
    post_tool_use.py   -- Auto-capture from Write/Edit/Bash
    stop.py            -- Session summary on exit
  utils/
    config.py, tokens.py
  infra/               -- Reference Terraform (AWS ECS+ArangoDB, Azure, GCP AlloyDB)
tests/                 -- 607 unit tests, no Docker, no API keys
```

## Prerequisites

None for local use. All local backends are embedded and zero-config:
- **SQLite** -- built into Python
- **ChromaDB** -- local sentence-transformers (all-MiniLM-L6-v2, 384D, ~90MB on first use)
- **NetworkX** -- in-process graph with JSON persistence

Cloud backends optional: PostgreSQL (pgvector+AGE), ArangoDB, AWS, Azure, GCP.

## Development

- Poetry manages dependencies -- `poetry run pytest`, `poetry run python`
- Run tests: `poetry run pytest tests/ -v` (no Docker, no API keys)
- Live integration tests (env-gated): Neon, Azure, ArangoDB

## Architecture

**Three storage roles, every memory persisted across all three:**

| Role | Purpose | Local | Cloud |
|------|---------|-------|-------|
| Document | Source of truth (content, tags, entity, ts, provenance, confidence) | SQLite | Postgres, AlloyDB, Arango, DynamoDB, Cosmos |
| Vector | Dense embedding per memory | ChromaDB | pgvector, ScaNN, Arango, OpenSearch, Cosmos DiskANN |
| Graph | Entity nodes + typed edges (`uses`, `authored-by`, `supersedes`) | NetworkX | Apache AGE, Arango, Neptune |

**5-layer retrieval pipeline (deterministic, no LLM):**
1. Tag Match (SQLite FTS)
2. Graph Expansion (BFS depth=2)
3. Vector Search (cosine similarity)
4. Fusion + Rank (RRF k=60, PageRank, confidence decay)
5. Diversity + Fit (MMR λ=0.7, greedy token-budget pack)

**Multi-agent primitives:** 6 RBAC roles (ORCHESTRATOR, PLANNER, EXECUTOR, RESEARCHER, REVIEWER, MONITOR), namespace isolation (row-level tenant column), provenance tracking, write quotas, token budgets per agent.

**Temporal:** Contradiction detection auto-supersedes older facts; nothing is deleted. Every fact has a validity window; `recall(as_of=...)` replays the past.

## Runtime topologies

- **Mode A — Embedded library**: `AgentMemory("./store")` in-process, sub-ms latency.
- **Mode B — Sidecar**: `memwright api` on `localhost:8080`, language-agnostic HTTP client.
- **Mode C — Shared service**: one Memwright service in front of an agent mesh (App Runner / Cloud Run / Container Apps).

Same API across all three. Only configuration changes.

## Key Conventions

- Local path: all backends embedded, no external services required
- Non-fatal errors in vector/graph are caught silently -- SQLite/document path never breaks
- Degradation is explicit and tiered: vector down → tag+graph; graph down → tag+vector; doc store is the only hard dependency
- Zero config: `AgentMemory("./path")` provisions all local backends automatically
- PyPI name: `memwright`; import: `agent_memory`; both `memwright` and `agent-memory` are CLI entry points

## Install for Claude Code

Single instruction users can give Claude Code: **`install agent memory`** (or run `/install-agent-memory`).

This triggers `commands/install-agent-memory.md` which interviews the user on:
1. Scope — global (`~/.claude/.mcp.json`) vs project (`.mcp.json`)
2. Store path — default `~/.memwright/` or custom
3. Backend — local (default) or cloud (Postgres/Arango/AWS/Azure/GCP)
4. Embedding provider — local (default), OpenAI, or cloud-native
5. Hooks — whether to wire session-start / post-tool-use / stop
6. Namespace + default token budget

Then it installs `memwright` via pipx/pip, writes the MCP config, optionally writes `settings.json` hooks, and runs `memwright doctor` to verify.

## Health Check

Run `memwright doctor <store-path>` or call `mem.health()`. The MCP server exposes `memory_health` -- call it first when integrating.

Checks: Document Store, Vector Store, Graph Store, Retrieval Pipeline.
