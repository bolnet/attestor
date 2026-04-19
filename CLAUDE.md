# Attestor

The memory layer for agent teams. Self-hosted, deterministic retrieval, zero LLM in the critical path.

PyPI: `attestor` -- Python import: `attestor`.

## Project Structure

```
attestor/
  core.py              -- AgentMemory class (main public API)
  client.py            -- MemoryClient (HTTP drop-in for remote Attestor)
  context.py           -- AgentContext (identity, role, namespace, token budget, provenance)
  models.py            -- Memory, RetrievalResult dataclasses
  cli.py               -- CLI entry point
  api.py               -- Starlette ASGI REST API
  store/
    base.py            -- DocumentStore / VectorStore / GraphStore interfaces
    registry.py        -- Backend selection; default = postgres + neo4j
    connection.py      -- Config layering / env resolution
    embeddings.py      -- Provider auto-detect (local / OpenAI / Bedrock / Vertex / Azure)
    postgres_backend.py-- pgvector (document + vector roles)
    neo4j_backend.py   -- Neo4j + GDS (graph role, PageRank / BFS)
    arango_backend.py  -- ArangoDB (doc + vector + graph in one, scoped backend only)
    aws_backend.py     -- DynamoDB + OpenSearch Serverless + Neptune
    azure_backend.py   -- Cosmos DB DiskANN (+ NetworkX in-process for graph)
    gcp_backend.py     -- AlloyDB (pgvector + AGE + ScaNN)
    schema.sql
  graph/
    extractor.py       -- Entity/relation extraction (4-output GraphRAG)
  retrieval/
    orchestrator.py    -- 5-layer cascade with RRF fusion
    tag_matcher.py
    scorer.py          -- Temporal + entity + PageRank boosts, confidence decay, MMR
  temporal/
    manager.py         -- Contradiction detection, supersession, timeline, as_of replay
  extraction/          -- Rule-based + optional LLM memory extraction
  mcp/
    server.py          -- MCP server (tools + resources + prompts)
  hooks/
    session_start.py   -- Context injection at session start
    post_tool_use.py   -- Auto-capture from Write/Edit/Bash
    stop.py            -- Session summary on exit
  utils/
    config.py, tokens.py
  infra/               -- Reference Terraform (AWS, Azure, GCP AlloyDB)
tests/                 -- Unit tests (live cloud tests env-gated)
```

## Prerequisites

Attestor requires two services for the default topology:

- **PostgreSQL 16+ with pgvector** -- holds the document and vector roles.
- **Neo4j 5+ with GDS** -- holds the graph role (PageRank, multi-hop BFS, Leiden).

Local development can run both via Docker (`attestor/infra/local/`). Cloud deployments can swap in a managed Postgres (Neon, RDS, Cloud SQL, AlloyDB) and managed Neo4j (AuraDB) or keep both self-hosted.

Other backends available as opt-in: ArangoDB, AWS (DynamoDB + OpenSearch + Neptune), Azure (Cosmos DB + NetworkX), GCP (AlloyDB with AGE).

## Development

- Poetry manages dependencies -- `poetry run pytest`, `poetry run python`
- Unit tests: `.venv/bin/pytest tests/ -q` (no external services required for unit layer)
- Live integration tests (env-gated): Postgres/Neon, Neo4j/AuraDB, Azure, ArangoDB

## Architecture

**Three storage roles, every memory persisted across the required backends:**

| Role | Purpose | Default Backend | Alternatives |
|------|---------|-----------------|--------------|
| Document | Source of truth (content, tags, entity, ts, provenance, confidence) | Postgres | AlloyDB, Arango, DynamoDB, Cosmos |
| Vector | Dense embedding per memory | Postgres (pgvector) | AlloyDB ScaNN, Arango, OpenSearch, Cosmos DiskANN |
| Graph | Entity nodes + typed edges (`uses`, `authored-by`, `supersedes`) | Neo4j (GDS) | AGE (AlloyDB), Arango, Neptune, NetworkX (Azure) |

**5-layer retrieval pipeline (deterministic, no LLM):**
1. Tag Match (Postgres FTS / trigram)
2. Graph Expansion (Neo4j BFS depth=2)
3. Vector Search (pgvector cosine)
4. Fusion + Rank (RRF k=60, PageRank, confidence decay)
5. Diversity + Fit (MMR λ=0.7, greedy token-budget pack)

**Multi-agent primitives:** 6 RBAC roles (ORCHESTRATOR, PLANNER, EXECUTOR, RESEARCHER, REVIEWER, MONITOR), namespace isolation (row-level tenant column), provenance tracking, write quotas, token budgets per agent.

**Temporal:** Contradiction detection auto-supersedes older facts; nothing is deleted. Every fact has a validity window; `recall(as_of=...)` replays the past.

## Runtime topologies

- **Mode A — Embedded library**: `AgentMemory(config)` in-process, talks directly to Postgres + Neo4j.
- **Mode B — Sidecar**: `attestor api` on `localhost:8080`, language-agnostic HTTP client shares the same Postgres + Neo4j.
- **Mode C — Shared service**: one Attestor service in front of an agent mesh (App Runner / Cloud Run / Container Apps) backed by managed Postgres + Neo4j.

Same API across all three. Only configuration changes.

## Key Conventions

- Postgres is the source of truth. Neo4j is derived graph state rebuildable from Postgres.
- Non-fatal errors in vector/graph are caught and logged -- the document path never breaks.
- Degradation is explicit and tiered: vector down → tag+graph; graph down → tag+vector; the document store is the only hard dependency.
- PyPI name: `attestor`; import: `attestor`; single CLI entry point `attestor`

## Install for Claude Code

Single instruction users can give Claude Code: **`install attestor`** (or run `/install-attestor`).

This triggers `commands/install-attestor.md` which interviews the user on:
1. Scope — global (`~/.claude/.mcp.json`) vs project (`.mcp.json`)
2. Postgres connection (local Docker, Neon, RDS, etc.)
3. Neo4j connection (local Docker, AuraDB, etc.)
4. Backend override (default postgres+neo4j, or arangodb/aws/azure/gcp)
5. Embedding provider — local sentence-transformers (opt-in), OpenAI, or cloud-native
6. Hooks — whether to wire session-start / post-tool-use / stop
7. Namespace + default token budget

Then it installs `attestor` via pipx/pip, writes the MCP config, optionally writes `settings.json` hooks, and runs `attestor doctor` to verify.

## Health Check

Run `attestor doctor` or call `mem.health()`. The MCP server exposes `memory_health` -- call it first when integrating.

Checks: Document Store (Postgres), Vector Store (pgvector), Graph Store (Neo4j), Retrieval Pipeline.
