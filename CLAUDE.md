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
    postgres_backend.py-- Postgres (document role; pgvector path remains as opt-in)
    pinecone_backend.py-- Pinecone (vector role; auto-detects Local Docker vs Cloud)
    neo4j_backend.py   -- Neo4j + GDS (graph role, PageRank / BFS)
    arango_backend.py  -- ArangoDB (doc + vector + graph in one, scoped backend only)
    aws_backend.py     -- DynamoDB + OpenSearch Serverless + Neptune
    azure_backend.py   -- Cosmos DB DiskANN (+ NetworkX in-process for graph)
    gcp_backend.py     -- AlloyDB (pgvector + AGE + ScaNN)
    schema.sql
  graph/
    extractor.py       -- Entity/relation extraction (4-output GraphRAG)
  retrieval/
    orchestrator.py    -- 6-step semantic-first cascade (vector → BM25 → RRF → graph → MMR → fit)
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

Attestor requires three services for the default topology (one per storage role):

- **PostgreSQL 16+** -- document role (source of truth for content, tags, entity, ts, provenance, confidence). pgvector remains in the schema as opt-in fallback.
- **Pinecone** -- vector role (dense embeddings, per-namespace isolation, cosine search). Pinecone Local (Docker, free) for development; Pinecone Cloud for production.
- **Neo4j 5+ with GDS** -- graph role (PageRank, multi-hop BFS, Leiden).

Local development can run all three via Docker (`attestor/infra/local/`). Cloud deployments can swap in managed Postgres (Neon, RDS, Cloud SQL, AlloyDB), managed Pinecone (Standard plan starts at $50/mo; free Starter tier = 5M embedding tokens/month), and managed Neo4j (AuraDB) or self-host any of them.

Other backends available as opt-in: pgvector (single-process self-contained), ArangoDB, AWS (DynamoDB + OpenSearch + Neptune), Azure (Cosmos DB + NetworkX), GCP (AlloyDB with AGE).

## Development

- Poetry manages dependencies -- `poetry run pytest`, `poetry run python`
- Unit tests: `.venv/bin/pytest tests/ -q` (no external services required for unit layer)
- Live integration tests (env-gated): Postgres/Neon, Neo4j/AuraDB, Azure, ArangoDB

## Architecture

**Three storage roles, every memory persisted across the required backends:**

| Role | Purpose | Default Backend | Alternatives |
|------|---------|-----------------|--------------|
| Document | Source of truth (content, tags, entity, ts, provenance, confidence) | Postgres | AlloyDB, Arango, DynamoDB, Cosmos |
| Vector | Dense embedding per memory | **Pinecone** (Local Docker / Cloud) | pgvector, AlloyDB ScaNN, Arango, OpenSearch, Cosmos DiskANN |
| Graph | Entity nodes + typed edges (`uses`, `authored-by`, `supersedes`) | Neo4j (GDS) | AGE (AlloyDB), Arango, Neptune, NetworkX (Azure) |

Default embedder is **Pinecone Inference `llama-text-embed-v2`** (NVIDIA-hosted, 1024-D) — matches the index dim and pairs naturally with the Pinecone vector store. Voyage / OpenAI / Ollama remain as opt-in providers via `attestor/store/embeddings.py`.

**6-step retrieval pipeline (deterministic, no LLM in the hot path):**
1. Vector top-K (Pinecone cosine; HyDE v2 lane optional)
2. BM25 lane (optional — Postgres FTS for v3, in-memory rank_bm25 for the experiment harness)
3. RRF blend (vector + BM25 → unified rank, k=60)
4. Graph narrow (Neo4j BFS depth=2 affinity bonus + synthetic triple injection)
5. MMR diversity (λ=0.7) + confidence decay
6. Token-budget pack (greedy fit to recall budget)

Implementation: `attestor/retrieval/orchestrator.py`. The pipeline is
semantic-first (vector → BM25 → RRF → graph → MMR → fit) as of 2026-04-19;
a separate tag-match utility lives at `attestor/retrieval/tag_matcher.py`
but is consumed by the entity extractor, not the recall pipeline.

**Multi-agent primitives:** 6 RBAC roles (ORCHESTRATOR, PLANNER, EXECUTOR, RESEARCHER, REVIEWER, MONITOR) — **enforced at the AgentContext layer** via `ROLE_PERMISSIONS` in `attestor/context.py`. Matrix: ORCHESTRATOR = READ+WRITE+FORGET; PLANNER/EXECUTOR/RESEARCHER = READ+WRITE; REVIEWER/MONITOR = READ only. `read_only=True` is an independent kill switch that strips WRITE+FORGET regardless of role. Direct `AgentMemory.add()` calls (without an AgentContext) bypass the matrix — RBAC is a context-layer guarantee, not a backend one. Namespace isolation is row-level on Postgres but **not yet enforced on Neo4j** (graph entity nodes are global across namespaces). Provenance tracking, write quotas, and per-agent token budgets are wired.

**Temporal:** Contradiction detection auto-supersedes older facts; nothing is deleted. Every fact has a validity window; `recall(as_of=...)` replays the past.

## Runtime topologies

- **Mode A — Embedded library**: `AgentMemory(config)` in-process, talks directly to Postgres + Pinecone + Neo4j.
- **Mode B — Sidecar**: `attestor api` on `localhost:8080`, language-agnostic HTTP client shares the same Postgres + Pinecone + Neo4j.
- **Mode C — Shared service**: one Attestor service in front of an agent mesh (App Runner / Cloud Run / Container Apps) backed by managed Postgres + Pinecone + Neo4j.

Same API across all three. Only configuration changes.

## Key Conventions

- Postgres is the source of truth. Pinecone vectors and Neo4j graph are derived state, both rebuildable from Postgres.
- Non-fatal errors in vector/graph are caught and logged -- the document path never breaks.
- Degradation is explicit and tiered: vector down → tag+graph; graph down → tag+vector; the document store is the only hard dependency.
- PyPI name: `attestor`; import: `attestor`; single CLI entry point `attestor`

## Install for Claude Code

Single instruction users can give Claude Code: **`install attestor`** (or run `/install-attestor`).

This triggers `commands/install-attestor.md` which interviews the user on:
1. Scope — global (`~/.claude/.mcp.json`) vs project (`.mcp.json`)
2. Postgres connection (local Docker, Neon, RDS, etc.)
3. Pinecone connection (Local Docker via `localhost:5080`, or Cloud via `*.pinecone.io` host + API key)
4. Neo4j connection (local Docker, AuraDB, etc.)
5. Backend override (default postgres+pinecone+neo4j, or pgvector / arangodb / aws / azure / gcp)
6. Embedding provider — Pinecone Inference (default), Voyage, OpenAI, or local Ollama
7. Hooks — whether to wire session-start / post-tool-use / stop
8. Namespace + default token budget

Then it installs `attestor` via pipx/pip, writes the MCP config, optionally writes `settings.json` hooks, and runs `attestor doctor` to verify.

## Health Check

Run `attestor doctor` or call `mem.health()`. The MCP server exposes `memory_health` -- call it first when integrating.

Checks: Document Store (Postgres), Vector Store (Pinecone Local or Cloud), Graph Store (Neo4j), Retrieval Pipeline.
