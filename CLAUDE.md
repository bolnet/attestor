# Memwright (agent-memory)

Embedded memory for AI agents. SQLite + pgvector + Neo4j.

## Project Structure

```
agent_memory/
  core.py              — AgentMemory class (main API)
  cli.py               — CLI entry point (argparse)
  models.py            — Memory, RetrievalResult dataclasses
  embeddings.py        — OpenAI/OpenRouter embedding client
  store/
    sqlite_store.py    — SQLite storage
    vector_store.py    — pgvector (PostgreSQL) store
    schema.sql         — SQLite schema
    schema_pg.sql      — PostgreSQL schema
  graph/
    neo4j_graph.py     — Neo4j entity graph
    extractor.py       — Entity/relation extraction
  retrieval/
    orchestrator.py    — Multi-layer retrieval with RRF fusion
    tag_matcher.py     — Tag matching layer
    scorer.py          — Scoring, boosts, dedup
  temporal/
    manager.py         — Contradiction detection, supersession, timeline
  extraction/
    extractor.py       — Memory extraction from conversations
  mcp/
    server.py          — MCP server for Claude Code integration
  utils/
    config.py          — MemoryConfig dataclass, load/save
tests/                 — Tests (require Docker running)
```

## Prerequisites

Docker must be running with containers up: `docker compose up -d`
This provides PostgreSQL (pgvector) and Neo4j.

## Development

- Python venv at `.venv` — use `.venv/bin/pytest`, `.venv/bin/python`
- Run tests: `.venv/bin/pytest tests/ -v`
- Tests require Docker containers running (PostgreSQL + Neo4j)

## Architecture

- **SQLite**: Core storage, always on
- **pgvector**: Semantic vector search (PostgreSQL)
- **Neo4j**: Entity graph for multi-hop traversal
- **Embeddings**: OpenAI text-embedding-3-small via OpenRouter or OpenAI direct
- **Retrieval**: 3-layer cascade (tag → graph expansion → vector) with RRF fusion
- **Temporal**: Automatic contradiction detection and supersession

## Key Conventions

- No fallbacks — pgvector and Neo4j are required, not optional
- Non-fatal errors in vector/graph operations are caught silently — core SQLite path never breaks
- Config auto-detects env vars: `PG_CONNECTION_STRING`, `NEO4J_PASSWORD`, `OPENROUTER_API_KEY`/`OPENAI_API_KEY`
- PyPI package name: `memwright` — Python import: `agent_memory`

## Health Check

Run `agent-memory doctor <store-path>` or call `mem.health()` to check all components.
The MCP server exposes `memory_health` tool — call it first when integrating.

## Environment Variables

- `PG_CONNECTION_STRING` — PostgreSQL connection (default: `postgresql://memwright:memwright@localhost:5432/memwright`)
- `NEO4J_PASSWORD` — Neo4j password (default: `memwright`)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_DATABASE` — Neo4j connection details
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` — required for embeddings
