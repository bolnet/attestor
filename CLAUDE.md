# Memwright (agent-memory)

Embedded memory for AI agents. SQLite + ChromaDB + NetworkX.

## Project Structure

```
agent_memory/
  core.py              -- AgentMemory class (main API)
  cli.py               -- CLI entry point (argparse)
  models.py            -- Memory, RetrievalResult dataclasses
  store/
    sqlite_store.py    -- SQLite storage
    chroma_store.py    -- ChromaDB vector store (local sentence-transformers)
    schema.sql         -- SQLite schema
  graph/
    networkx_graph.py  -- NetworkX entity graph with JSON persistence
    extractor.py       -- Entity/relation extraction
  retrieval/
    orchestrator.py    -- Multi-layer retrieval with RRF fusion
    tag_matcher.py     -- Tag matching layer
    scorer.py          -- Scoring, boosts, dedup
  temporal/
    manager.py         -- Contradiction detection, supersession, timeline
  extraction/
    extractor.py       -- Memory extraction from conversations
  mcp/
    server.py          -- MCP server for Claude Code integration
  utils/
    config.py          -- MemoryConfig dataclass, load/save
tests/                 -- Tests (no Docker required)
```

## Prerequisites

None. All backends are embedded and zero-config:
- **SQLite** -- built into Python
- **ChromaDB** -- local persistent vector store with sentence-transformers embeddings
- **NetworkX** -- in-process graph with JSON persistence

No Docker, no external databases, no API keys required.

## Development

- Poetry manages dependencies -- use `poetry run pytest`, `poetry run python`
- Run tests: `poetry run pytest tests/ -v`
- Tests run without Docker or API keys

## Architecture

- **SQLite**: Core storage, always on
- **ChromaDB**: Semantic vector search with local sentence-transformers (all-MiniLM-L6-v2)
- **NetworkX**: In-process entity graph with multi-hop BFS traversal, JSON persistence
- **Retrieval**: 3-layer cascade (tag -> graph expansion -> vector) with RRF fusion
- **Temporal**: Automatic contradiction detection and supersession

## Key Conventions

- All backends are embedded -- no external services required
- Non-fatal errors in vector/graph operations are caught silently -- core SQLite path never breaks
- Zero config: `AgentMemory("./path")` provisions all backends automatically
- PyPI package name: `memwright` -- Python import: `agent_memory`

## Health Check

Run `agent-memory doctor <store-path>` or call `mem.health()` to check all components.
The MCP server exposes `memory_health` tool -- call it first when integrating.

Checks: SQLite, ChromaDB, NetworkX Graph, Retrieval Pipeline.
