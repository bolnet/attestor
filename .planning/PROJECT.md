# Memwright v2 — Zero-Config Memory for Claude Code

## What This Is

Memwright is an embedded memory system for AI agents, rebuilt to match claude-mem's zero-friction UX while adding graph-based entity reasoning they can't do. Users install via `pip install memwright` or `/plugin marketplace add bolnet/agent-memory` and memories are automatically captured during Claude Code sessions — no Docker, no API keys, no configuration.

## Core Value

Zero-config automatic memory that just works — install and forget. Memories captured passively via hooks, recalled intelligently via tag + graph + vector fusion.

## Requirements

### Validated

- SQLite core storage with ACID guarantees — existing
- 3-layer retrieval cascade with RRF fusion — existing
- Temporal contradiction detection and supersession — existing
- MCP server with 8 tools — existing
- CLI with doctor, add, recall, search, timeline — existing
- Memory extraction from conversations — existing
- PyPI package (memwright 0.1.3) — existing
- MCP Registry listing — existing

### Active

- [ ] Replace pgvector with ChromaDB via MCP stdio subprocess (like claude-mem)
- [ ] Replace Neo4j with NetworkX in-memory graph (JSON-persisted)
- [ ] Remove all Docker dependencies — zero infrastructure
- [ ] Use ChromaDB built-in sentence-transformers — no API key needed
- [ ] Auto-provision all storage on first use (SQLite + ChromaDB + NetworkX)
- [ ] Claude Code plugin packaging (hooks, skills, CLAUDE.md)
- [ ] SessionStart hook — inject relevant memories into context
- [ ] PostToolUse hook — auto-capture observations from tool use
- [ ] Stop hook — summarize session, store key decisions
- [ ] Plugin skills (/memwright:mem-recall, /memwright:mem-timeline, /memwright:mem-health)
- [ ] MCP resources — memories as @-mentionable resources
- [ ] MCP prompts — native /mcp__memwright__recall slash commands
- [ ] Both install paths: pip (any MCP client) + plugin (Claude Code power users)

### Out of Scope

- pgvector / Neo4j / Docker — removed entirely, clean break
- OpenAI/OpenRouter embedding API — using ChromaDB built-in instead
- Web viewer UI — nice-to-have, not blocking
- Localized modes — not a priority for dev tool
- Benchmark updates (MAB/LOCOMO) — keep as-is, update after core stable

## Context

**Competitive landscape:** claude-mem has 35K stars with SQLite + ChromaDB, zero-Docker, automatic hook-based capture. They have no graph, no temporal reasoning, no benchmarks.

**Strategic position:** Match their UX (zero-config, auto-capture), beat them on retrieval (graph + temporal + RRF fusion), undercut on cost (local embeddings vs Claude subprocess per observation).

**Existing codebase:** Brownfield — significant code exists for SQLite store, retrieval orchestrator, temporal manager, MCP server, CLI. The retrieval pipeline, temporal logic, and MCP tools carry forward. pgvector, Neo4j, Docker health checks, and all Docker-related code gets deleted.

**Claude Code ecosystem:** Plugin system supports hooks (SessionStart, PostToolUse, Stop), skills (SKILL.md → slash commands), bundled MCP servers, and CLAUDE.md injection. MCP protocol supports resources (@-mentionable) and prompts (native slash commands).

## Constraints

- **Package name**: Keep `memwright` on PyPI for continuity
- **Python import**: Keep `agent_memory` module name
- **No API keys by default**: ChromaDB built-in embeddings (sentence-transformers) run locally
- **ChromaDB via stdio**: Vector search runs as MCP stdio subprocess, matching claude-mem pattern
- **Benchmarks untouched**: mab.py and locomo.py stay as-is for now

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ChromaDB via MCP stdio (not in-process) | Match claude-mem architecture exactly, process isolation | — Pending |
| NetworkX (not Neo4j) | Zero-Docker, in-memory ~0.1ms traversal, JSON-persisted | — Pending |
| Remove pgvector+Neo4j entirely | Clean break, no fallback complexity, simpler codebase | — Pending |
| ChromaDB built-in embeddings | No API key needed, true zero-config | — Pending |
| Plugin + pip dual distribution | Plugin for Claude Code power users, pip for any MCP client | — Pending |
| Auto-capture via hooks only | No explicit memory_add in default flow, passive like claude-mem | — Pending |

---
*Last updated: 2026-03-15 after initialization*
