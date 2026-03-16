# Roadmap: Memwright v2

## Overview

Memwright v2 is a brownfield rebuild that replaces Docker-dependent backends (pgvector, Neo4j) with zero-config local alternatives (ChromaDB, NetworkX), then layers Claude Code plugin integration (hooks, skills) and enhanced MCP features on top. The journey goes from "works but needs Docker" to "install and forget."

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Zero-Infrastructure Backends** - Replace pgvector/Neo4j with ChromaDB + NetworkX, remove all Docker deps, auto-provision on first use
- [ ] **Phase 2: Claude Code Plugin and Auto-Capture** - Plugin packaging, SessionStart/PostToolUse/Stop hooks for automatic memory capture
- [ ] **Phase 3: MCP Enhancements and Distribution** - MCP resources/prompts, skills, PyPI/registry publishing with new deps

## Phase Details

### Phase 1: Zero-Infrastructure Backends
**Goal**: Memwright works end-to-end with zero external dependencies — no Docker, no API keys, no configuration
**Depends on**: Nothing (first phase)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, CLEAN-05, VEC-01, VEC-02, VEC-03, VEC-04, VEC-05, GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04, GRAPH-05, PROV-01, PROV-02, PROV-03, PROV-04
**Success Criteria** (what must be TRUE):
  1. `AgentMemory("./test")` initializes all storage (SQLite, ChromaDB, NetworkX) without any env vars, Docker, or config files
  2. `mem.add(content, tags, entity)` stores to SQLite, indexes in ChromaDB with local embeddings, and populates NetworkX graph
  3. `mem.recall(query)` returns fused results from all three retrieval layers (tag, graph, vector) using RRF
  4. `mem.health()` reports all components healthy with no Docker/container checks
  5. No pgvector, Neo4j, psycopg, or neo4j driver code remains anywhere in the codebase
**Plans:** 1/3 plans executed

Plans:
- [ ] 01-01-PLAN.md — Delete pgvector/Neo4j/Docker code and strip config (CLEAN-01..05)
- [ ] 01-02-PLAN.md — Implement ChromaDB vector store and NetworkX graph (VEC-01..05, GRAPH-01..05)
- [ ] 01-03-PLAN.md — Wire auto-provisioning and update health check (PROV-01..04)

### Phase 2: Claude Code Plugin and Auto-Capture
**Goal**: Memwright installs as a Claude Code plugin and automatically captures/recalls memories during sessions via hooks
**Depends on**: Phase 1
**Requirements**: PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-05, HOOK-01, HOOK-02, HOOK-03, HOOK-04, HOOK-05
**Success Criteria** (what must be TRUE):
  1. `/plugin marketplace add bolnet/agent-memory` installs memwright with working MCP server and CLAUDE.md injected
  2. On session start, relevant memories are automatically injected into conversation context
  3. When the user writes/edits files or runs commands, observations are automatically captured as memories
  4. On session end, a session summary with key decisions is stored as a memory
  5. Hooks are registered in `hooks/hooks.json` and callable as shell commands
**Plans**: TBD

Plans:
- [ ] 02-01: Plugin manifest, marketplace catalog, MCP bundle, CLAUDE.md (PLUG-01..05)
- [ ] 02-02: SessionStart, PostToolUse, Stop hooks with registration (HOOK-01..05)

### Phase 3: MCP Enhancements and Distribution
**Goal**: Memwright is published with rich MCP integration (resources, prompts, skills) and available via both pip and plugin marketplace
**Depends on**: Phase 2
**Requirements**: SKILL-01, SKILL-02, SKILL-03, MCP-01, MCP-02, MCP-03, MCP-04, MCP-05, DIST-01, DIST-02, DIST-03, DIST-04
**Success Criteria** (what must be TRUE):
  1. `/memwright:mem-recall`, `/memwright:mem-timeline`, and `/memwright:mem-health` skills work as slash commands in Claude Code
  2. Entities and recent memories are @-mentionable via MCP resources (`@memwright:entity://name`)
  3. `/mcp__memwright__recall` and `/mcp__memwright__timeline` work as native MCP prompt commands
  4. `pip install memwright` on a clean machine gives a working zero-config memory system (no extras needed)
  5. MCP Registry listing reflects new version with updated deps and capabilities
**Plans**: TBD

Plans:
- [ ] 03-01: MCP resources, prompts, and updated tools (MCP-01..05)
- [ ] 03-02: Skills and distribution packaging (SKILL-01..03, DIST-01..04)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Zero-Infrastructure Backends | 1/3 | In Progress|  |
| 2. Claude Code Plugin and Auto-Capture | 0/2 | Not started | - |
| 3. MCP Enhancements and Distribution | 0/2 | Not started | - |
