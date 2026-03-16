# Phase 2: Claude Code Plugin and Auto-Capture - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Package memwright as a Claude Code plugin with lifecycle hooks for automatic memory capture. Plugin installs via marketplace, captures tool observations automatically via PostToolUse hook, injects relevant memories on SessionStart, and summarizes sessions on Stop. No explicit memory_add needed in the default flow.

</domain>

<decisions>
## Implementation Decisions

### Context injection
- Default token budget: **20,000 tokens** for SessionStart injection (~100-200 memories)
- Budget is configurable (users can lower to save tokens)
- All other context injection decisions (selection strategy, format, injection mechanism, scoping, privacy, first-run behavior) are Claude's discretion

### Claude's Discretion
Broad discretion granted across all other areas:

**Hook observation logic:**
- Which tools to capture (Write/Edit/Bash vs all)
- Observation granularity (file+action summary vs full I/O vs extracted facts)
- Extraction method (rule-based vs LLM)
- Dedup strategy (store-all vs batch-and-merge)

**Context injection (except budget):**
- Memory selection strategy (recency vs semantic vs hybrid)
- Injection format (structured vs natural language)
- Injection mechanism (additionalContext vs CLAUDE.md)
- Project scoping (per-project vs global+project)
- Superseded memory handling
- First-run experience
- Privacy/exclusion mechanisms

**Session summarization:**
- Summarization scope (decisions-only vs full narrative)
- Generation method (rule-based vs LLM)
- Storage approach (regular memory vs separate table)
- Timing strategy (Stop hook vs background flush)

**Plugin directory layout:**
- Plugin location (mono-repo vs separate)
- Pip/plugin install interaction
- Hook script language (Python vs shell wrappers)
- Memory store location (global vs per-project vs project-hashed)

</decisions>

<specifics>
## Specific Ideas

- Match claude-mem's passive UX — user installs once, memories captured automatically, no manual intervention
- 20K token injection budget is aggressive intentionally — user wants maximum context richness
- Existing `recall()` already handles token budget, temporal filtering, and RRF scoring — reuse for injection
- Existing `agent_memory/extraction/` has both rule_based.py and llm_extractor.py — can be used for observations
- SessionStart hook returns `additionalContext` field — cleanest injection path per Claude Code docs

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `agent_memory/core.py`: `AgentMemory.recall()` and `recall_as_context()` — direct reuse for SessionStart injection
- `agent_memory/core.py`: `AgentMemory.add()` — stores observations with tags, categories, entities
- `agent_memory/core.py`: `AgentMemory.extract()` — extracts memories from conversation messages
- `agent_memory/extraction/rule_based.py`: Rule-based extraction patterns
- `agent_memory/extraction/llm_extractor.py`: LLM-based extraction (optional)
- `agent_memory/mcp/server.py`: Existing MCP server with 8 tools — plugin bundles this

### Established Patterns
- Config via `MemoryConfig` dataclass with env var overrides
- Non-fatal error handling: vector/graph failures don't block core SQLite
- Token budget in `recall()`: already handles fitting results to budget

### Integration Points
- Plugin hooks call Python entry points that use `AgentMemory` API
- MCP server bundled via plugin `.mcp.json` pointing to existing server
- CLAUDE.md in plugin provides usage instructions auto-loaded by Claude Code
- `hooks/hooks.json` registers lifecycle hooks for SessionStart, PostToolUse, Stop

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-claude-code-plugin-and-auto-capture*
*Context gathered: 2026-03-16*
