---
phase: 03-mcp-enhancements-and-distribution
plan: 01
subsystem: mcp
tags: [mcp, resources, prompts, claude-code]

requires:
  - phase: 01-core-rebuild
    provides: ChromaDB vector store, NetworkX graph, core AgentMemory API
  - phase: 02-claude-code-plugin
    provides: MCP server with 8 tools, hooks, mcp subcommand

provides:
  - MCP resources for entities and recent memories (@-mentionable in Claude Code)
  - MCP prompts for recall and timeline (/mcp__memwright__recall, /mcp__memwright__timeline)
  - _build_handlers function for testable resource/prompt handler extraction

affects: [03-02, distribution, mcp-registry]

tech-stack:
  added: [pytest-asyncio]
  patterns: [_build_handlers extraction for MCP handler testability]

key-files:
  created: [tests/test_mcp_server.py]
  modified: [agent_memory/mcp/server.py]

key-decisions:
  - "_build_handlers extracted as standalone function for direct test invocation without full MCP server setup"
  - "read_resource returns plain JSON string (not ReadResourceContents) -- MCP SDK wraps it automatically"

patterns-established:
  - "_build_handlers pattern: extract async handler functions for unit testing MCP resources/prompts"

requirements-completed: [MCP-01, MCP-02, MCP-03, MCP-04, MCP-05]

duration: 4min
completed: 2026-03-16
---

# Phase 3 Plan 1: MCP Resources and Prompts Summary

**MCP resources for entities/memories (@-mentionable) and recall/timeline prompts for Claude Code native commands**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-16T02:41:59Z
- **Completed:** 2026-03-16T02:46:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- MCP resources expose entities and recent memories as @-mentionable items in Claude Code
- MCP prompts register recall and timeline as native /mcp__memwright__ commands
- Graph None guard prevents crashes when NetworkX is unavailable
- Full regression: 253 tests pass including 21 new MCP server tests

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests for MCP resources and prompts** - `2ec225c` (test)
2. **Task 1 (GREEN): Implement MCP resources and prompts** - `ac68031` (feat)

## Files Created/Modified
- `agent_memory/mcp/server.py` - Added _build_handlers, list_resources, read_resource, list_resource_templates, list_prompts, get_prompt
- `tests/test_mcp_server.py` - 21 tests: 10 tool handlers, 7 resources, 4 prompts

## Decisions Made
- Extracted _build_handlers as standalone function returning dict of async handlers for direct test invocation
- read_resource returns plain JSON string; MCP SDK auto-wraps in TextResourceContents
- Recent memories limited to 50 most recent active (via mem.search(limit=50))

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed pytest-asyncio for async test support**
- **Found during:** Task 1 RED phase
- **Issue:** pytest.mark.asyncio unrecognized without pytest-asyncio package
- **Fix:** pip install pytest-asyncio
- **Verification:** All async tests run correctly
- **Committed in:** part of task commits

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor dependency addition. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MCP server now has full resources + prompts + tools
- Ready for plan 03-02 (distribution/packaging)

---
*Phase: 03-mcp-enhancements-and-distribution*
*Completed: 2026-03-16*
