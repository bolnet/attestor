---
phase: 02-claude-code-plugin-and-auto-capture
plan: 02
subsystem: hooks
tags: [claude-code, hooks, mcp, cli, lifecycle]

requires:
  - phase: 01-zero-infrastructure-backends
    provides: AgentMemory API (add, recall, recall_as_context, search)
provides:
  - SessionStart hook injecting memories into Claude Code sessions
  - PostToolUse hook capturing Write/Edit/Bash observations as memories
  - Stop hook summarizing session with file change and command counts
  - hooks.json registering all 3 lifecycle events
  - CLI `memwright hook {name}` entry points for hooks
  - CLI `memwright mcp` zero-config MCP server subcommand
affects: [02-claude-code-plugin-and-auto-capture]

tech-stack:
  added: []
  patterns: [stdin-stdout-json-hooks, cwd-based-project-scoping, silent-failure-on-error]

key-files:
  created:
    - agent_memory/hooks/__init__.py
    - agent_memory/hooks/session_start.py
    - agent_memory/hooks/post_tool_use.py
    - agent_memory/hooks/stop.py
    - .claude-plugin/hooks/hooks.json
    - tests/test_hooks.py
  modified:
    - agent_memory/cli.py

key-decisions:
  - "20K token budget for SessionStart context injection (per user decision)"
  - "Per-project memory scoping via {cwd}/.memwright/ directory"
  - "Rule-based session summary (no LLM) counting file changes and commands"
  - "Read-only Bash commands (cat/ls/grep/etc.) silently skipped by PostToolUse"

patterns-established:
  - "Hook pattern: handle(payload) -> dict with stdin/stdout JSON main()"
  - "Silent failure: all hooks return empty response on any error, never block session"
  - "CWD-scoped store: {cwd}/.memwright/ for per-project memory isolation"

requirements-completed: [HOOK-01, HOOK-02, HOOK-03, HOOK-04, HOOK-05]

duration: 3min
completed: 2026-03-16
---

# Phase 2 Plan 2: Claude Code Hooks and MCP Subcommand Summary

**3 lifecycle hooks (SessionStart/PostToolUse/Stop) with hooks.json registration and zero-config `memwright mcp` CLI subcommand**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T02:25:57Z
- **Completed:** 2026-03-16T02:29:36Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- SessionStart hook injects relevant memories with 20K token budget into Claude Code sessions
- PostToolUse hook captures Write/Edit/Bash tool observations as memories (ignores read-only tools)
- Stop hook generates rule-based session summaries with file change and command counts
- hooks.json registers all 3 events with correct commands and timeouts
- `memwright mcp` starts MCP server zero-config (no positional path, uses $MEMWRIGHT_PATH or ~/.memwright)
- `memwright hook {session-start|post-tool-use|stop}` CLI entry points work

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SessionStart and PostToolUse hooks** - `2705444` (feat)
2. **Task 2: Implement Stop hook, mcp subcommand, hook CLI, hooks.json** - `e3b3dd2` (feat)

_Note: TDD tasks had RED (fail) then GREEN (pass) phases within each commit._

## Files Created/Modified
- `agent_memory/hooks/__init__.py` - Package init (empty)
- `agent_memory/hooks/session_start.py` - SessionStart hook: recalls memories, returns additionalContext
- `agent_memory/hooks/post_tool_use.py` - PostToolUse hook: captures tool observations as memories
- `agent_memory/hooks/stop.py` - Stop hook: rule-based session summary from recent memories
- `.claude-plugin/hooks/hooks.json` - Registers 3 hooks with events, commands, timeouts
- `agent_memory/cli.py` - Added `mcp` and `hook` subcommands
- `tests/test_hooks.py` - 22 tests covering all hooks and CLI subcommands

## Decisions Made
- 20K token budget for SessionStart context injection (per user decision from context)
- Per-project memory scoping via {cwd}/.memwright/ directory
- Rule-based session summary (no LLM calls) -- counts file changes and commands
- Read-only Bash commands (cat/ls/head/tail/echo/grep/find/rg/pwd/which/type) silently skipped
- All hooks fail silently returning empty responses -- never block Claude Code sessions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 3 lifecycle hooks implemented and registered
- .mcp.json from plan 02-01 can now invoke `memwright mcp` successfully
- hooks.json ready for Claude Code plugin integration
- 22 tests provide regression coverage

---
*Phase: 02-claude-code-plugin-and-auto-capture*
*Completed: 2026-03-16*
