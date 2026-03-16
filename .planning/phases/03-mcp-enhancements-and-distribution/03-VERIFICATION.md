---
phase: 03-mcp-enhancements-and-distribution
verified: 2026-03-16T03:00:00Z
status: human_needed
score: 11/12 must-haves verified
re_verification: false
human_verification:
  - test: "Confirm memwright 2.0.0 is published to PyPI"
    expected: "pip install memwright==2.0.0 succeeds from a clean environment"
    why_human: "Cannot reach PyPI from this environment to verify live publication. pyproject.toml is correct at 2.0.0 but the actual publish step is a manual or CI action outside the codebase."
  - test: "Confirm MCP Registry listing reflects v2.0.0 or latest"
    expected: "io.github.bolnet/memwright on the MCP Registry shows v2.0.0 or redirects to pip install"
    why_human: "DIST-04 states auto-sync from PyPI post-publish. Whether the registry has updated depends on publication having occurred. Cannot verify live registry state programmatically."
---

# Phase 3: MCP Enhancements and Distribution Verification Report

**Phase Goal:** Memwright is published with rich MCP integration (resources, prompts, skills) and available via both pip and plugin marketplace
**Verified:** 2026-03-16T03:00:00Z
**Status:** human_needed (11/12 automated checks pass; 2 items require human confirmation)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Entities are listed as MCP resources and readable via resource URI | VERIFIED | `_build_handlers` → `list_resources` guards `_graph is not None` at line 32; entity URIs use `memwright://entity/{key}`; test_list_resources_with_graph passes |
| 2 | Recent memories are listed as MCP resources and readable via resource URI | VERIFIED | `list_resources` calls `mem.search(limit=50)` and creates memory URIs; `read_resource` handles `memwright://memory/{id}`; test_read_resource_memory passes |
| 3 | recall prompt is registered and returns formatted prompt messages | VERIFIED | `list_prompts` returns Prompt(name="recall", arguments=[PromptArgument("query", required=True)]); `get_prompt("recall")` calls `mem.recall(query)` at line 145; test_get_prompt_recall passes and asserts `mem.recall.assert_called_once_with` |
| 4 | timeline prompt is registered and returns formatted prompt messages | VERIFIED | `list_prompts` returns Prompt(name="timeline", arguments=[PromptArgument("entity", required=True)]); `get_prompt("timeline")` calls `mem.timeline(entity)` at line 164; test_get_prompt_timeline passes |
| 5 | All 8 existing tools still work with ChromaDB/NetworkX backends | VERIFIED | Full suite: 253 tests pass. TestToolHandler covers all 8 tools: memory_add, memory_get, memory_recall, memory_search, memory_forget, memory_timeline, memory_stats, memory_health |
| 6 | User can invoke /memwright:mem-recall in Claude Code and get memory search results | VERIFIED | `.claude-plugin/skills/mem-recall/SKILL.md` exists, references `memory_recall` MCP tool, has usage examples |
| 7 | User can invoke /memwright:mem-timeline in Claude Code and get entity history | VERIFIED | `.claude-plugin/skills/mem-timeline/SKILL.md` exists, references `memory_timeline` MCP tool, has usage examples |
| 8 | User can invoke /memwright:mem-health in Claude Code and see component status | VERIFIED | `.claude-plugin/skills/mem-health/SKILL.md` exists, references `memory_health` MCP tool |
| 9 | No Docker-era dependencies (psycopg, neo4j-driver) pulled in by pip install | VERIFIED | `pyproject.toml` `dependencies = ["chromadb>=0.4.0", "networkx>=3.0"]`; grep for neo4j/psycopg returns no matches |
| 10 | Plugin marketplace artifacts exist and are valid (.claude-plugin/plugin.json, marketplace.json, .mcp.json) | VERIFIED | All three files exist, parse as valid JSON, contain required fields (name, version, mcpServers key) |
| 11 | pyproject.toml is v2.0.0 with correct metadata | VERIFIED | version="2.0.0", description updated to zero-config messaging, keywords include "zero-config"/"mcp"/"claude-code"/"plugin", classifier "Development Status :: 4 - Beta", optional-deps extraction/mcp/dev/all all present |
| 12 | pip install memwright and MCP Registry listing reflect published v2.0.0 | NEEDS HUMAN | pyproject.toml is correct at 2.0.0; live publication and registry sync cannot be confirmed programmatically |

**Score:** 11/12 truths verified (12th is publication state — outside codebase scope)

---

## Required Artifacts

### Plan 03-01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `agent_memory/mcp/server.py` | MCP server with resources, prompts, and tools | VERIFIED | 501 lines; exports `create_server`, `run_server`, `_build_handlers`; registers list_resources, read_resource, list_resource_templates, list_prompts, get_prompt via server decorators |
| `tests/test_mcp_server.py` | Tests for MCP resources, prompts, and tool handlers | VERIFIED | 270 lines; 21 tests: TestToolHandler (10), TestResources (7), TestPrompts (4); all pass |

### Plan 03-02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.claude-plugin/skills/mem-recall/SKILL.md` | Skill definition for natural language memory recall | VERIFIED | Contains "mem-recall", references `memory_recall` MCP tool, has examples |
| `.claude-plugin/skills/mem-timeline/SKILL.md` | Skill definition for entity timeline queries | VERIFIED | Contains "mem-timeline", references `memory_timeline` MCP tool, has examples |
| `.claude-plugin/skills/mem-health/SKILL.md` | Skill definition for health check | VERIFIED | Contains "mem-health", references `memory_health` MCP tool |
| `pyproject.toml` | Package metadata with correct deps and version | VERIFIED | version=2.0.0, chromadb+networkx base deps, 4 extras, no psycopg/neo4j |
| `.claude-plugin/plugin.json` | Plugin manifest for Claude Code plugin marketplace | VERIFIED | Valid JSON; name="bolnet/agent-memory", version="0.2.0", entry_point="pip install memwright" |
| `marketplace.json` | Marketplace listing metadata | VERIFIED | Found at `.claude-plugin/marketplace.json` (SUMMARY noted this location); valid JSON; name, description, install_command fields present |
| `.mcp.json` | MCP server configuration | VERIFIED | Valid JSON; `mcpServers.agent-memory` entry with command="memwright", args=["mcp"] |

---

## Key Link Verification

### Plan 03-01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `agent_memory/mcp/server.py` | `agent_memory/graph/networkx_graph.py` | `get_entities()` guarded by `if mem._graph is not None` | WIRED | Line 32: `if mem._graph is not None:` guard; line 33: `mem._graph.get_entities()` call confirmed |
| `agent_memory/mcp/server.py` | `agent_memory/core.py` | `mem.recall()`/`mem.timeline()` in prompt handlers | WIRED | Line 145: `results = mem.recall(query)`; line 164: `memories = mem.timeline(entity)`; both inside `_build_handlers` |

### Plan 03-02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `.claude-plugin/skills/mem-recall/SKILL.md` | `agent_memory/mcp/server.py` | References `memory_recall` MCP tool | WIRED | SKILL.md line 13: `` `memory_recall` MCP tool with `query` parameter``; tool exists in `list_tools()` at server.py line 273 |
| `pyproject.toml` | `agent_memory/` | `packages = ["agent_memory"]` in hatch build config | WIRED | `[tool.hatch.build.targets.wheel]` packages = ["agent_memory"] at line 48 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MCP-01 | 03-01 | MCP resources expose entities as @-mentionable | SATISFIED | `list_resources` creates `memwright://entity/{key}` resources; test_list_resources_with_graph passes |
| MCP-02 | 03-01 | MCP resources expose recent memories as @-mentionable | SATISFIED | `list_resources` creates `memwright://memory/{id}` resources from `mem.search(limit=50)` |
| MCP-03 | 03-01 | MCP prompts register `recall` as native command | SATISFIED | `list_prompts` returns Prompt(name="recall"); `get_prompt` handles "recall" via `mem.recall()` |
| MCP-04 | 03-01 | MCP prompts register `timeline` as native command | SATISFIED | `list_prompts` returns Prompt(name="timeline"); `get_prompt` handles "timeline" via `mem.timeline()` |
| MCP-05 | 03-01 | Existing 8 MCP tools work with new backends | SATISFIED | 253 full-suite tests pass; all 8 tools covered by TestToolHandler |
| SKILL-01 | 03-02 | `/memwright:mem-recall` skill exists | SATISFIED | `.claude-plugin/skills/mem-recall/SKILL.md` present with tool reference and usage |
| SKILL-02 | 03-02 | `/memwright:mem-timeline` skill exists | SATISFIED | `.claude-plugin/skills/mem-timeline/SKILL.md` present with tool reference and usage |
| SKILL-03 | 03-02 | `/memwright:mem-health` skill exists | SATISFIED | `.claude-plugin/skills/mem-health/SKILL.md` present with tool reference and usage |
| DIST-01 | 03-02 | `pip install memwright` works zero-config | SATISFIED (with caveat) | pyproject.toml correct; live PyPI publication needs human confirmation |
| DIST-02 | 03-02 | Plugin marketplace available | SATISFIED | `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `.mcp.json` all present and valid |
| DIST-03 | 03-02 | PyPI package has correct deps (no psycopg/neo4j) | SATISFIED | `dependencies = ["chromadb>=0.4.0", "networkx>=3.0"]`; no legacy deps |
| DIST-04 | 03-02 | MCP Registry listing updated | NEEDS HUMAN | Plan notes auto-sync post-publish; cannot confirm live registry state |

**Orphaned requirements check:** All 12 Phase 3 requirement IDs (SKILL-01/02/03, MCP-01/02/03/04/05, DIST-01/02/03/04) are claimed by plans 03-01 and 03-02. No Phase 3 requirements in REQUIREMENTS.md are orphaned.

---

## Anti-Patterns Found

No anti-patterns detected in phase files. Scanned:
- `agent_memory/mcp/server.py`
- `tests/test_mcp_server.py`
- All three SKILL.md files

No TODO/FIXME/placeholder comments, empty implementations, or stub return values found.

---

## Human Verification Required

### 1. PyPI Publication of v2.0.0

**Test:** From a clean Python environment (not this venv), run `pip install memwright==2.0.0`
**Expected:** Package installs successfully and `import agent_memory` works. No psycopg or neo4j packages pulled in as transitive dependencies.
**Why human:** pyproject.toml is correctly configured at 2.0.0 but the PyPI publish action is external to the codebase. The current MEMORY.md records the latest published version as 0.1.3 (as of the project memory snapshot). Whether 2.0.0 has been published requires checking https://pypi.org/project/memwright/ or running pip install in isolation.

### 2. MCP Registry Sync for v2.0.0

**Test:** Visit https://registry.mcp.io/servers/io.github.bolnet/memwright (or equivalent MCP Registry URL) and confirm the listing references v2.0.0 or the updated install command.
**Expected:** Registry shows updated version / pip install path consistent with v2.0.0 release.
**Why human:** DIST-04 implementation relies on the registry auto-syncing from PyPI post-publish. This is a live external service state that cannot be queried programmatically from this environment.

---

## Gaps Summary

No implementation gaps were found. All code artifacts are substantive (not stubs), all key links are wired, all 12 requirement IDs are covered, and 253 tests pass.

The two human-needed items (DIST-01 live publication, DIST-04 registry sync) are external publication steps — they depend on running `git push` + a PyPI release CI job. Everything in the codebase is correctly prepared for that release. The gap, if any, is operational (publish has not yet been verified live) rather than a code deficiency.

---

_Verified: 2026-03-16T03:00:00Z_
_Verifier: Claude (gsd-verifier)_
