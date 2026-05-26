# Attestor — Native Claude Code Plugin + Per-Project Memory Isolation

**Date:** 2026-05-25
**Status:** Design approved, pending implementation plan

## Problem

Attestor already ships Claude Code integration surfaces (an MCP server, three
hooks, a skill, an install wizard), but two things are missing for it to feel
*native*:

1. **No plugin packaging.** There is no `.claude-plugin/plugin.json` and no
   marketplace, so installation is a manual wizard rather than
   `/plugin install attestor`.
2. **No per-project memory isolation.** The single most important requirement.
   Today every Claude Code project shares one store and one namespace, so
   memory bleeds across unrelated projects.

### The isolation requirement (non-negotiable)

Memory must be **scoped to the current working directory**. The folder Claude
Code runs in is the **default project and the home for that project's memory**.
Memory must **never** leak across projects. A project is treated as a
first-class tenant — the way Attestor treats a `user:` namespace.

## What already exists (verified in code, 2026-05-25)

CLAUDE.md is stale on isolation. Verified firsthand:

| Capability | Status | Evidence |
|---|---|---|
| Postgres multi-tenancy + Row-Level Security | Exists | `store/schema.sql` (`users`→`projects`→`memories(scope, project_id)`); RLS on every tenant table keyed to `attestor.current_user_id`, set at `store/postgres_backend.py:157` |
| Pinecone per-namespace isolation | Exists | threaded via `AgentMemory.add(namespace=…)` |
| Neo4j namespace scoping | Exists (CLAUDE.md stale) | nodes/edges carry `namespace`; composite `(key, namespace)` unique constraint; BFS filters `coalesce(n.namespace,'default')=$ns` — `store/neo4j_backend.py:170-240` |
| Identity by external id | Exists | `core/identity_service.py`: `ensure_user(external_id)`, `find_user_by_external_id`, `create_project` |
| **Derive project identity from `cwd`** | **Missing** | hooks call `recall()`/`add()` with no namespace → all `default`; MCP `memory_add` defaults `namespace="default"`; `_paths.resolve_store_path()` → one shared `~/.attestor` |
| **Plugin packaging** | **Missing** | no `.claude-plugin/plugin.json`, no marketplace manifest |

The cross-project bleed exists **only** because the `cwd → tenant` mapping is
never made. The isolation primitives are already built and tested.

## Decisions

- **"Native" = Claude Code plugin packaging + marketplace.** (Chosen over
  zero-infra default and full-hook-coverage as the primary framing.)
- **Isolation realized via project namespace across the canonical stack** —
  NOT a new embedded backend, NOT per-project physical DBs. Keep the single
  canonical PG+Pinecone+Neo4j stack.
- **Project = git root if present, else cwd.** Subdirectories of one repo share
  memory; unrelated folders never do.
- **Project maps to an RLS tenant** via `ensure_user(external_id=project_root)`,
  reusing the already-tested user-level Row-Level Security so the database
  itself refuses cross-project reads/writes. Zero schema change. Writes use
  `scope='project'`.

## Design

### Part 1 — Project = current directory = RLS tenant (isolation fix)

1. **`attestor/_project.py`** (new, small): `resolve_project_id(cwd) ->
   project_root`. Walk up for a `.git` dir; fall back to the absolute `cwd`.
   Pure function, unit-testable, no I/O beyond filesystem stat.
2. **Tenant mapping:** given a project root, `ensure_user(external_id=root)` and
   set `attestor.current_user_id` to that user for the connection. RLS then
   hard-isolates. Memory writes use `scope='project'`.
3. **Hooks:** `hooks/session_start.py` (recall), `hooks/post_tool_use.py`
   (add), `hooks/stop.py` (summary) each already receive `payload["cwd"]`;
   derive the tenant there and pass it down. Preserve existing timeout /
   never-crash-the-host contract.
4. **MCP server (`mcp/server.py`):** derive the tenant from the server's launch
   `cwd` and use it as the default tenant for every tool, replacing the
   `namespace="default"` default. Explicit per-call namespace still overrides
   (for intentional user/global-scope facts).
5. **Defense-in-depth:** also stamp `namespace=project:<root>` so Pinecone and
   Neo4j are scoped even on any path that bypasses RLS.

### Part 2 — Plugin packaging

6. **`.claude-plugin/plugin.json`** declaring the bundled MCP server
   (`attestor mcp`), hooks (SessionStart / PostToolUse / Stop → `attestor hook
   …`), the skill, and the `/install-attestor` command.
7. **Marketplace manifest** so users do `/plugin marketplace add bolnet/attestor`
   then `/plugin install attestor`.
8. **Trim the install wizard:** the plugin auto-wires MCP + hooks, so install
   only interviews for backend connection (canonical stack must be reachable —
   local Docker or cloud) + embedding provider. No more hand-writing
   `settings.json`.

## Testing

The isolation tests are the critical deliverable:

- Two distinct `cwd`s → two tenants → assert **zero cross-read** across
  Postgres **and** Pinecone **and** Neo4j.
- RLS denial test: a connection scoped to tenant A cannot read tenant B rows
  even with a missing `WHERE`.
- Git-root-vs-subdir test: subdir of a repo resolves to the same tenant as the
  repo root; an unrelated sibling folder resolves to a different tenant.
- Hook contract tests: hooks still never crash the host and honor the timeout.
- `resolve_project_id` unit tests (git present / absent, nested, symlinks).

Target: maintain 80%+ coverage on new modules.

## Effort estimate

- Isolation core (`_project.py` + wire 3 hooks + MCP): ~1–1.5 days
- Isolation tests (critical path): ~0.5 day
- Plugin packaging + marketplace + install trim: ~1 day
- Docs + fix stale CLAUDE.md Neo4j claim: ~0.5 day

≈ **3–3.5 days**, isolation core + tests on the critical path.

## Risks / open points

- **Not zero-infra.** Keeps the canonical stack; the plugin still needs
  PG+Pinecone+Neo4j reachable (local Docker or cloud). Install-and-go without
  Docker would require reintroducing an embedded backend (declined).
- **MCP `cwd` assumption.** Relies on Claude Code launching the MCP server with
  `cwd = workspace`. If one global server serves multiple workspaces in a
  single session, per-call tenant derivation is needed instead — verify launch
  behavior during planning.
- **Legacy `default` data.** Existing memories in the shared `~/.attestor`
  remain under `default` and do not auto-migrate into project tenants. Clean
  break for new installs.

## Out of scope

- Reintroducing an embedded/zero-infra backend.
- Per-project physical databases (separate PG schema / Neo4j db / Pinecone
  index per project).
- Expanding hook coverage to UserPromptSubmit / PreToolUse / PreCompact /
  SessionEnd (possible later; not part of this work).
