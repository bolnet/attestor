# Code Review — Native Claude Code Plugin + Per-Project Isolation

**Date:** 2026-05-25
**Reviewer:** code-reviewer agent (merge-request review, pre-commit)
**Scope:** this session's changes only (see plan `2026-05-25-attestor-claude-code-native-plugin.md`)
**Verdict at review time:** BLOCK (2 CRITICAL). **After dispositions below: resolved.**

Each finding was verified firsthand against the code before acting (some agent
claims were inaccurate — noted).

## CRITICAL

### C1 — `external_id VARCHAR(256)` overflow degrades the RLS tenant ✅ FIXED
`users.external_id` is `VARCHAR(256)` (`store/schema.sql:19`). The original
`cc-project:<absolute-path>` could overflow on deep paths; Postgres raises
`value too long`, `resolve_tenant` catches it and falls back to
`user_id=None`, silently dropping RLS.
- **Verified:** column width confirmed. Also confirmed the v4 doc read filters
  by `metadata->>'_namespace'` (`_postgres_document.py:256,335,439`), so our
  hooks/MCP (which always pass `namespace`) stay isolated even in the degraded
  path — but losing RLS silently is still unacceptable.
- **Fix:** `project_external_id`/`project_namespace` now hash the root
  (`sha256`, `_project.py:_digest`) → fixed-length (~75 chars), charset-safe
  (also fixes Pinecone namespace `/`-separator risk). Human path preserved in
  the user's `display_name`/`metadata` via `resolve_tenant`.

### C2 — `pagerank()` not namespace-scoped ✅ ADDRESSED (doc corrected)
`pagerank()` takes no namespace (`neo4j_backend.py:341`) — it computes over the
global entity graph.
- **Verified:** confirmed no namespace param. However it only *reorders results
  that are already namespace-filtered* (session_start recalls the project's
  memories, then boosts) — it does **not** surface another project's memories.
  The agent overstated this as a data leak; it is a global ranking *signal*.
- **Disposition:** the real defect was the CLAUDE.md overclaim. Corrected to
  state the PageRank GDS projection is global (ranking-only caveat, no data
  leak). Scoping the GDS projection by namespace is a possible follow-up.

## HIGH

### H1 — stop hook vector `search()` path + RLS pin — NOT A BUG (current call is safe)
The stop hook calls `search(after=..., limit=50)` with `query=None`, which
takes the `list_memories` path (RLS-gated by the preceding `_resolve`), not the
vector branch. The agent acknowledged this. Left as-is; documented constraint.

### H2 — `ensure_user` is a sync DB call in an async handler — PRE-EXISTING, not fixed
Valid latency note, but the entire `_handle_tool` is already synchronous and
called directly in the async `call_tool`. This is the existing architecture,
not a regression introduced here. Out of scope for this change; flagged for a
future server-wide `asyncio.to_thread` pass.

### H3/H4 — plugin.json `skills`/`commands` paths "missing" — ❌ AGENT WRONG, dismissed
`skills/` and `commands/` exist at the **repo root**, and Claude Code resolves
plugin component paths relative to the plugin root (the dir containing
`.claude-plugin/`), i.e. the repo root. `./skills/` and `./commands/` are
correct. The agent checked `.claude-plugin/skills/` (wrong base). Verified the
dirs exist. No change.

## MEDIUM

- **M1 symlink divergence** ✅ added `test_symlink_resolves_to_canonical_tenant`
  (confirms `resolve()` canonicalization → same tenant).
- **M2 MCP tenant fixed at launch cwd** — documented design (one server per
  workspace). Comment already states it. No change.
- **M3 no lru_cache on `resolve_project_root`** — hooks are per-process (no
  cross-call benefit); MCP resolves once. Negligible. No change.
- **M4 `hooks.json` `$schema`** — matches the shape used by the shipped
  `everything-claude-code/hooks/hooks.json`. No change.

## LOW

- **L1 live test leaves rows** ✅ added `mem.forget(written.id)` cleanup.
- **L2 `configs/attestor.yaml` bench-narrative comments** — real (violates the
  "no bench stats in repo" rule) but **outside this session's diff**. Flagged to
  the user; not touched here.
- **L3 `evals/longmemeval/runner.py` inline `import os`** — **outside this
  session's diff**. Flagged; not touched.

## Post-fix verification
- `tests/test_project_resolution.py`, `test_hook_tenant.py`,
  `test_mcp_tenant_default.py`: **19 passed**; `test_tenant_isolation_live.py`:
  skipped (env-gated).
- Existing MCP/hook suite: green (no regressions).
- `_project.py`: ruff-clean.
