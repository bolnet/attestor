# Rename Plan: memwright / agent-memory → attestor

**Branch:** `enterprise-security-hardening`
**Target release:** `attestor` v3.0.0
**Current state:** v2.0.7 on PyPI as `memwright`; GitHub repo `bolnet/attestor` (already renamed); PyPI still `memwright`; Anthropic marketplace plugin submitted as `agent-memory` (pending approval).

---

## 0. Strategy decisions (decide these before any code)

### D1. Hard rename (v3.0.0), not soft rename

- Package name becomes `attestor` on PyPI as a **new distribution**.
- Internal Python package becomes `attestor` (was `agent_memory`).
- `memwright` v3.0.0 is published as a **thin meta-package** — `Requires-Dist: attestor` + `DeprecationWarning` on import — for two releases (v3.0, v3.1) then retired.
- `agent-memory` CLI entry point is retained for two releases as an alias of `attestor` CLI.
- **Rationale:** the rebrand is strategic positioning (audit-grade, Attestor brand, attestor.dev). A permanent aliasing layer is drag. Two-release deprecation window is industry standard and respects existing installs.

### D2. Dual-read runtime compat (not dual-write)

- Default store path becomes `~/.attestor/` (new).
- CLI + hooks read `ATTESTOR_PATH` first, fall back to `MEMWRIGHT_PATH` with a one-time deprecation warning.
- If `~/.attestor/` does not exist but `~/.memwright/` does, emit clear warning + point to `attestor migrate` subcommand.
- **No auto-migration** of user data — explicit `attestor migrate` is safer than surprise moves.
- **Rationale:** user stores may contain months of audit-relevant memories. Silent moves are unacceptable for an "audit-grade" product.

### D3. Atomic slice commits, not one giant rename

Each phase below is a single logical commit that leaves tests green. No intermediate broken state. Seven phases total.

### D4. External identifiers migrated in parallel, not serially

PyPI publish + MCP registry update + marketplace resubmit happen together in Phase 7, after Phases 0–6 land on main. Doing them serially risks weeks of broken external state.

### D5. PyPI name `attestor` must be secured first

Before any code change: `pip search attestor` / PyPI check. If taken, fallback to `attestor-memory` or coordinate with current owner. **This is a go/no-go gate for the entire plan.**

---

## 1. Blast radius summary (from full audit)

| Category | Files touched | Breaking for users? | Notes |
|---|---|---:|---|
| Python package dir (`agent_memory/` → `attestor/`) | ~45 .py files moved | ✅ | `git mv` preserves history |
| Imports (`from agent_memory import …`) | 150+ occurrences across 80+ files | ✅ | Single sed pass |
| `pyproject.toml` (name, packages, scripts, URLs) | 1 | ✅ | Distribution name + CLI entry |
| CLI prog name, help text, logs | ~30 strings in `cli.py`, `init_wizard.py`, `api.py` | 🟡 | Cosmetic + `prog=` arg |
| Runtime config paths (`~/.memwright/`) | ~15 hardcoded occurrences | ✅ | Centralize to one helper |
| Env vars (`MEMWRIGHT_PATH`, `MEMWRIGHT_DATA_DIR`) | ~12 occurrences | ✅ | Dual-read during deprecation |
| MCP URIs (`memwright://entity/...`) | 8 in `mcp/server.py` | ✅ | Update clients + `.mcp.json` |
| Tests (`from agent_memory import …`) | 60+ files | Internal | Same sed pass |
| Infra (Docker env, Terraform vars, container prefix) | 3 Dockerfiles + .tf + `infra/docker.py` | ✅ | Deployed services require coordinated restart |
| Docs (`README.md`, `CLAUDE.md`, `INSTALL.md`, `docs/index.html`, install/uninstall slash commands, demo `.tape` + `.md`) | 200+ text occurrences | N/A | Marketing only — no runtime impact |
| External: PyPI, MCP registry, Anthropic marketplace, GitHub Pages URL | 4 external systems | ✅ | Out-of-repo steps |

**Total in-repo files touched:** ~150
**Total occurrences:** ~650

---

## 2. Seven-phase execution plan

Each phase = one atomic commit. Tests must pass at the end of every phase.

### Phase 0 — Prep + centralize (no user-visible change)

**Goal:** collapse all hardcoded paths/names into single constants so subsequent phases are one-line changes.

Changes:
- New `agent_memory/_branding.py`:
  ```python
  PACKAGE_NAME = "attestor"       # new canonical name
  LEGACY_NAME = "memwright"       # for back-compat warnings
  DEFAULT_STORE_DIRNAME = ".attestor"
  LEGACY_STORE_DIRNAME = ".memwright"
  ENV_STORE_PATH = "ATTESTOR_PATH"
  LEGACY_ENV_STORE_PATH = "MEMWRIGHT_PATH"
  MCP_URI_SCHEME = "attestor"
  LEGACY_MCP_URI_SCHEME = "memwright"
  ```
- New `agent_memory/_paths.py`:
  ```python
  def resolve_store_path() -> Path:
      """Dual-read ATTESTOR_PATH → MEMWRIGHT_PATH → ~/.attestor (new) → ~/.memwright (warn)."""
  ```
- Refactor every hardcoded `"~/.memwright"` / `os.environ["MEMWRIGHT_PATH"]` to use the helper. **No behavior change yet** — helper still returns `~/.memwright` path.
- Refactor every hardcoded `memwright://` URI to use `LEGACY_MCP_URI_SCHEME`.

Files touched: `cli.py`, `api.py`, `ui/app.py`, `hooks/*.py`, `context.py`, `mcp/server.py`, `locomo.py` (cache dir), `infra/docker.py`, `store/_extras.py`.

Commit: `refactor: centralize package name, store path, env var, and MCP scheme constants`

Gate: `pytest tests/ -v` green.

### Phase 1 — Python package directory + imports

**Goal:** `agent_memory/` → `attestor/` with all imports updated.

Steps (all in one commit):
1. `git mv agent_memory attestor` (preserves history)
2. Bulk import rewrite:
   ```bash
   grep -rl "agent_memory" --include="*.py" | xargs sed -i '' \
       -e 's/from agent_memory\./from attestor./g' \
       -e 's/from agent_memory import/from attestor import/g' \
       -e 's/^import agent_memory$/import attestor/g' \
       -e 's/^import agent_memory\./import attestor./g'
   ```
3. Update `pyproject.toml`:
   - `[tool.poetry] packages = [{ include = "attestor" }]`
   - `[project.scripts] attestor = "attestor.cli:main"` + keep `agent-memory` + `memwright` as aliases pointing at `attestor.cli:main`
   - **Do not change `[project] name` yet** — distribution name change happens in Phase 7.
4. Run `pytest tests/ -v`.

Commit: `refactor(pkg): rename python package agent_memory → attestor`

Gate: all 600+ tests green. No import errors. MCP server, CLI, hooks, UI boot.

### Phase 2 — User-facing strings, help, docstrings, logs

**Goal:** all user-facing text says "Attestor" (not "Memwright" or "AgentMemory").

Changes:
- `attestor/__init__.py` docstring
- `attestor/cli.py:prog="attestor"` + help epilog
- `attestor/ui/__init__.py` docstring
- `attestor/mcp/server.py` docstring
- `attestor/init_wizard.py` print strings
- Export filenames: `memwright-export.json` → `attestor-export.json` (`ui/app.py:186,217`)
- Error messages that mention `pip install "memwright[...]"` — update to `attestor` with footnote that old name still works during deprecation

Keep the `AgentMemory` class name (for now — breaking it is a separate decision; see §6 Open Questions).

Commit: `refactor(strings): update user-facing text to Attestor brand`

Gate: visual smoke test — boot CLI, MCP, UI — every prompt/help/banner says Attestor.

### Phase 3 — Runtime paths + env vars (dual-read compat)

**Goal:** new installs use `~/.attestor/`; existing installs keep working with deprecation warnings.

Changes in `attestor/_paths.py::resolve_store_path()`:
1. If `$ATTESTOR_PATH` set → use it.
2. Else if `$MEMWRIGHT_PATH` set → use it + emit `DeprecationWarning("MEMWRIGHT_PATH is deprecated, use ATTESTOR_PATH")`.
3. Else if `~/.attestor/` exists → use it.
4. Else if `~/.memwright/` exists and `~/.attestor/` does not → use `~/.memwright/` + emit a one-time user-facing warning pointing at `attestor migrate`.
5. Else → create `~/.attestor/` (new default).

Add `attestor migrate` subcommand:
- Reads `~/.memwright/` contents
- Copies (not moves) to `~/.attestor/` atomically
- Verifies byte-equal on documents.db, chroma, graph
- Leaves `~/.memwright/` intact with a `MIGRATED_TO_ATTESTOR.txt` breadcrumb
- Supports `--dry-run`, `--force`, `--source`, `--dest`

Same logic for env vars (`ATTESTOR_DATA_DIR` ← `MEMWRIGHT_DATA_DIR`) and cache dir (`~/.cache/attestor/` ← `~/.cache/memwright/`).

Tests:
- `tests/test_paths.py` — all 5 resolution branches
- `tests/test_migrate.py` — fresh `~/.memwright/` → migrated store loadable

Commit: `feat(paths): switch default store to ~/.attestor with MEMWRIGHT_PATH back-compat + attestor migrate`

Gate: tests green. Manual: install fresh store, confirm `~/.attestor/` created. Install with existing `~/.memwright/` present, confirm warning + migrate works.

### Phase 4 — MCP surface

**Goal:** MCP resource URIs, server name, and plugin metadata use `attestor://`.

Changes:
- `attestor/mcp/server.py`: `memwright://entity/...` → `attestor://entity/...` (8 occurrences, lines 36–105). Accept both schemes on read for back-compat for 2 releases.
- MCP server metadata `name="attestor"`
- `.claude-plugin/plugin.json`: `"name": "attestor"`, update description
- `.claude-plugin/marketplace.json`: update catalog entry
- `.mcp.json`: server key `attestor` + command `attestor mcp` (keep `memwright` as alias for one release)
- `hooks/hooks.json`: hook names may reference old binary — verify
- `skills/mem-*/SKILL.md`: rename files `skills/attestor-recall/SKILL.md` etc. + update content

Commit: `refactor(mcp): rename MCP surface (URIs, server name, plugin manifest) to attestor`

Gate: MCP server boots; `mcp__memory__memory_recall` equivalent still works via new tool names; plugin loads in Claude Code.

### Phase 5 — Infra / Docker / Terraform

**Goal:** deployable services use `attestor` names.

Changes:
- `attestor/infra/aws_openarangodb/Dockerfile:24` — `ENV ATTESTOR_DATA_DIR=/data/attestor` (accept `MEMWRIGHT_DATA_DIR` via dual-read startup script)
- `docker-compose.yml` — same
- `ecs.tf:151` — env var, output names
- `attestor/infra/docker.py:23` — `_CONTAINER_PREFIX = "attestor-"`
- `.dockerignore` — add `.attestor/`, keep `.memwright/` for one release
- `.github/workflows/docker.yml` — image tags, build args

**Breaking for already-deployed services.** Document explicitly in CHANGELOG + release notes:
> Users with deployed App Runner / ECS / Cloud Run services: rotate `MEMWRIGHT_DATA_DIR` → `ATTESTOR_DATA_DIR` in environment config during your next deploy. Data volumes mounted at the old path will continue to be read; new data writes to the new path only if the env var is updated.

Commit: `refactor(infra): Attestor branding in Docker, Terraform, and CI`

Gate: local docker-compose up; Terraform plan clean on fresh state.

### Phase 6 — Docs, README, slash commands, demos

**Goal:** all marketing + docs say Attestor.

Changes:
- `README.md` (40+ refs) — regenerate masthead SVG reference, tagline, install snippets, roadmap section
- `CLAUDE.md` — top-of-file title; package + import refs
- `docs/INSTALL.md` — binary name, paths
- `docs/index.html` — hero, meta, OG tags (also satisfies roadmap Section 1 repositioning: swap hero to *"What did the agent know, and when did it know it?"* + keywords `auditable / deterministic / bitemporal`)
- `docs/demo/*.md` + `*.tape` — CLI commands in recordings
- `commands/install-agent-memory.md` → rename to `commands/install-attestor.md` + update content (keep old filename as a thin redirect that invokes the new one, for users who already know the old slash command name)
- `commands/uninstall-agent-memory.md` → `commands/uninstall-attestor.md`
- `.planning/install-wizard-refinements.md` — update to new paths
- Privacy policy URL: `bolnet.github.io/agent-memory/privacy.html` → new URL (see Phase 7)
- SVG files under `docs/` — update any embedded text

Regenerate the masthead if it bakes "MEMWRIGHT" into the SVG (docs/timeline.svg, docs/architecture.svg — check and replace).

Commit: `docs: rename Memwright → Attestor across all documentation`

Gate: visual diff of rendered README on GitHub; landing page smoke test locally.

### Phase 7 — Release + external migrations (coordinated, same day)

**Goal:** publish v3.0.0 on all external surfaces.

Sequence (single day, windowed):

1. **Register PyPI name** `attestor` (if not already claimed). Gate: if unavailable, pause plan and resolve.
2. **Bump version** to `3.0.0` in `pyproject.toml`; `name = "attestor"`.
3. **Build + publish `attestor==3.0.0`** via existing OIDC GitHub Action.
4. **Publish `memwright==3.0.0`** as a meta-package from a sibling `packaging/memwright-shim/` directory:
   - `pyproject.toml`: `name = "memwright"`, `dependencies = ["attestor>=3.0.0"]`
   - `memwright/__init__.py`:
     ```python
     import warnings
     warnings.warn("`memwright` is renamed to `attestor`. Update: pip install attestor", DeprecationWarning, stacklevel=2)
     from attestor import *  # type: ignore
     ```
5. **MCP registry** — register `io.github.bolnet/attestor`. Keep `io.github.bolnet/memwright` active pointing at same server for one release.
6. **Anthropic marketplace plugin** — withdraw + resubmit as `attestor` if current submission is still pending; otherwise update existing entry after approval.
7. **GitHub Pages** — move `bolnet.github.io/agent-memory/` → new domain `attestor.dev` (matches MEMORY.md domain strategy). Add redirect at old URL.
8. **CHANGELOG.md** v3.0.0 entry with explicit migration guide (see §5 below).
9. **Tag and release** `v3.0.0` on GitHub.
10. **Announcement** — README badge, social post (coordinates with roadmap Week 7 benchmark splash).

Commit: `chore(release): v3.0.0 Attestor rebrand`

Gate:
- `pip install attestor` works on clean venv
- `pip install memwright` still works, emits deprecation warning, and `from memwright import AgentMemory` returns the same class as `from attestor import AgentMemory`
- `attestor doctor` passes against a fresh store
- `attestor doctor` passes against a migrated `~/.memwright/` store

---

## 3. User migration path (documented in release notes)

### For library users

```bash
pip uninstall memwright
pip install attestor
```

Imports:
```diff
- from agent_memory import AgentMemory
+ from attestor import AgentMemory
```

Temporary: `from memwright import AgentMemory` still works via shim for two releases but emits `DeprecationWarning`.

### For CLI users

```bash
# both still work for v3.0 and v3.1
attestor doctor    # new canonical
agent-memory doctor   # deprecated alias
memwright doctor      # deprecated alias
```

### For Claude Code users with existing stores

```bash
attestor migrate   # copies ~/.memwright/ → ~/.attestor/ non-destructively
```

Or set `MEMWRIGHT_PATH=/path/to/old/store` and Attestor will read from it with a warning.

### For deployed services

Add `ATTESTOR_DATA_DIR` environment variable alongside existing `MEMWRIGHT_DATA_DIR`. Image `v3.0.0` reads both. Drop `MEMWRIGHT_*` env vars by v3.2.

---

## 4. Backwards compatibility matrix

| Surface | v2.x (memwright) | v3.0–v3.1 (both) | v3.2+ (attestor only) |
|---|---|---|---|
| Python import `agent_memory` | ✅ works | ❌ removed (Phase 1) | ❌ |
| Python import `memwright` | ❌ never existed | ✅ shim + warning | ❌ |
| Python import `attestor` | ❌ | ✅ canonical | ✅ canonical |
| CLI `memwright` | ✅ | ✅ alias | ❌ |
| CLI `agent-memory` | ✅ | ✅ alias | ❌ |
| CLI `attestor` | ❌ | ✅ canonical | ✅ |
| Env `MEMWRIGHT_PATH` | ✅ | ✅ dual-read + warn | ❌ |
| Env `ATTESTOR_PATH` | ❌ | ✅ canonical | ✅ |
| `~/.memwright/` auto-read | default | fallback + warn | ❌ |
| MCP URI `memwright://` | ✅ | ✅ accepted (read) | ❌ |
| MCP URI `attestor://` | ❌ | ✅ canonical (emit) | ✅ |

---

## 5. CHANGELOG v3.0.0 entry (draft)

> ### Breaking
> - Renamed Python package `agent_memory` → `attestor`. Update imports.
> - Renamed PyPI distribution `memwright` → `attestor`. `pip install memwright` now installs a thin shim that depends on `attestor` and emits a DeprecationWarning. The shim will be removed in v3.2.
> - Renamed default store path `~/.memwright/` → `~/.attestor/`. Existing stores continue to work (Attestor auto-detects) but run `attestor migrate` to move them.
> - MCP resource URIs changed from `memwright://` → `attestor://`. Old scheme is still read for one release.
> - Docker env var `MEMWRIGHT_DATA_DIR` deprecated in favor of `ATTESTOR_DATA_DIR`. Both are read in v3.x.
>
> ### New
> - `attestor migrate` CLI subcommand for non-destructive store migration.
> - `attestor` binary (replaces `memwright` / `agent-memory` as canonical entry point).
>
> ### Migration
> See [MIGRATING.md](./MIGRATING.md) for the full checklist.

---

## 6. Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyPI name `attestor` taken | **LOW/UNKNOWN** | Blocks entire plan | **D5 gate** — check first. Fallbacks: `attestor-memory`, `attestor-io`. Negotiate or choose alt. |
| User runtime store at `~/.memwright/` silently ignored | MEDIUM | Data loss perception | Phase 3 dual-read + prominent CLI warning + `attestor migrate` |
| Deployed services break on Docker env var rename | HIGH | Service downtime | Phase 5 accepts both env vars; release notes call out rotation before v3.2 |
| Anthropic marketplace submission still in review | HIGH | Gets approved under old name, then breaks | Withdraw+resubmit as `attestor`, accept delayed approval |
| MCP registry has two entries temporarily | LOW | Confusion | Acceptable for 1 release; clearly mark `memwright` entry as deprecated |
| Intermediate commit breaks imports | LOW | CI red | Phases are atomic; bulk sed + `pytest` gate after each |
| SVG masthead bakes text | LOW | Visual drift | Grep SVGs for "MEMWRIGHT"; regenerate if needed |
| `AgentMemory` class name still says "Agent" | MEDIUM | Inconsistent brand | **Open question** — see §6 below |
| Users with deep deprecated code find shim gone in v3.2 | LOW | Support load | Two-release window is standard; CHANGELOG explicit |
| Benchmarks (`locomo.py`, `mab.py`, `longmemeval.py`) output JSON that references old name | LOW | Reproducibility | Attach `{"system": "attestor"}` field in output; include both names in result metadata for continuity |

---

## 7. Open design questions (decide before Phase 2)

1. **Rename the main class `AgentMemory` → `Attestor` (or `AttestorStore`)?** Arguments:
   - Pro: consistent brand, clean API, matches "audit-grade" positioning.
   - Con: biggest user-visible import change; `AgentMemory` is semantic; may confuse people looking for "an attestor".
   - Recommendation: **keep `AgentMemory` for v3.0**. Add `Attestor` as an alias in `attestor/__init__.py`. Re-evaluate at v4.0.
2. **Keep `agent-memory` CLI alias forever, or deprecate?** Recommend deprecate in v3.2 along with `memwright`.
3. **Should `memwright` PyPI name be transferred to Anthropic / reserved permanently to prevent squatting?** Yes — keep the shim package alive even past v3.2 as a squatter-defense measure (no code, just the name).
4. **Rebrand timing vs. roadmap Week 1 repositioning.** Phase 6 (docs) overlaps with roadmap Section 1 homepage rewrite. **Do them together** — single PR, single narrative.
5. **Database filename `memwright.db` → `attestor.db` inside stores?** Recommend keep the file name stable (users should not care) and rename only the default *store directory* name. Less surface area, less migration pain.

---

## 8. Timeline (calendar estimate)

Assumes one engineer working full-time on the rename.

| Phase | Est. duration | Parallelizable? |
|---|---|---|
| Phase 0 — prep constants + helpers | 0.5 day | No |
| Phase 1 — package dir + imports | 1 day | No |
| Phase 2 — strings + help | 0.5 day | Yes, with Phase 3 |
| Phase 3 — paths + env + `migrate` | 1.5 days | Yes, with Phase 2 |
| Phase 4 — MCP + plugin manifest | 0.5 day | Yes, after Phase 1 |
| Phase 5 — infra + Docker + CI | 0.5 day | Yes, after Phase 1 |
| Phase 6 — docs + commands + demos | 1 day | Yes, after Phase 2 |
| Phase 7 — release + external | 0.5 day | No — must be last |
| Verification + bugfix buffer | 1 day | No |
| **Total** | **~1 calendar week** | |

Lines up with roadmap Week 1–2 window for repositioning (Phase 6 absorbs Section 1 homepage rewrite).

---

## 9. Verification checklist (pre-release gate)

- [ ] `pytest tests/ -v` — all green
- [ ] `attestor doctor` — green on fresh `~/.attestor/`
- [ ] `attestor doctor` — green on store migrated from `~/.memwright/` via `attestor migrate`
- [ ] `MEMWRIGHT_PATH=/tmp/old-store attestor doctor` — green + deprecation warning emitted once
- [ ] `pip install attestor` on clean venv — binary `attestor` on PATH, import works
- [ ] `pip install memwright` on clean venv — installs shim, `from memwright import AgentMemory` works + warns, binary `memwright` on PATH as alias
- [ ] MCP server boots; Claude Code plugin loads; `memory_recall` returns results
- [ ] LOCOMO + MAB + (if landed) LongMemEval smoke runs finish — JSON output tags `system: "attestor"`
- [ ] Landing page (`docs/index.html`) contains all three roadmap keywords: `auditable`, `deterministic`, `bitemporal`
- [ ] No file under `attestor/` still contains the literal string `memwright` except in:
   - `attestor/_branding.py` (LEGACY_NAME)
   - `attestor/_paths.py` (back-compat)
   - `attestor/mcp/server.py` (URI back-compat)
   - `attestor/cli.py` (migrate subcommand)
   - CHANGELOG / release notes

---

## 10. Rollback plan

If any gate fails and rollback is needed:

- **Before Phase 7 (external publish):** `git revert` the phase commit(s); nothing external is affected.
- **After Phase 7:** cannot unpublish PyPI releases, but can publish `attestor==3.0.1` that is effectively a no-op (or yanks the release). Keep `memwright==2.0.7` stable on PyPI by not deleting it. Users on v2.x are unaffected.
- **User data:** `attestor migrate` is non-destructive — users can always fall back to `~/.memwright/` by setting `MEMWRIGHT_PATH`.

---

## 11. Immediate next action

Three choices, in order of value:

1. **Check PyPI name availability for `attestor`** (D5 gate). Must happen before any code.
2. Draft and commit **Phase 0** (centralize constants + helpers) — no behavior change, safe to land immediately.
3. Confirm **Open Question 1** (class rename) with the user before Phase 2 touches docstrings.

Recommended: do 1 and 2 today in parallel; decide 3 before Phase 2.
