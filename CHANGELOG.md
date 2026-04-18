# Changelog

All notable changes to Attestor (formerly Memwright) are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-04-18

**Attestor rebrand.** `memwright` is now `attestor`. The library, CLI, default store path, MCP URI scheme, and Docker env var all change. v3.x ships compatibility shims for each surface; they are removed in v3.2.

See [MIGRATING.md](./MIGRATING.md) for the full migration checklist.

### Breaking

- **Python package renamed** `agent_memory` → `attestor`. Update imports: `from agent_memory import ...` → `from attestor import ...`.
- **PyPI distribution renamed** `memwright` → `attestor`. `pip install memwright` now installs a thin shim that depends on `attestor` and emits a `DeprecationWarning` on import. The shim is removed in v3.2.
- **Default store path** `~/.memwright/` → `~/.attestor/`. Existing stores at `~/.memwright/` are still auto-detected and read in v3.x; run `attestor migrate` to copy non-destructively to the new location.
- **MCP resource URIs** changed from `memwright://` → `attestor://`. The old scheme is still accepted for reads for one release.
- **Docker / env var** `MEMWRIGHT_DATA_DIR` deprecated in favor of `ATTESTOR_DATA_DIR`. Both are read in v3.x.
- **Canonical CLI** is now `attestor`. `memwright` and `agent-memory` remain as deprecated aliases through v3.x; both will be removed in v3.2.

### Added

- `attestor migrate` CLI subcommand for non-destructive store migration from `~/.memwright/` to `~/.attestor/`.
- `attestor` binary as the canonical entry point (`memwright` and `agent-memory` continue to work).
- `ATTESTOR_PATH` environment variable (canonical). `MEMWRIGHT_PATH` is still read in v3.x with a deprecation warning.
- Hero repositioning and brand refresh across README, docs site, install wizard, and demo recordings.

### Changed

- All user-facing strings (CLI help, log messages, error text, hook output) updated to reference Attestor.
- Docker images, Terraform modules, and CI workflows rebranded to Attestor.
- Default cloud database names migrated to `attestor` (ChromaDB collection dual-registers the old name for back-compat).
- Documentation, SVG diagrams, and demo scripts regenerated with the Attestor brand.

### Compatibility matrix

| Surface | v2.x | v3.0 – v3.1 | v3.2+ |
|---|---|---|---|
| `import agent_memory` | works | removed | removed |
| `import memwright` | never existed | shim + warning | removed |
| `import attestor` | — | canonical | canonical |
| CLI `memwright` / `agent-memory` | works | alias + warning | removed |
| CLI `attestor` | — | canonical | canonical |
| Env `MEMWRIGHT_PATH` / `MEMWRIGHT_DATA_DIR` | works | read + warn | removed |
| Env `ATTESTOR_PATH` / `ATTESTOR_DATA_DIR` | — | canonical | canonical |
| `~/.memwright/` auto-read | default | fallback + warn | removed |
| MCP URI `memwright://` | works | read-accepted | removed |
| MCP URI `attestor://` | — | emitted | emitted |

### Migration

```bash
pip uninstall memwright
pip install attestor
attestor migrate            # copies ~/.memwright/ → ~/.attestor/ if present
attestor doctor             # verifies all three storage roles
```

See [MIGRATING.md](./MIGRATING.md) for import rewrites, MCP config changes, Docker env var rotation, and the v3.2 cleanup checklist.

## [2.0.7] — 2026-04-14

Last release under the `memwright` name. See the [v2 release history](https://github.com/bolnet/attestor/releases?q=v2.) on GitHub.
