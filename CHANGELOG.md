# Changelog

All notable changes to Attestor (formerly Memwright) are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0a5] — 2026-04-28

Surfaces previously-silent `try/except: pass` failures in the vector and graph write paths so dimension-mismatch and similar configuration drift become visible in logs instead of presenting as 0-hit recall results with no diagnostic trail. Three call sites updated (`core.py:add()` vector branch, `add()` graph branch, `update()` re-index) — each now `logger.warning(...)`s the swallowed exception. No behavior change beyond log surface; the document path remains the only hard dependency and still completes successfully when vector/graph writes fail.

Concrete failure mode that motivated the change: a Postgres schema provisioned with `vector(1536)` (OpenAI text-embedding-3-large default) silently rejected every bge-m3 (1024-dim) embedding via `psycopg2.errors.DataException`, leaving the `embedding` column NULL on every memory and turning recall into a no-op. The new log line surfaces the actual error message so operators can diagnose and migrate the schema.

Distribution: this release also mirrors the image to **5 Docker registries** simultaneously via the auto-publish workflow — `bolnet2025/attestor` (Docker Hub), `ghcr.io/bolnet/attestor` (GHCR), `quay.io/bolnet/attestor` (Quay.io), and `public.ecr.aws/m6h5j7o3/attestor` (AWS ECR Public, manual one-shot).

## [4.0.0a4] — 2026-04-27

Introspection-only fallback for the MCP server so external registries (Glama, Smithery) can verify the listing without provisioning Postgres + Neo4j. When `ATTESTOR_MCP_TOLERATE_INIT_FAILURE=1` is set, `attestor mcp` swallows backend connection errors at startup and continues to advertise its 8 tools via `tools/list`. Any `tools/call` against a non-initialized server returns a clear "configure backends to enable execution" error instead of crashing. Production deployments leave the env var unset and behave as before — strict, fail-closed init.

Also adds a top-level `Dockerfile` that builds an introspection-only image (~150 MB, single layer over `python:3.12-slim`) for use as the registry-listing artifact. Real production deployments continue to use `attestor/infra/local/docker-compose.yml` for the local stack, or the imperative cloud-deploy guides under `docs/install/{aws,gcp,azure}.md` for managed-service deploys.

## [4.0.0a3] — 2026-04-27

Adds the `mcp-name: io.github.bolnet/attestor` marker to the PyPI README so the MCP Registry can validate ownership of the PyPI package and accept the `server.json` publish. No code changes vs `4.0.0a2`.

## [4.0.0a2] — 2026-04-27

**Hotfix on the same day as 4.0.0a1.** Adds `requests` to required core dependencies so the documented default embedder (local Ollama `bge-m3`) actually works on a fresh `pip install attestor`. In 4.0.0a1, `attestor doctor` raised `RuntimeError: No embedding provider available` even with Ollama running, because `OllamaEmbeddingProvider` lazy-imports `requests` and that wasn't in the wheel's deps. Smoke-verified end-to-end against PyPI: `pip install --pre attestor==4.0.0a2` now succeeds and Ollama probe returns a 1024-D bge-m3 embedding.

Users on `4.0.0a1` should `pip install --upgrade --pre "attestor==4.0.0a2"` (or pin to `attestor==4.*`).

## [4.0.0a1] — 2026-04-27

**v4 — deterministic, auditable memory tier purpose-built for multi-agent chat systems.** Greenfield rebuild on a v4-native Postgres schema with hard tenant isolation, bi-temporal facts, and a no-LLM retrieval critical path. v3 was alpha-only with no production users; there is no automated migration path, and no v3 data to carry forward. Drop your previous DB and install fresh.

This is the first public alpha cut of v4 to PyPI. API surface should be considered subject to breaking change until 4.0.0 stable; install with `pip install "attestor==4.*"` to track the alpha line.

### Packaging — alpha release fixes

- Top-level package now re-exports `AgentContext`, `AgentRole`, `Visibility` from `attestor.context` so the README's first code example (`from attestor import AgentMemory, AgentContext, AgentRole`) works in a fresh `pip install`.
- `attestor.conversation.__init__` no longer eagerly imports `apply` / `episodes` / `ingest` (they close an import cycle through `attestor.extraction`). The heavier submodules are imported explicitly: `from attestor.conversation.ingest import ConversationIngest`.
- `attestor.conversation.apply` moved `Decision` and `ExtractedFact` imports under `TYPE_CHECKING` (PEP 563 lazy via `from __future__ import annotations`), breaking the same cycle at the type-annotation layer.
- `psycopg2-binary>=2.9.0` and `neo4j>=5.0.0` promoted to required core dependencies. The codebase imports both eagerly (e.g. `attestor/conversation/episodes.py`, `store/postgres_backend.py`, `store/neo4j_backend.py`); they were previously listed as optional `[project.optional-dependencies]` extras, which made `pip install attestor` produce a wheel that crashed on first import. Cloud-only deployments swapping to ArangoDB / DynamoDB / Cosmos / AlloyDB still install these as harmless extras until the eager backend imports are made lazy.

### Added

#### Track A — conversation ingest
- v4 schema with `users → projects → sessions → episodes → memories` hierarchy and Row-Level Security on every tenant table (`tenant_isolation_*` policies on the `attestor.current_user_id` session var).
- Bi-temporal columns on `memories`: `valid_from`/`valid_until` (event time) + `t_created`/`t_expired` (transaction time) + `superseded_by` chain. Nothing is deleted; supersession marks rows superseded.
- Episodes table for verbatim conversation turns; `ConversationIngest.ingest_round()` writes the episode + runs speaker-locked extraction in two passes (the +53.6 Mem0 fix) + resolves conflicts against existing similar memories.
- Auto-generated SOLO defaults: a singleton `local` user, an Inbox project (`metadata.is_inbox`), and daily sessions so zero-config ingest works.

#### Track B — retrieval upgrades
- Fact-augmented embedding text (`fact + raw user/assistant turns + tags`) — measurably better recall than embedding the bare fact.
- BM25 lane via trigger-maintained `content_tsv` tsvector + GIN index (since `to_tsvector('english', ...)` isn't IMMUTABLE).
- Reciprocal Rank Fusion (RRF, k=60) of vector + BM25 lanes in the orchestrator.
- Local Ollama `bge-m3` (1024-D, 8K context) is the default embedder; OpenRouter / OpenAI / Bedrock / Vertex / Azure are fallbacks.

#### Track C — temporal reasoning
- `recall(as_of=...)` bi-temporal replay — answers exactly what was active at a past point in transaction time. The regulator/audit case from the README is now real.
- `time_window` queries via `tstzrange` overlap on `(valid_from, valid_until)`.
- Orchestrator drops the `status='active'` filter when `as_of` or `time_window` is set, so superseded rows correctly return for replay.

#### Track D — Chain-of-Note reading
- `AgentMemory.recall_as_pack(query)` returns a structured `ContextPack` (id, content, validity window, confidence, source_episode_id) — citation-friendly view for agent consumption.
- Default Chain-of-Note prompt with a NOTES → SYNTHESIS → CITE → ABSTAIN → CONFLICT structure. The ABSTAIN clause is the load-bearing piece (every frontier model defaults to confabulation otherwise).

#### Track E — sleep-time consolidation
- `consolidation_state` on episodes (`pending|processing|done|failed`) with `FOR UPDATE SKIP LOCKED` queue + stale-lease reclaim.
- `SleepTimeConsolidator.run_once / run_forever` worker. Reflection prompts surface stable patterns / changed beliefs / contradictions; session-end promotion classifies session memories into `KEEP_SESSION / PROMOTE_PROJECT / PROMOTE_USER / DISCARD`.
- Provenance trail: consolidator-produced facts get `extraction_model="consolidation:<model>"`.

#### Track F — auth, MCP, signing, GDPR
- Ed25519 provenance signatures over the canonical payload `v1|<id>|<agent_id>|<t_created>|<content_hash>`. Signature stored on the row; `verify_memory()` detects raw-DB tampering.
- 5 MCP prompts: `record_decision`, `handoff_to`, `resume_thread`, `audit_decision`, `propose_invalidation`.
- Per-user quotas: `user_quotas` table with auto-init trigger + counter triggers maintained by `INSERT/DELETE` on memories/sessions/projects so quota checks are a single SELECT, not a COUNT scan.
- JWT auth middleware (Starlette `BaseHTTPMiddleware`) for HOSTED mode with HS256/RS256, `sub`/`exp`/`aud`/`iss` claim verification.
- GDPR delete + export: `purge_user()` cascades through six tables; `deletion_audit` is RLS-EXEMPT (it must outlive the user it logs, with `user_id TEXT NOT NULL` recorded as a string, not an FK). Audit row is INSERTed BEFORE the cascade so a partial-cascade failure still leaves a trail.

#### Track G — eval harness + CI gate
- Four runners share a standard `BenchmarkSummary` shape:
  - **Regression suite** (`evals/regression/`) — deterministic, no-LLM, YAML-driven catalog runs in CI on every PR.
  - **LongMemEval** wrapper (`evals/longmemeval/`) over the existing engine.
  - **BEAM** long-context runner (`evals/beam/`) with per-bucket aggregation (1k / 8k / 32k / 128k / 512k / 1M).
  - **AbstentionBench** (`evals/abstention/`) with phrase-based detector matching the CoN ABSTAIN clause output; F1 as primary metric so always-abstain and always-answer both score zero.
- `evals.gate` CLI loads `*_summary.json` files, diffs against `docs/bench/v4-baseline.json`, exits 1 on regression. Per-benchmark threshold overrides; bootstrap-friendly (missing baseline = no possible regression = pass).
- `evals.publish_baseline` CLI for promoting a verified run into the published baseline. Dry-run, partial promotion (`--only`), threshold preservation.
- `.github/workflows/evals.yml` — always-on unit + gate jobs; `workflow_dispatch`-only heavy benchmarks job (needs API-key secrets + a Postgres service container).

### Hardening

- `attestor doctor --v4-schema` validates structural invariants against a Postgres connection: required extensions (`vector`, `btree_gist`, `uuid-ossp`), all tenant tables have RLS enabled, audit tables are RLS-EXEMPT, content_tsv + quota counter triggers exist, SECURITY DEFINER lookup helpers are present.
- `attestor_user_id_for_external(text)` SECURITY DEFINER helper fixes the chicken-and-egg in user provisioning (RLS gates SELECT on id-match-var; bootstrap path can't SELECT a user whose id it doesn't yet know).
- Split `users` RLS policy into 4 (SELECT/UPDATE/DELETE strict on id-match-var, INSERT WITH CHECK true) so a non-SUPERUSER, NOBYPASSRLS runtime role can self-provision its own row.

### Removed (vs v3 alpha)

- The v3 schema is dropped entirely. No automated migration; no v3 data exists.
- `agent_memory` legacy import path is gone. Use `from attestor import ...` only.

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
