# Memwright Fundraising Roadmap — Feature Checklist

**Source:** `~/Downloads/memwright-fundraising-roadmap.md` (April 2026)
**Audited:** 2026-04-18
**Legend:** ✅ Done · 🟡 Partial · ❌ Missing

---

## Existing Primitives (Claimed in roadmap intro — all verified)

| Primitive | Status | Fulfilled by |
|---|---|---|
| RBAC roles (6) | ✅ | `agent_memory/context.py:49-56` — `AgentRole` enum (ORCHESTRATOR, PLANNER, EXECUTOR, RESEARCHER, REVIEWER, MONITOR) |
| Namespace isolation | ✅ | `agent_memory/models.py:21`; enforced in every store (SQLite row-level `namespace` col, pgvector, ArangoDB, DynamoDB, Cosmos, AlloyDB) |
| Supersession / contradiction detection | ✅ | `agent_memory/temporal/manager.py:43-73` — `check_contradictions()`, `supersede()` |
| Provenance tracking | ✅ | `content_hash`, `source_id`, `confidence`, `access_count`, `last_accessed` in `models.py:38-51` |
| Deterministic retrieval (no LLM in hot path) | ✅ | `agent_memory/retrieval/orchestrator.py` — 5-layer cascade (tag FTS → graph BFS → vector → RRF+PageRank → MMR) |
| Bitemporal / `as_of` replay | ✅ | `valid_from`, `valid_until`, `superseded_by`, `event_date` in `models.py`; timeline/current_facts in `temporal/manager.py`; UI `as_of` query param |

**Multi-role storage (do not re-build):** `store/registry.py:22-73` — pluggable document / vector / graph roles across SQLite, ChromaDB, NetworkX, pgvector (Postgres/Neon/AlloyDB), Apache AGE, ArangoDB, DynamoDB+OpenSearch+Neptune, Cosmos DiskANN.

---

## Tier 1 — Enterprise security review survival (Weeks 1–6)

| # | Feature | Status | Fulfilled by / Gap |
|---|---|---|---|
| F1 | Audit log export (immutable, signed, hash-chained, Splunk/Datadog/S3/Blob/GCS) | 🟡 | JSON export exists (`core.py:721-763` `export_json` / `import_json`) — **MISSING:** cryptographic signing, per-namespace hash-chain, object-lock destinations |
| F2 | PII redaction at ingest (Presidio/spaCy, FINRA preset) | ❌ | No redaction hooks; no pre-write pipeline. Greenfield build. |
| F3 | BYOK encryption (AWS KMS / Azure Key Vault / GCP KMS / Vault) | ❌ | Cloud backends (`aws_backend.py`, `azure_backend.py`, `gcp_backend.py`) pass credentials but no CMK wiring or rotation story |
| F4 | Enterprise SSO + SCIM (Okta, Entra ID, Google OIDC; WorkOS/Clerk/Stytch) | ❌ | No auth layer in `api.py`; MCP server has no SSO |
| F5 | Benchmark splash (LongMemEval + MAB head-to-head vs Mem0/Zep/Letta) | 🟡 | `agent_memory/locomo.py` (LOCOMO runner), `agent_memory/mab.py` (MAB runner), `bench_claude.py`. **MISSING:** LongMemEval runner, published head-to-head results, blog, repo |
| F6 | Finance reference scenario (14-day Coach Chat, notebook + video + docs) | ❌ | No finance notebook under `docs/demo/` — only install/wizard tapes |

---

## Tier 2 — Multi-agent narrative (Weeks 7–14)

| # | Feature | Status | Fulfilled by / Gap |
|---|---|---|---|
| F7 | Multi-agent write consistency (CRDT, `mem.branch/merge/diff`) | 🟡 | `context.py` has `AgentContext` + `AgentRole`; write quotas exist in context. **MISSING:** CRDT semantics, transactional write batches, branch/merge/diff API |
| F8 | Temporal SQL API (`SELECT … AS OF TIMESTAMP`, `memwright replay --as-of`) | 🟡 | `temporal/manager.py` has `as_of` programmatically; `ui/app.py` exposes `as_of`; `models.py` has bitemporal fields. **MISSING:** SQL surface, `replay` CLI subcommand |
| F9 | Role-based redaction (Reviewer/Monitor/Auditor differential views) | 🟡 | Roles and `Visibility` enum in `context.py:41-46` exist. **MISSING:** per-role redaction filters on recall, hashed-only Auditor view |
| F10 | Microsoft Agent Framework context provider | ❌ | No `agent-framework` / Semantic Kernel adapter |
| F11 | LangGraph + CrewAI + AutoGen native integrations | ❌ | Only MCP server (`mcp/server.py`) and HTTP client (`client.py`) — no first-party adapters |
| F12 | Observability dashboard (Prometheus: drift, hit rate, contradictions, supersession velocity) | 🟡 | `core.py:509-717` has `_ops_log` ring buffer + `health()` with p50/p95/p99 latency + sparkline; UI renders stats. **MISSING:** Prometheus `/metrics` endpoint, drift/velocity gauges, Grafana dashboard JSON |

---

## Tier 3 — Vertical packaging (Weeks 15–24)

| # | Feature | Status | Fulfilled by / Gap |
|---|---|---|---|
| F13 | "Memwright for Financial Advisors" SKU (FINRA profile, advisor/client templates) | ❌ | No SKU packaging, no FINRA redaction config, no pre-built advisor agent roles |
| F14 | SOC 2 Type II kickoff (Vanta/Drata) | ❌ | Process work, no codebase artifact |
| F15 | FedRAMP Moderate-aligned package (FIPS 140-3, STIG base image, air-gapped, SBOM) | ❌ | Reference Terraform in `agent_memory/infra/` not FIPS/STIG; no SBOM generation in CI |
| F16 | FinMemBench (own benchmark, supersession + disclosure labels) | ❌ | Only LOCOMO/MAB runners; no finance ground-truth dataset |
| F17 | Enterprise pricing page (OSS → Pro $25K → Enterprise $120K+) | ❌ | No pricing page in `bolnet.github.io/agent-memory/` |
| F18 | First reference customer case study | ❌ | No case study artifact |

---

## "What NOT to build" — confirm not overbuilding

| Anti-feature | Status in codebase | Note |
|---|---|---|
| Vector store abstraction | ✅ Already done | `store/registry.py` + `store/base.py` interfaces — multi-backend is core architecture, not scope creep |
| LLM-judge evals | ✅ Kept out of critical path | `retrieval/orchestrator.py` has no LLM; extraction LLM (`extraction/llm_extractor.py`) is optional, off-path |
| Consumer personal memory | ✅ Not built | Oriented to agent teams |
| Generic enterprise search | ✅ Not built | Memory-scoped only |
| Agent framework | ✅ Not built | Memory layer only |
| Horizontal MCP server chase | ✅ Single MCP server, not many | `mcp/server.py` — 8 tools, focused scope |

---

## Summary

- **Technical foundation (~80% claimed):** confirmed — all six roadmap primitives exist with clear file owners.
- **Tier 1 blockers for enterprise pilot:** F1 signing/chain, F2 PII, F3 BYOK, F4 SSO/SCIM, F6 finance demo — all greenfield.
- **Tier 2 differentiators already half-built:** F7, F8, F9, F12 each sit on existing scaffolding; each is a 1–3 week extension rather than new subsystem.
- **Tier 3 is packaging + compliance process**, not new architecture.
