# Memwright Fundraising Execution Plan

**Source roadmap:** `~/Downloads/memwright-fundraising-roadmap.md`
**Created:** 2026-04-18
**Target:** Close $2.5M seed at $18–22M post-money SAFE by Week 24

This plan follows the roadmap's own cadence (Section 8) and three-things-this-week (Section 9) verbatim. No reframing. File paths are wiring guides only, not scope expansion.

---

## THIS WEEK (Section 9 — do in order)

### 1. Rewrite homepage hero around "audit-grade" and "bitemporal"

- Change `docs/index.html:1476` from *"Your agents don't forget. They never knew."* → **"What did the agent know, and when did it know it?"**
- Change `docs/index.html:1481` sub-hero to: *"The bitemporal, provenance-traced memory layer for multi-agent systems in regulated industries."*
- Update `<title>`, `<meta name="description">`, `og:title`, `og:description` (`docs/index.html:6-13`) to contain all three keywords: **auditable, deterministic, bitemporal**
- Update `README.md:17` tagline to match

### 2. Start the LongMemEval run

- Create `agent_memory/longmemeval.py` (same pattern as `locomo.py` / `mab.py`)
- Run head-to-head vs Mem0, Zep, Letta on identical hardware
- **Roadmap kill-switch:** if we cannot beat Zep on temporal correctness, STOP feature work and fix retrieval

### 3. Reach two contacts at regulated FIs

- 3-line pitch to SoFi / Fidelity / Schwab / regional bank alumni
- 20-minute discovery calls — not selling
- Sales action, no code

---

## WEEKS 1–6 — Ship Tier 1 + draft deck + draft data room

*(Section 8 row 1)*

| Wk | F# | Feature | Implementation |
|---|---|---|---|
| 1–2 | F1 | Audit log export | New `agent_memory/audit/` module. Every `add`/`recall`/`supersede`/`forget` emits HMAC-SHA256 signed log line; per-namespace hash chain. Export sinks: Splunk HEC, Datadog Logs API, S3 / Azure Blob / GCS object-lock. |
| 2–3 | F2 | PII redaction at ingest | Microsoft Presidio as pluggable pre-write hook. Config `redaction_profile: "finra"` (SSN, account numbers, DOB, PAN, routing numbers). Redaction events emitted to F1 audit log. |
| 3–4 | F3 | BYOK encryption | Customer-managed KMS: AWS KMS, Azure Key Vault, GCP Cloud KMS, HashiCorp Vault. Envelope-encrypt document + vector + graph stores. Document key rotation. |
| 4 | F4 | Enterprise SSO + SCIM | **Do NOT roll your own (roadmap).** Use WorkOS / Clerk / Stytch. Okta + Azure AD + Google Workspace OIDC. SCIM 2.0 provisions into the existing 6 `AgentRole` values. |
| 4–5 | F5 | Benchmark splash | Publish LongMemEval + Vectorize Agent Memory Benchmark head-to-head vs Mem0 / Zep / Letta. Repo + methodology + blog. **Target: top-3 LongMemEval, #1 on a temporal sub-benchmark.** |
| 5–6 | F6 | Finance reference scenario | 14-day Coach Chat demo covering: goal change, risk-tolerance change, compliance-flagged topic, regulatory disclosure, cross-session continuity. Ship as notebook + video + docs page. |

**Parallel:** draft pitch deck, draft data room.

---

## WEEK 7 — Benchmark release blitz

*(Section 8 row 2)*

- Publish benchmark (F5 artifact)
- Send to 5 AI-thought-leader accounts: **Swyx, Logan Kilpatrick, Deedy Das, Andrew Ng's team, Harrison Chase**
- Target: **1M+ impressions**
- One podcast appearance: **Latent Space** or **Cognitive Revolution** or **No Priors**

---

## WEEK 8 — First LOI outbound

*(Section 8 row 3)*

- **Target: 10 conversations → 3 LOIs**
- Named prospects (Section 3): SoFi, Fidelity, Schwab, Betterment, Wealthfront, Morgan Stanley, TIAA, Voya, Empower, Vanguard
- Secondary: regional banks deploying Agentforce or Glean

---

## WEEKS 9–12 — Tier 2 features + first paid pilot conversion

*(Section 8 row 4)*

| Wk | F# | Feature | Implementation |
|---|---|---|---|
| 9 | F7 | Multi-agent write consistency ("Git for memory") | CRDT semantics **or** transactional write batches. Ship `mem.branch()` / `mem.merge()` / `mem.diff()`. Formal conflict-resolution model beyond "newer wins." **Write short technical paper — roadmap calls this a "fundraise accelerant."** |
| 10 | F8 | Temporal SQL API | `SELECT ... AS OF TIMESTAMP '2026-03-15 14:30:00'` against document store. CLI: `memwright replay --namespace=client:acme --as-of=2026-03-15`. |
| 11 | F9 | Role-based memory redaction | Reviewer sees Planner writes; Monitor sees same with PII redacted; external Auditor role sees only hashed provenance chain. |
| 11 | F10 | Microsoft Agent Framework context provider | Ship as `ContextProvider` in MS Agent Framework v1.0. Match Neo4j integration depth. |
| 12 | F11 | LangGraph + CrewAI + AutoGen native integrations | **First-party adapters (not MCP).** Working examples for each. |
| 12 | F12 | Observability dashboard | Memory drift, recall hit rate, contradiction frequency, supersession velocity per agent, token-budget utilization. **Grafana-compatible Prometheus metrics.** Paid add-on for dual-license. |

**Parallel:** convert 1 pilot → paid at $25K+ (traction bar).

### Traction-bar pull-forwards (Weeks 11–12)

Roadmap places F14 / F17 / F18 in Tier 3 (Week 15+), but traction bar requires them **before** Week 13 investor outreach. Pull forward:

- F14 SOC 2 Type II kickoff → **engage Vanta / Drata Week 11**, observation Week 12
- F17 Enterprise pricing page → **Week 12**
- F18 First case study (with one quantified outcome) → **Week 12**

---

## WEEK 13 — Begin investor outreach

*(Section 8 row 5 · Section 7 order)*

**No more than 3 in parallel until term sheet.**

Tier 1 — memory specialists FIRST (in order):
1. **Astasia Myers, Felicis** — warm via Letta investor, Felicis portfolio founder, or Quiet Capital alum
2. **Deedy Das, Menlo Ventures** — cold with benchmark artifact; Anthology Fund in parallel
3. **Sarah Catanzaro, Amplify** — warm via Amplify founder

Submit Memwright directly to the **Anthology Fund** (public application) at the same time.

---

## WEEKS 14–20 — Fundraise in parallel with Tier 3

*(Section 8 row 6)*

| Wk | F# | Feature | Implementation |
|---|---|---|---|
| 15 | F13 | "Memwright for Financial Advisors" SKU | Named + priced + preconfigured. Bundle: F2 FINRA redaction profile + advisor–client namespace templates + pre-built roles (Advisor, Compliance Reviewer, Supervisor, Auditor) + F1 SEC-ready audit export. **List prices: $120K/yr <50 advisors, $350K/yr enterprise.** "Request Demo" live by Week 20. |
| 15–16 | F14 | SOC 2 Type II | (kickoff pulled to W11; continue observation window) |
| 16–18 | F15 | FedRAMP Moderate-aligned package | FIPS 140-3 crypto libraries. STIG-hardened base image. Air-gapped install manifest with SBOM. (Package for federal SI inclusion — not full ATO.) |
| 18–19 | F16 | FinMemBench | Dataset of realistic multi-turn advisor conversations with ground-truth labels for **supersession, disclosure obligations, temporal accuracy**. Score every incumbent. **You own the eval.** |
| 19 | F17 | Enterprise pricing page | (pulled to W12; keep live) Three tiers: **OSS Core (MIT, free) → Pro ($25K/yr, observability + SSO + priority) → Enterprise ($120K+, FINRA pack + audit export + BYOK + SLAs).** |
| 20 | F18 | First reference customer case study | One named logo, one quantified outcome. Roadmap's example: *"SoFi-equivalent Coach Chat reduced re-prompting tokens 47% and passed FINRA compliance review 3 weeks ahead of schedule."* |

---

## WEEK 20 — Target first term sheet

*(Section 8 row 7)*

**Round shape (roadmap fixed):**
- $2.5M
- $18–22M post-money SAFE
- Post-money cap
- No side letters
- 10–15% founder dilution
- **Do NOT accept a priced round at seed**
- Reserve 10–12% option pool

## WEEK 24 — Close round

*(Section 8 row 8)*

---

## Traction bar — must have ALL before Week 13 pitch

*(Section 6)*

- [ ] 2–3 signed LOIs from financial services firms *(Week 8 output)*
- [ ] 1 paying pilot at $25K+ *(Weeks 9–12 output)*
- [ ] Top-3 LongMemEval **OR** #1 novel benchmark *(Weeks 4–5 output)*
- [ ] 1,500+ GitHub stars *(benchmark splash lifts ~1K → 1.5K)*
- [ ] 3K+ monthly PyPI downloads
- [ ] SOC 2 Type II observation started *(F14 pulled to W11)*
- [ ] 1 Microsoft Agent Framework or LangGraph native integration merged *(F10/F11 output)*
- [ ] Enterprise pricing page live *(F17 pulled to W12)*
- [ ] 1 case study with quantified outcome *(F18 pulled to W12)*

**What you should NOT have:** meaningful ARR. Pre-seed investors underwrite team + benchmark + wedge + LOIs, not revenue.

---

## Guardrails — what we will NOT build

*(Section 5)*

- No new vector store abstraction (integrations only)
- No LLM-judge evals — **determinism is sacred**
- No consumer personal memory
- No generic enterprise search
- No agent framework
- No horizontal MCP server chase

---

## Honest risks to watch

*(Section 10 — not action items, but pressure tests)*

- SEP-2076 or similar could make memory an MCP primitive in 12–18 months → mitigation: vertical governance moat, not API surface
- Hyperscaler absorption (AgentCore Memory, Vertex Memory Bank, Fabric) → mitigation: air-gapped + EU-sovereign + FINRA-compliant long tail
- Mem0 category lead + AWS exclusivity → mitigation: they serve developer personalization, we serve enterprise compliance
- Interloom ($16.5M funded, adjacent) → mitigation: tacit ops knowledge ≠ audit-grade agent state
- "Feature, not a company" → mitigation: vertical compliance wedge (Interloom, Glean, Writer proved this pattern at $200M+ ARR)

---

## Investor ladder — pitch order

*(Section 7 — do not exceed 3 in parallel until term sheet)*

**Tier 1 — memory specialists**
1. Astasia Myers, Felicis
2. Deedy Das, Menlo Ventures (+ Anthology Fund)
3. Sarah Catanzaro, Amplify

**Tier 2 — enterprise AI infra**
4. Tim Chen, Essence VC
5. Guy Ward Thomas, DN Capital
6. Grace Isford, Lux Capital
7. Jake Flomenberg / Peter Wagner, Wing VC

**Tier 3 — enterprise / verticalized**
8. Ed Sim, Boldstart
9. Bessemer AI team
10. Sonya Huang, Sequoia

**Tier 4 — specialist / bet-on-founder**
11. Aditya Agarwal, South Park Commons
12. First Round Capital
13. The Anthology Fund (Menlo × Anthropic)

**Highest-leverage intro paths today:**
- Submit to Anthology Fund (public application)
- DM Deedy Das on X with benchmark results when published
- Reach Astasia via any Felicis portfolio founder
- Reach Sarah Catanzaro via LangChain / Amplify founders
- Cold conversion ~3%; benchmark splash can triple that
