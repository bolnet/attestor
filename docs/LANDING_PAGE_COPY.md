# Memwright Landing Page Copy

## Structure: 21 sections → 9 sections

---

## 1. HERO

**Eyebrow:** `OPEN SOURCE MEMORY FOR AI AGENTS`

**Headline:**
Your agent forgets everything.
We fixed that.

**Subhead:**
Memwright gives AI agents persistent, ranked memory that stays out of the context window. No Docker. No API keys. No monthly bill. Just install and go.

**Supporting line:**
One package. Works with Claude Code, Cursor, Windsurf, or any MCP client. Scales from your laptop to AWS, Azure, and GCP.

**Install command:**
```
$ poetry add memwright && claude mcp add memory -- memwright mcp
```

**Meta line:** Free. Open source. Apache 2.0. Python 3.10-3.14. Works right now.

**Stats bar:**
- 81.2% LOCOMO accuracy
- 1.4ms P50 recall
- 8 MCP Tools
- 5 Cloud Backends
- 607 Tests
- $0/mo

**CTAs:** GitHub | PyPI | MCP Registry | Benchmarks

---

## 2. THE PROBLEM (promoted from old section 5)

**Tag:** `THE PROBLEM`

**Headline:**
Claude Code starts every session from zero.
Its built-in "memory" makes things worse.

**Opening quote (italic):**
"Four hours debugging a gnarly database migration, 30 back-and-forth messages about schema evolution. Closed the terminal, came back after dinner — Claude had no idea what we'd figured out."

**Side-by-side comparison:**

| MEMORY.md | Memwright |
|-----------|-----------|
| x 200-line hard limit — content beyond silently truncated | ✓ Memory lives on disk, outside the context window entirely |
| x At 40K characters, responses degrade "like wading through quicksand" | ✓ Token budget you control — 2K, 4K, 20K, your choice |
| x Claude systematically ignores rules defined in MEMORY.md | ✓ 3-layer ranked search returns only the best memories |
| x Entire file loads into context every message — no search, no ranking | ✓ Contradictions resolved algorithmically — no LLM call |
| x 15K+ tokens of unranked noise by month six | ✓ Same 2K cost at month 6. More data = better results. |

**Token cost over time chart:**

| | MEMORY.md | Memwright |
|---|---|---|
| Month 1 | 2K | 2K |
| Month 3 | 8K | 2K |
| Month 6 | 15K | 2K |

**Bottom line:** More memories makes Memwright better — more candidates to rank from — while the context cost stays the same.

---

## 3. WHY MEMWRIGHT (consolidates old Mission + Position + Landscape)

**Tag:** `WHY MEMWRIGHT`

**Headline:**
No LLM in retrieval. No framework lock-in.
No vendor bill. Just memory.

**2x3 grid of cards:**

### Zero config
`poetry add memwright`. Two commands. Done. No Docker, no database, no API keys. SQLite + ChromaDB + NetworkX provision automatically.

### Token-budget aware
`memory_recall(query, budget=2000)` — the only memory system that asks "how much space do you have?" before answering. Month 1 and month 12 cost the same.

### No hidden LLM calls
Tag matching, graph traversal, vector search, RRF fusion. All algorithmic. Same query = same results. Every time. No GPT calls on every add like Mem0.

### Contradiction handling
"User works at Google" auto-supersedes "User works at Meta." Full history preserved. Zero inference calls. No vector similarity coin-flip.

### Multi-agent native
Namespace isolation, 6 RBAC roles, provenance tracking, write quotas, token budgets. Built for orchestrated pipelines, not bolted on.

### Runs everywhere
Your laptop, AWS App Runner, GCP Cloud Run, Azure Container Apps. PostgreSQL, ArangoDB, or bare SQLite. Same API. Same results.

---

## 4. VS THE COMPETITION (clean table with latency)

**Tag:** `HONEST COMPARISON`

**Headline:**
We're not the only option.
We're the only free, standalone, fast one.

### Feature Comparison

| | Memwright | Mem0 | Zep | Letta | OpenAI | LangChain |
|---|---|---|---|---|---|---|
| **LOCOMO** | **81.2%** | 66.9% | ~75% | 74% | 52.9% | — |
| **Setup** | poetry add | API key | Neo4j | Docker+PG | ChatGPT only | Framework lock-in |
| **Graph memory** | Free, all tiers | $249/mo Pro only | Yes, all tiers | Agent-managed | No | No |
| **LLM in retrieval** | None (RRF + PageRank) | Yes (every add) | None | Yes (agent calls) | Unknown | Varies |
| **Self-host** | Yes (zero config) | Yes | Via Graphiti | Docker required | No API access | Yes (OSS) |
| **Cost floor** | **$0 forever** | $19/mo | $25/mo | $20/mo | N/A | Free |

### Latency Comparison (P50 recall)

| System | P50 | Notes |
|---|---|---|
| **Memwright (PG Docker)** | **1.4ms** | Full 3-layer pipeline, 81.2% LOCOMO |
| Ruflo | 2-3ms | Vector lookup only, not full retrieval |
| **Memwright (local)** | **9ms** | Zero-config, no Docker, no API keys |
| **Memwright (GCP Cloud Run)** | **156ms** | Full cloud API, scale-to-zero |
| Mem0 | 200ms | LLM in retrieval path |
| Zep | <200ms | P95 ~632ms under concurrency |
| Mem0 Graph | 660ms | Graph variant, much slower |

**Footnote:** LOCOMO scores are self-reported across vendors. Latency measured with full 3-layer pipeline (tag + graph + vector). Run yours: `memwright locomo`

---

## 5. HOW IT WORKS (kept, slightly trimmed)

**Tag:** `HOW IT WORKS`

**Headline:**
10,000 memories on disk. Only the best 4 enter your context window.

**Subhead:**
MEMORY.md dumps everything into context. Every line. Every message. Memwright stores memories in a separate process — SQLite + ChromaDB + NetworkX, on disk. Your context window never sees a memory until Claude calls memory_recall.

**Pipeline flow (5 steps):**

1. **Tag Match** — SQLite — Exact and partial tag hits. Fast. Deterministic.
2. **Graph Expansion** — NetworkX — Multi-hop BFS. Query "Python" finds "FastAPI," "Django," "pip" through graph edges.
3. **Vector Search** — ChromaDB — Semantic similarity for when exact matches miss.
4. **RRF Fusion + Scoring** — PageRank + Confidence Decay — Memories found by multiple layers score dramatically higher.
5. **MMR Diversity + Budget Fitting** — Eliminates near-duplicates. Packs top memories into your token budget. The other 9,996 never enter context.

---

## 6. RUNS EVERYWHERE (carousel — kept as-is)

**Tag:** `RUNS EVERYWHERE`

**Headline:**
Your laptop. Your cloud. Your air-gapped server.
It just works.

**Subhead:**
Most memory systems pick a lane. Memwright picks yours.

**Carousel cards:** Local | AWS | Azure | GCP | ArangoDB | PostgreSQL | Docker/On-Prem

**Bottom line:** You're already paying for the brain. Memwright gives it a memory — on infrastructure you already own.

---

## 7. QUICK START (tabbed — kept as-is)

**Tag:** `QUICK START`

**Headline:**
Two minutes. Zero dollars. Full memory.

**Tabs:** MCP Server | Plugin | Python API

---

## 8. DESIGN PRINCIPLES (trimmed from 7 cards to inline)

**Tag:** `WHAT WE BELIEVE`

**Headline:**
Opinionated. On purpose.

**Cards (keep all 7):**
- Zero config beats configuration.
- Degradation beats failure.
- History beats deletion.
- Math beats LLMs in retrieval.
- Layers beat platforms.
- Dedup beats bloat.
- Your disk beats their cloud.

---

## 9. CLOSING

**Headline:**
Two commands. Zero dollars. Your agent remembers.

**Install command:**
```
$ poetry add memwright && claude mcp add memory -- memwright mcp
```

**Subline:** Free. Open source. Apache 2.0. Built by Surendra Singh — 15 years in financial services technology.

**CTAs:** GitHub | PyPI | MCP Registry

---

## SECTIONS CUT (moved to README / docs)

These were on the old page but don't belong on a conversion-focused landing page:

- **Mission** ("We hate dementia") — tone-deaf risk, replaced by concrete problem statement
- **Landscape** (6 competitor bashing cards) — replaced by clean comparison table
- **Position** ("Not a framework") — folded into Why Memwright cards
- **Audiences** (4 persona cards) — too marketing-speak for dev audience
- **Plugin** (4 cards) — pre-approval detail, mention in Quick Start tab instead
- **8 MCP Tools** (tool grid) — docs-level detail
- **Multi-Agent** (code sample + 4 cards) — docs-level detail, mentioned in Why Memwright card
- **Python API** (code samples) — in Quick Start tab + README
- **Embeddings** (provider table) — docs-level detail
- **Testing** (stats bar) — folded into hero stats
- **CLI** (19 pills) — docs-level detail
- **Compatibility** (integration table) — docs-level detail, mentioned in hero subhead

---

## NAV (simplified)

Old: Mission | Landscape | The Problem | How It Works | Platforms | Benchmarks | Quick Start

New: Problem | Why | Compare | How It Works | Platforms | Quick Start

---

## META TAGS

**Title:** Memwright — Persistent memory for AI agents

**Description:** Zero-config persistent memory for AI agents. 81.2% LOCOMO. 1.4ms recall. $0/month. Works with Claude Code, Cursor, Windsurf. Runs on your laptop, AWS, Azure, GCP.

**OG Title:** Memwright — Your agent forgets everything. We fixed that.

**OG Description:** Persistent, ranked memory for AI agents. No Docker. No API keys. No monthly bill. One poetry add. Your agent never forgets.
