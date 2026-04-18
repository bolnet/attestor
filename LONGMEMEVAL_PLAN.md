# LongMemEval Plan

**Owner:** Attestor (memwright)
**Branch:** `enterprise-security-hardening`
**Drives:** Roadmap F5 (Benchmark splash) + Section 9 item 2 (this-week task)
**Kill-switch:** if Attestor cannot beat Zep on the temporal sub-benchmark, **stop feature work and fix retrieval** (roadmap §10)

---

## 1. Goal

Produce a **publishable, reproducible head-to-head** of Attestor vs Mem0 / Zep / Letta / OpenAI memory on LongMemEval, with a specific focus on the **temporal** sub-benchmark where the product thesis claims differentiation (bitemporal + supersession + as_of replay).

### Success criteria (must-hit before publishing)

| Metric | Target | Why |
|---|---|---|
| Overall LongMemEval accuracy | **top-3** of published systems | Roadmap traction-bar item |
| Temporal sub-benchmark | **#1 OR within 2pp of #1** | Direct thesis validation; this is the pitch |
| Single-session-user (SSU) | ≥ Mem0 baseline | Sanity check — ensures we don't regress on easy cases |
| Multi-session reasoning | ≥ Zep baseline | Our graph layer should help here |
| Knowledge updates | **#1** | Supersession is a unique Attestor primitive |
| Abstention | ≥ 50% F1 | Non-hallucination bar for compliance narrative |
| Cost per run | < $40 | Reproducibility gate for a blog post |
| Runtime | < 2 hours on a single workstation | For iteration speed |

### Explicit non-goals

- Tuning for LongMemEval (no eval-specific prompts, no test-set leakage)
- LLM judges in the retrieval path — **determinism is sacred** (roadmap §5)
- Running MAB or LOCOMO as part of this plan — those already exist and will be re-run separately
- Beating SOTA on every sub-benchmark — we only need top-3 overall + #1 temporal

---

## 2. LongMemEval dataset overview

- **Source:** `https://github.com/xiaowu0162/LongMemEval` (also on HuggingFace)
- **Size:** 500 curated QA pairs across long multi-turn chat histories
- **Five memory abilities scored:**
  1. Information extraction
  2. Multi-session reasoning
  3. Temporal reasoning
  4. Knowledge updates
  5. Abstention
- **Scoring:** LLM-as-judge on "correct / incorrect / abstained"
- **Format:** JSON; each sample is `(history, question, gold_answer, category)` tuples

Context7 + upstream repo are the source of truth for the dataset schema. First implementation step is to confirm schema vs `locomo10.json` (known format) before coding.

---

## 3. Architecture — mirror existing `locomo.py`

Build `agent_memory/longmemeval.py` (~600-900 LOC) in the same shape as `agent_memory/locomo.py:1-752`:

```
agent_memory/longmemeval.py
├── URL + CATEGORY_NAMES constants               (mirror locomo.py:53-83)
├── ANSWER_PROMPT / JUDGE_PROMPT                 (mirror locomo.py:85-121)
├── _get_client() / _chat()                      (share helpers with locomo.py)
├── download_longmemeval(dest)                   (mirror locomo.py:203-209)
├── load_longmemeval(path) -> list[Sample]       (mirror locomo.py:212-242)
├── ingest_history(mem, sample, ...)             (mirror locomo.py:245-320)
├── answer_question(mem, sample, ...)            (mirror locomo.py answer loop)
├── judge_answer(generated, gold)                (mirror locomo.py JUDGE_PROMPT)
├── run(args) -> Report                          (main entry)
└── _save_report(report, path)                   (JSON dump for reproducibility)
```

### Share code, don't duplicate

- Reuse `token_f1`, `_upgrade_embeddings_for_benchmark` from `mab.py:1-60` (already shared with `locomo.py:26`)
- Factor out `_get_client`, `_chat`, `JUDGE_PROMPT`, `ANSWER_PROMPT` helpers into a new `agent_memory/_bench_common.py` — touched during plan execution, not a blocker
- Keep the three benchmark runners independently invokable

### Typed data model

Use frozen dataclasses, per `rules/python/coding-style.md`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LMESample:
    sample_id: str
    category: str            # one of 5 abilities
    history: list[dict]
    question: str
    gold_answer: str
    session_dates: list[str]

@dataclass(frozen=True)
class LMEResult:
    sample_id: str
    category: str
    generated: str
    gold: str
    label: str               # "correct" | "wrong" | "abstained"
    latency_ms: float
    recall_tokens_used: int
```

### CLI wiring

Add to `agent_memory/cli.py` (mirror `locomo` args at `cli.py:186-213`):

```
agent-memory longmemeval [--data PATH] [--judge-model M] [--answer-model M]
                         [--use-extraction] [--max-samples N] [--budget N]
                         [--output JSON] [--verbose] [--env-file .env]
                         [--backend sqlite|chroma|postgres|arango]
```

---

## 4. Implementation tasks (ordered)

Each task is small enough to ship + commit independently.

| # | Task | File(s) | LOC | Test |
|---|---|---|---|---|
| 1 | Confirm LongMemEval schema via context7 + upstream repo; write fixture | `tests/fixtures/lme_mini.json` | ~200 | — |
| 2 | Factor shared bench helpers | new `agent_memory/_bench_common.py`; touch `locomo.py`, `mab.py` | 100 | existing suites green |
| 3 | `load_longmemeval`, `download_longmemeval` + parsing into `LMESample` | `longmemeval.py` | 150 | `test_longmemeval_load` |
| 4 | `ingest_history` (raw + extraction modes, pronoun resolution) | `longmemeval.py` | 200 | `test_longmemeval_ingest` |
| 5 | `answer_question` with recall + `as_of=last_turn_date` to force temporal correctness | `longmemeval.py` | 150 | `test_longmemeval_answer` |
| 6 | `judge_answer` (LLM-as-judge, JSON output parsing, robust) | `longmemeval.py` | 100 | `test_longmemeval_judge_parsing` |
| 7 | `run(args)` with per-category accuracy + wall-clock + cost accounting | `longmemeval.py` | 150 | `test_longmemeval_run_mini` |
| 8 | CLI subcommand | `cli.py` | 30 | smoke test via `subprocess.run` |
| 9 | Competitor harness adapters (Mem0, Zep, Letta, OpenAI) — separate script, gated behind extras | `bench/competitors/*.py` | 400 | skip when deps missing |
| 10 | Reporting: `docs/BENCHMARK_RESULTS.md` + CSV + chart SVG | `docs/`, `scripts/` | 200 | — |

Total: ~1,500 new LOC + ~400 competitor adapter LOC.

### Task 1 gate

Before writing any Python, use context7 to pull the upstream LongMemEval README + dataset schema. Confirm:
- Dataset URL
- Field names (`question`, `answer`, `category`, `history`, `date`-style fields)
- License for redistribution in a `docs/BENCHMARK_RESULTS.md` table

Block further work if the dataset format differs materially from the LOCOMO shape we assumed.

---

## 5. Competitor head-to-head methodology

**Each system gets:**
- Identical hardware (single workstation, no GPU required)
- Identical LLM for answering (default: `openai/gpt-4.1-mini` via OpenRouter)
- Identical LLM for judging (default: `openai/gpt-4.1-mini`)
- Identical dataset slice (full 500 or documented subset)
- Identical token budget where exposed (4K recall context)

**Per-system notes:**

| System | Version | Key flag | Risk |
|---|---|---|---|
| Attestor | HEAD of `enterprise-security-hardening` | `--use-extraction --backend=sqlite` | Known MAB regression — upgrade embeddings first (see §6) |
| Mem0 | latest PyPI | use their published bench script | Their default local config uses OSS embeddings; match ours |
| Zep | latest Docker | use their bench harness | Temporal edges are their strong point — direct rival |
| Letta | latest PyPI | use MemGPT core loop | Sub-agent memory, but weaker temporal model |
| OpenAI memory | API memory feature | via `conversations` API | Black-box — document what we can and can't control |

**Reproducibility package** (shipped with the blog post):
- Exact dataset hash
- `requirements.txt` pinned
- Environment variables documented (OPENROUTER_API_KEY, etc.)
- Single-command repro: `make bench-longmemeval-all`

---

## 6. Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **MAB embedding regression transfers to LongMemEval** — `all-MiniLM-L6-v2` (384D) too weak for multi-hop (MEMORY.md) | **HIGH** | Before full run, upgrade embeddings per Phase 6: swap to `BAAI/bge-large-en-v1.5` or `text-embedding-3-large`. Already scaffolded via `_upgrade_embeddings_for_benchmark` (`mab.py:26`). |
| LongMemEval-specific prompt engineering tempts us → eval overfit | MEDIUM | Freeze `ANSWER_PROMPT` + `JUDGE_PROMPT` before first run. No sample-specific tuning. |
| Judge model bias (GPT-4.1-mini too lenient / too strict vs Zep's judge) | MEDIUM | Publish dual scores: `gpt-4.1-mini` and `claude-haiku-4-5` as judges. Roadmap §2 pillar #2: determinism — but this is judging, not retrieval. |
| Competitor harness breaks (Mem0 API change, etc.) | MEDIUM | Pin versions; document in repro package; isolate via extras. |
| Dataset redistribution licensing | LOW | Link to upstream, don't fork. |
| Top-3 achievable on overall accuracy but temporal sub-benchmark fails kill-switch | MEDIUM | This is the explicit roadmap gate. If it fails: freeze F1-F18 work, diagnose temporal scorer, re-test. **Mitigation = fix product, not ship anyway.** |

---

## 7. Deliverables

| # | Artifact | Location |
|---|---|---|
| D1 | `longmemeval.py` runner | `agent_memory/longmemeval.py` |
| D2 | CLI subcommand | `agent_memory/cli.py:+30` |
| D3 | Unit + integration tests | `tests/test_longmemeval.py` |
| D4 | Competitor adapters | `bench/competitors/*.py` |
| D5 | Reproducibility Makefile | `bench/Makefile` |
| D6 | Results JSON (per-system, per-category) | `docs/bench/longmemeval-2026-04.json` |
| D7 | Public results page | `docs/BENCHMARK_RESULTS.md` |
| D8 | Chart SVGs (overall + temporal sub-bench) | `docs/bench/*.svg` |
| D9 | Blog post draft | `docs/blog/longmemeval-head-to-head.md` |
| D10 | Social launch artifacts for Week 7 blitz | `docs/bench/social/*.md` |

---

## 8. Timeline (inside roadmap Week 4–5 slot for F5)

| Day | Work |
|---|---|
| 1 | Task 1: context7 schema confirmation + fixture; Phase 6 embedding upgrade dry-run |
| 2 | Tasks 2–3: helpers + loader + tests |
| 3 | Tasks 4–5: ingest + answer loop |
| 4 | Tasks 6–7: judge + orchestration; first Attestor-only dry run on 50 samples |
| 5 | Task 8: CLI wiring; full Attestor run (500 samples); decide go/no-go on gate |
| 6–7 | Task 9: competitor adapters, one at a time (Mem0 → Letta → Zep → OpenAI) |
| 8 | Task 10: reporting, charts, blog draft |
| 9 | Internal review; publish to `docs/` |
| 10 | Week 7 blitz — cross-post to Deedy Das / Swyx / HN (deferred to roadmap schedule) |

---

## 9. Immediate next steps

Three choices, in order of value:

1. **Use context7 to pull the LongMemEval repo + schema**, then write Task 1's fixture + `load_longmemeval`.
2. **Upgrade embeddings first** (Phase 6) and re-run MAB to confirm the fix before LongMemEval benefits from it.
3. **Factor `_bench_common.py`** so LongMemEval starts on a clean base.

Recommended order: **2 → 1 → 3**, because a pending 400D-embedding regression will pollute LongMemEval numbers and waste a full run.

## 10. Out of scope

- Publishing LongMemEval-specific scorecards for backends we don't officially support (Arango, Postgres live-deployed) — they will run through the same `AgentMemory` API and pick up the benchmark automatically, but production-grade certification is a separate track.
- Human-judged LongMemEval runs (academic rigor, not fundraising-relevant at this stage).
- Continuous regression CI for LongMemEval (too expensive; LOCOMO + MAB cover this).
