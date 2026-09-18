# Token Savings: How Attestor Reports Them

Attestor's value on the token axis is structural, not a single headline number:
**recall injects a flat, budget-capped payload per call, while replaying the
full conversation history into context grows unbounded.** This document
describes *how to ask Attestor what it injected* — on demand — and is precise
about what is and is not measured.

> No benchmark numbers are committed to this repo by policy. Comparative
> figures (e.g. the standalone token benchmark) live outside the codebase; see
> [Benchmarking](#benchmarking-the-savings) for where.

## What "tokens saved" means here

There is **no persisted, cross-project running total** of tokens saved. Nothing
in the code accumulates a grand sum across projects or sessions. Instead,
Attestor reports the **packed payload size of each recall** — the number of
tokens it actually injected to answer one query. The "savings" is the
difference between:

- **Packed recall** — flat per call, capped by the recall `budget`.
- **Full-context replay** — the entire (growing) history re-sent each turn.

Attestor measures the first directly. The second is the baseline you are
avoiding; it is not stored.

## Asking on demand

### CLI — `attestor recall … --show-tokens`

```bash
attestor recall ~/.attestor "what did we decide about auth?" --show-tokens
```

Prints, after the results:

```
[tokens] packed <N> tokens across <M> memories (cap <budget>).
This payload is flat regardless of conversation length —
full-context replay grows unbounded.
```

Implementation: `attestor/cli/commands/memory.py` (`_cmd_recall`), using
`estimate_tokens()` from `attestor/utils/tokens.py`.

### MCP — `memory_recall` response (for Claude, mid-session)

The `memory_recall` tool returns a `tokens` block on **every** recall, so an
agent can ask "how many tokens did that inject?" without shelling out to the
CLI:

```json
{
  "count": 2,
  "tokens": {
    "packed": 8,
    "budget": 4096,
    "note": "Packed payload is flat per recall regardless of conversation length — full-context replay grows unbounded."
  },
  "memories": [ ... ]
}
```

- `packed` — sum of `estimate_tokens()` over the returned memories.
- `budget` — the cap applied to this recall.

Implementation: `attestor/mcp/server.py` (`memory_recall` branch of
`_handle_tool`).

## How packing works

The recall pipeline ends in a token-budget pack so the payload never exceeds
the configured cap:

- **`fit_to_budget()`** (`attestor/retrieval/scorer.py`) — greedy fit by score
  descending; always includes at least one result.
- **`_long_context_pack()`** (`attestor/retrieval/orchestrator/postprocess.py`)
  — verbatim top-scored pack for large-context answerers, no MMR diversity
  penalty.

Token counts come from a pure word-count heuristic
(`estimate_tokens`, ~1.3 tokens/word) — no LLM in the path.

## What is NOT measured

- **No cross-project aggregate.** There is no "total tokens saved across all
  projects" counter anywhere in the code.
- **No in-harness tokens-per-query benchmark.** The LME-S benchmark harness
  does not instrument tokens/query; it measures recall@K and answer accuracy.
- **No automatic baseline computation.** The "full-context replay" comparison
  is framed in the output note, not calculated against a real transcript.

## Benchmarking the savings

A standalone token benchmark lives outside this repo (the `context-clock`
project) and powers the comparative narrative on the marketing site
(`docs/context-clock-benchmark.html`). Treat its figures as a single
synthetic comparison, not a measured aggregate — and keep the numbers out of
this repository per the no-bench-stats-in-repo policy.

## Surfaces summary

| Surface | Reports packed tokens? | Where |
|---|---|---|
| CLI `recall --show-tokens` | Yes (opt-in flag) | `attestor/cli/commands/memory.py` |
| MCP `memory_recall` | Yes (always, `tokens` block) | `attestor/mcp/server.py` |
| REST `/recall` | Not yet | `attestor/api.py` |
