# build-bench — live software-build token-savings benchmark

Drive a **real multi-agent software build** through the Claude Code **Workflow**
tool, one step at a time, and measure how many context tokens attestor saves
versus a memoryless agent that must re-feed the whole accumulated history.

This is the *live, long-horizon* complement to the static LME-S harness: it
exercises attestor's **capture → recall** loop turn-by-turn as a real codebase
grows, which is exactly where the value compounds.

## Methodology — single-run counterfactual

Build the software **once** with attestor recall. At step `i` the ledger records
two numbers (canonical `attestor.utils.tokens.estimate_tokens`):

| number | meaning |
|---|---|
| `recall_tokens` | what attestor packed for step `i` — bounded by `--budget`, ~flat |
| `full_history_tokens` | tokens of the corpus through step `i-1` — what a **naive** agent re-feeds; grows every step |

`savings_ratio = full_history_tokens / recall_tokens`. The naive number is a
*computed counterfactual on the same run*, not a second executed build. Both
baselines see the same information units (every step's captured decisions); the
naive agent carries all of them forward, attestor surfaces only the relevant
bounded subset. Savings are ~0 early (little/all-relevant history) and grow as
the build diversifies and history dwarfs the recall budget.

## Pieces

| file | role |
|---|---|
| `ledger.py` | deterministic token accounting (unit-tested in `tests/test_build_bench_ledger.py`) |
| `recall_step.py` | Mode-A wrapper: recall context for a step → JSON `{recall_tokens, n_memories, context}` |
| `capture_step.py` | Mode-A wrapper: persist step decisions to attestor + append a ledger row |
| `workflow.js` | the Workflow orchestrator: `recall → build subagent → capture+account`, sequential |
| `_env.py` | loads `<store>/.env` so bare `python -m` runs get Postgres creds |

The deterministic token math lives in Python (no LLM); the Workflow only
sequences subagents and shells into these wrappers (the JS sandbox can't call
Python/attestor directly).

## Run

Smoke (default = Phase 0 M0.3, 3 steps on the fitness backend):

```
Workflow({ scriptPath: "evals/build_bench/workflow.js" })
```

Scale to ~100 steps — pass your own step list and a fresh ledger/namespace:

```
Workflow({
  scriptPath: "evals/build_bench/workflow.js",
  args: {
    namespace: "bench-fitness-build-100",
    ledger: "/tmp/build_bench_100.json",
    branch: "bench/token-savings-100",
    budget: 4000,
    steps: [ { id, query, task, entity }, ... ]   // ~100 entries
  }
})
```

The per-step ledger lands at `args.ledger`; `to_json().summary` has the
cumulative rollup. **Numbers go to a memory file, never the repo.**

## Notes
- Each run uses its own attestor namespace → hard tenant isolation, so parallel
  benchmark runs can't cross-contaminate recall.
- `--budget` is the realistic injected-context cap (default 4000). Naive history
  is unbounded — that's the point.
- The two-runs A/B variant (naive actually hits the context wall ~turn 40 and
  can't finish) is a separate, more expensive demo, not implemented here.
