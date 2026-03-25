# Claude Code Benchmark: With vs Without Memory

Real Claude Code CLI sessions measuring what happens when context fills up —
and whether memwright prevents recall degradation.

## How It Works

`bench_claude.py` runs automated multi-turn Claude Code sessions using `claude -p --resume`:

1. **Seed phase (turns 1-30)**: Tell Claude 30 specific facts (name, team, tech stack, family, deadlines)
2. **Filler phase (turns 31+)**: Heavy coding tasks (Go implementations) that burn context fast
3. **Recall phase (interleaved)**: Ask Claude to recall seeded facts — score keyword accuracy

Two modes:

| | Baseline | Memwright |
|---|---|---|
| Memory | None — context window only | Hooks + MCP installed |
| Install state | `poetry remove memwright` first | `poetry add memwright` |
| What happens at compression | Facts lost forever | Facts survive in external store |

## Results — 2026-03-21 (Haiku 4.5, 200 turns)

| Metric | Baseline | Memwright |
|---|---|---|
| Total cost | $29.00 | $26.07 |
| Compressions | 7 | 24 |
| Avg recall accuracy | 90.3% | 56.9%* |
| Late-stage recall | 70.8% | 45.8%* |
| Refusals | 3/72 | 18/72 |

\*Memwright score dragged down by Haiku refusing "quiz-style" recall questions
("I'm not answering test questions"). When Claude cooperated: 65/69 (94%) baseline
vs 41/54 (76%) memwright.

### Key Observations

- **Baseline recall degrades**: 100% early → 100% mid → 70.8% late. Facts from early
  turns are genuinely lost after context compression.
- **Memwright costs less**: $26 vs $29 (10% savings) despite 3.4x more compressions.
- **Memwright compresses more often**: Hooks add per-turn overhead (~2-4K tokens for
  SessionStart injection + PostToolUse capture), filling context faster.
- **Haiku gets combative**: When asked quiz-style recall questions mid-coding-session,
  Haiku refuses to answer ("I'm not answering test questions"). This is a benchmark
  design issue, not a memwright issue. Fix: rephrase as natural requests.

### Compression Timeline

```
Baseline:  turns [17, 35, 40, 171, 175, 177, 180]              (7 total)
Memwright: turns [24, 28, 30, 35, 38, 45, 53, 60, 68, 69, ... ] (24 total)
```

## Usage

```bash
# Quick test (10 turns)
python bench_claude.py --mode memwright --max-turns 10

# Full baseline run (requires memwright NOT installed)
poetry remove memwright
python bench_claude.py --mode baseline

# Full memwright run (installs if missing)
poetry add memwright
python bench_claude.py --mode memwright

# Run both sequentially + auto-compare
python bench_claude.py --mode both

# Compare existing results
python bench_claude.py --compare
python bench_claude.py --compare-files results1.json results2.json
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | baseline | `baseline`, `memwright`, or `both` |
| `--model` | haiku | Claude model (`haiku`, `sonnet`, `opus`) |
| `--max-turns` | 200 | Number of conversation turns |
| `--output-dir` | benchmark-logs | Where to save JSON results |
| `--compare` | | Compare latest baseline vs memwright results |

## Result Files

Each run produces a JSON file in `benchmark-logs/`:

```
bench-baseline-haiku-20260321-103238.json
bench-memwright-haiku-20260321-020351.json
```

Per-turn data includes: input/output/cache tokens, context %, cost, latency,
recall keywords expected vs found, compression detection.

## Known Issues

1. **Recall prompt phrasing**: Quiz-style questions ("What's my name?") cause Haiku
   to refuse mid-session. Need to rephrase as natural requests ("I need to update my
   profile — what's the name and company I should use?").
2. **Hook overhead**: Memwright hooks add ~2-4K tokens/turn, causing more frequent
   compressions. This is the cost of persistent memory — but facts survive compression.
3. **Compression detection**: Uses heuristic (effective context drops >40%). May miss
   some compression events or false-positive on cache misses.
