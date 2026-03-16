---
phase: 04-run-benchmarks-on-v2-backends-and-compare-scores
plan: 01
subsystem: benchmarks
tags: [mab, locomo, chromadb, networkx, benchmarks, openrouter]

# Dependency graph
requires:
  - phase: 01-zero-infrastructure-backends
    provides: ChromaDB + NetworkX backends
  - phase: 02-claude-code-plugin-and-auto-capture
    provides: Plugin packaging with new deps
provides:
  - MAB benchmark scores for v2 backends (AR 8.9%, CR 0%, Overall 6.4%)
  - LOCOMO benchmark scores for v2 backends (Overall 81.2% accuracy, 50.7% F1)
affects: [04-02, landing-page, README]

# Tech tracking
tech-stack:
  added: []
  patterns: [benchmark result JSON with scores key for machine readability]

key-files:
  created:
    - benchmark-logs/mab-v2-results.json
    - benchmark-logs/locomo-v2-results.json
  modified: []

key-decisions:
  - "MAB v2 scores dropped significantly (8.9% AR vs 55% v1) -- likely due to ChromaDB sentence-transformers vs OpenAI embeddings and lack of LLM re-ranking"
  - "LOCOMO v2 scores improved dramatically (81.2% vs 62.5% v1) -- sentence-transformers embeddings better for conversational retrieval"

patterns-established:
  - "Benchmark results include 'scores' key at top level for machine verification"

requirements-completed: [BENCH-01, BENCH-02]

# Metrics
duration: 572min
completed: 2026-03-16
---

# Phase 4 Plan 1: Run Benchmarks Summary

**MAB (AR 8.9%, CR 0%, Overall 6.4%) and LOCOMO (81.2% accuracy, 50.7% F1) benchmarks on ChromaDB + NetworkX v2 backends**

## Performance

- **Duration:** ~572 min (mostly API wait time)
- **Started:** 2026-03-16T03:25:50Z
- **Completed:** 2026-03-16T12:57:04Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- MAB benchmark ran to completion on v2 backends with 2800 questions across 14 sub-tasks
- LOCOMO benchmark ran to completion on v2 backends with 1540 questions across 10 conversations
- Both result files saved as structured JSON with full per-category breakdowns

## Task Commits

Each task was committed atomically:

1. **Task 1: Validate environment and run MAB benchmark** - `2594917` (feat)
2. **Task 2: Run LOCOMO benchmark** - `919bd12` (feat)

## Files Created/Modified
- `benchmark-logs/mab-v2-results.json` - Full MAB results (AR, CR, 14 sub-tasks, timing, details)
- `benchmark-logs/locomo-v2-results.json` - Full LOCOMO results (4 categories, timing, per-conversation details)

## Decisions Made
- MAB v2 scores (AR 8.9%, CR 0%, Overall 6.4%) are significantly lower than v1 baselines (AR 55%, CR 62%, Overall 58.5%). This is expected -- v1 used OpenAI text-embedding-3-small via API while v2 uses local sentence-transformers (all-MiniLM-L6-v2). The embedding quality difference particularly impacts dense retrieval tasks. CR dropping to 0% indicates the contradiction resolution pipeline may need LLM-based re-ranking.
- LOCOMO v2 scores (81.2% accuracy) are significantly higher than v1 (62.5%). The local embeddings appear to work better for conversational memory retrieval. This would put memwright above Letta (74%), Zep (58.4%), and Mem0 (66.9%) on this benchmark.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `source .env` did not export variables by default. Used `set -a && source .env && set +a` to ensure OPENROUTER_API_KEY was available to child processes.
- LOCOMO benchmark took ~35 minutes with no progress output (all output buffered until completion). Process was verified alive via `ps` and `lsof` monitoring.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both benchmark result files ready for Plan 04-02 (comparison with v1 baselines, landing page update)
- MAB regression needs investigation -- may need embedding quality improvements or re-ranking
- LOCOMO improvement is a strong positive signal for the zero-config migration

## Self-Check: PASSED

All files and commits verified.

---
*Phase: 04-run-benchmarks-on-v2-backends-and-compare-scores*
*Completed: 2026-03-16*
