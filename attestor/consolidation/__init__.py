# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Sleep-time consolidation (Phase 7, roadmap §E).

Background worker that drains the per-episode consolidation queue,
re-extracts with a stronger model, and produces three artifacts:

  1. Refined facts via the existing ADD/UPDATE/INVALIDATE/NOOP path
  2. Session-level summaries (rolled into thread metadata)
  3. Cross-thread *pattern* reflection (stable preferences,
     contradictions for human review) — see ``pattern_reflection``.
  4. Cross-thread *distillation* reflection (compress N episodic
     memories into target_count semantic ones, supersede the
     originals, audit the chain) — see ``reflection``.

Public surface:
  ConsolidationQueue        — enqueue/dequeue/mark_done/mark_failed
  SleepTimeConsolidator     — per-episode worker
  ReflectionEngine          — cross-thread *pattern* synthesis
                              (LangMem shape; lives in
                              ``pattern_reflection``)
  run_reflection            — cross-thread *distillation* with
                              supersession + provenance (lives in
                              ``reflection``)
  ReflectionResult          — return value of ``run_reflection``
"""

from attestor.consolidation.consolidator import (
    ConsolidationResult,
    SleepTimeConsolidator,
)
from attestor.consolidation.pattern_reflection import (
    REFLECTION_PROMPT as PATTERN_REFLECTION_PROMPT,
    ChangedBelief,
    Contradiction,
    PatternReflectionResult,
    ReflectionEngine,
    StablePattern,
)
from attestor.consolidation.queue import ConsolidationQueue, QueuedEpisode
from attestor.consolidation.reflection import (
    DEFAULT_TARGET_COUNT,
    REFLECTION_PROMPT,
    ReflectionResult,
    run_reflection,
)

__all__ = [
    "ConsolidationQueue",
    "QueuedEpisode",
    "ConsolidationResult",
    "SleepTimeConsolidator",
    # Distillation reflection (PR feat/reflection-consolidation)
    "DEFAULT_TARGET_COUNT",
    "REFLECTION_PROMPT",
    "ReflectionResult",
    "run_reflection",
    # Pattern-surfacing reflection (legacy phase 7.3)
    "PATTERN_REFLECTION_PROMPT",
    "ReflectionEngine",
    "PatternReflectionResult",
    "StablePattern",
    "ChangedBelief",
    "Contradiction",
]
