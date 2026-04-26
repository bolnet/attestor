"""Sleep-time consolidation (Phase 7, roadmap §E).

Background worker that drains the per-episode consolidation queue,
re-extracts with a stronger model, and produces three artifacts:

  1. Refined facts via the existing ADD/UPDATE/INVALIDATE/NOOP path
  2. Session-level summaries (rolled into thread metadata)
  3. Cross-thread reflection memories (stable preferences, contradictions
     for human review)

Public surface:
  ConsolidationQueue   — enqueue/dequeue/mark_done/mark_failed
  SleepTimeConsolidator— per-episode worker
  ReflectionEngine     — cross-thread synthesis (Phase 7.3)
"""

from attestor.consolidation.queue import ConsolidationQueue, QueuedEpisode

__all__ = ["ConsolidationQueue", "QueuedEpisode"]
