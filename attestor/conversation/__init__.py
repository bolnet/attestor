"""Attestor v4 conversation layer — round-level capture + extraction.

Per LongMemEval Finding 1 (roadmap §A.1), a *round* (one user message
+ one assistant message) is the optimal granularity for both storage
and retrieval. This package owns the end-to-end ingest path:

  ConversationTurn         — speaker-tagged, verbatim dataclass
  EpisodeRepo              — verbatim user+assistant turns into ``episodes``
  ConversationIngest       — orchestrator (Phase 3.5)
  apply_decisions          — ADD/UPDATE/INVALIDATE/NOOP through supersession

Extraction lives in ``attestor.extraction``; this package wires it.

This ``__init__`` deliberately re-exports only ``ConversationTurn`` (the
lightest, dependency-free piece). The heavier submodules (``apply``,
``episodes``, ``ingest``) transitively import ``attestor.extraction.*``
and would close an import cycle if pulled in eagerly here:

    extraction.conflict_resolver
      → extraction.round_extractor
        → conversation.turns           # triggers conversation/__init__
          → conversation.ingest        # ← cycle if imported here
            → extraction.conflict_resolver

Import the heavy submodules explicitly when you need them:

    from attestor.conversation.ingest import ConversationIngest
    from attestor.conversation.apply import apply_decisions
    from attestor.conversation.episodes import EpisodeRepo
"""

from attestor.conversation.turns import ConversationTurn

__all__ = ["ConversationTurn"]
