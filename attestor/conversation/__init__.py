"""Attestor v4 conversation layer — round-level capture + extraction.

Per LongMemEval Finding 1 (roadmap §A.1), a *round* (one user message
+ one assistant message) is the optimal granularity for both storage
and retrieval. This package owns the end-to-end ingest path:

  ConversationTurn         — speaker-tagged, verbatim dataclass
  EpisodeRepo              — verbatim user+assistant turns into ``episodes``
  ConversationIngest       — orchestrator (Phase 3.5)
  apply_decisions          — ADD/UPDATE/INVALIDATE/NOOP through supersession

Extraction lives in ``attestor.extraction``; this package wires it.
"""

from attestor.conversation.episodes import Episode, EpisodeRepo
from attestor.conversation.turns import ConversationTurn

__all__ = [
    "ConversationTurn",
    "Episode",
    "EpisodeRepo",
]
