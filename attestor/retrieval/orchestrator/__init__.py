"""Retrieval orchestrator package — split from a 1099-line module.

Public API (re-exported for back-compat with existing callers):

    from attestor.retrieval.orchestrator import (
        RetrievalOrchestrator,
        RetrievalRuntimeConfig,
    )

The split into ``core``, ``helpers``, ``postprocess``, ``debug``, and
``config`` submodules is purely organizational; the public surface and
behavior of the recall pipeline are byte-identical to the pre-split
``orchestrator.py``.
"""

from __future__ import annotations

from attestor.retrieval.orchestrator.config import (
    GRAPH_MAX_DEPTH,
    RetrievalRuntimeConfig,
)
from attestor.retrieval.orchestrator.core import RetrievalOrchestrator

__all__ = [
    "GRAPH_MAX_DEPTH",
    "RetrievalOrchestrator",
    "RetrievalRuntimeConfig",
]
