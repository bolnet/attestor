"""Attestor core — split from a 1563-line module into a cohesive package.

Public surface (unchanged):
    AgentMemory     — the main embedded API
    ResolvedContext — identity tuple returned by ``AgentMemory._resolve``

Re-exports (test surface — ``test_embedder_dim_check.py`` patches these
through the ``attestor.core`` namespace, so they must be importable here):
    DEFAULT_BACKENDS, instantiate_backend, resolve_backends
"""

from __future__ import annotations

# Registry symbols are re-exported at the package level because:
#   1. The legacy ``attestor.core`` module exposed them (tests patch
#      ``attestor.core.instantiate_backend`` / ``resolve_backends`` /
#      ``DEFAULT_BACKENDS`` — see tests/test_embedder_dim_check.py).
#   2. ``agent_memory.AgentMemory.__init__`` looks them up via this
#      package (``attestor.core``) so test patches take effect.
# These imports must run BEFORE ``agent_memory`` is imported.
from attestor.store.registry import (  # noqa: F401
    DEFAULT_BACKENDS,
    instantiate_backend,
    resolve_backends,
)

from attestor.core.agent_memory import AgentMemory
from attestor.core.identity_service import ResolvedContext

__all__ = [
    "AgentMemory",
    "ResolvedContext",
    "DEFAULT_BACKENDS",
    "instantiate_backend",
    "resolve_backends",
]
