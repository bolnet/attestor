"""Deprecated shim for the renamed ``attestor`` package.

Installing ``memwright`` now pulls ``attestor`` as a dependency and re-exports
its public API under the ``memwright`` namespace so existing imports keep
working. This shim emits a ``DeprecationWarning`` on import and will be
removed in v3.2. Migrate with:

    pip install attestor
    # and replace `from memwright import X` with `from attestor import X`
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`memwright` has been renamed to `attestor`. "
    "Install `attestor` (`pip install attestor`) and replace "
    "`from memwright import X` with `from attestor import X`. "
    "This shim will be removed in v3.2.",
    DeprecationWarning,
    stacklevel=2,
)

from attestor import AgentMemory, Memory, RetrievalResult  # noqa: E402,F401
from attestor import __version__ as _attestor_version  # noqa: E402

__version__ = _attestor_version
__all__ = ["AgentMemory", "Memory", "RetrievalResult", "__version__"]
