"""Runtime path resolution helpers.

Phase 0 preserves legacy behavior exactly: environment reads still use
MEMWRIGHT_* variable names and the default store still lives at
``~/.memwright``. Centralizing these reads here means Phase 3 can add
dual-read logic (ATTESTOR_PATH first, fall back to MEMWRIGHT_PATH,
then ``~/.attestor``, then ``~/.memwright``) without touching call-sites.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import _branding as brand


def _legacy_default_store() -> str:
    """Legacy ``~/.memwright`` default, expanded."""
    return os.path.expanduser(f"~/{brand.LEGACY_STORE_DIRNAME}")


def _legacy_default_data_dir() -> str:
    """Legacy default data dir (same as store path on local installs)."""
    return os.path.expanduser(f"~/{brand.LEGACY_STORE_DIRNAME}")


def resolve_store_path(override: Optional[str] = None) -> str:
    """Resolve the memory store path.

    Resolution order (Phase 0 — legacy only):
      1. ``override`` if provided (e.g., from --path CLI flag)
      2. $MEMWRIGHT_PATH
      3. ``~/.memwright``

    Phase 3 will insert $ATTESTOR_PATH ahead of $MEMWRIGHT_PATH and
    ``~/.attestor`` ahead of ``~/.memwright``.
    """
    if override:
        return os.path.expanduser(override)
    env_path = os.environ.get(brand.LEGACY_ENV_STORE_PATH)
    if env_path:
        return os.path.expanduser(env_path)
    return _legacy_default_store()


def resolve_data_dir(override: Optional[str] = None) -> str:
    """Resolve the deployed-service data dir (Docker, App Runner, API).

    Resolution order (Phase 0 — legacy only):
      1. ``override`` if provided
      2. $MEMWRIGHT_DATA_DIR
      3. ``~/.memwright``
    """
    if override:
        return os.path.expanduser(override)
    env_path = os.environ.get(brand.LEGACY_ENV_DATA_DIR)
    if env_path:
        return os.path.expanduser(env_path)
    return _legacy_default_data_dir()


def resolve_cache_dir() -> Path:
    """Cache directory for benchmarks, embedding models, etc.

    Phase 0: ``~/.cache/memwright``. Phase 3 will migrate to ``~/.cache/attestor``
    with a fallback for existing users.
    """
    return Path.home() / ".cache" / brand.LEGACY_CACHE_DIRNAME
