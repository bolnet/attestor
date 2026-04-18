"""Runtime path resolution helpers.

Resolution order for ``resolve_store_path``:
  1. explicit ``override`` arg
  2. ``$ATTESTOR_PATH``
  3. ``~/.attestor``

The same ordering applies to ``resolve_data_dir`` against
``$ATTESTOR_DATA_DIR``, and to ``resolve_cache_dir`` against
``~/.cache/attestor``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import _branding as brand


def _default_store() -> Path:
    return Path.home() / brand.DEFAULT_STORE_DIRNAME


def resolve_store_path(override: Optional[str] = None) -> str:
    """Resolve the memory store path.

    Precedence: ``override`` → ``$ATTESTOR_PATH`` → ``~/.attestor``.
    """
    if override:
        return os.path.expanduser(override)

    env = os.environ.get(brand.ENV_STORE_PATH)
    if env:
        return os.path.expanduser(env)

    return str(_default_store())


def resolve_data_dir(override: Optional[str] = None) -> str:
    """Resolve the deployed-service data dir (Docker, App Runner, API).

    Precedence: ``override`` → ``$ATTESTOR_DATA_DIR`` → ``~/.attestor``.
    """
    if override:
        return os.path.expanduser(override)

    env = os.environ.get(brand.ENV_DATA_DIR)
    if env:
        return os.path.expanduser(env)

    return str(_default_store())


def resolve_cache_dir() -> Path:
    """Cache directory for benchmarks, embedding models, etc."""
    return Path.home() / ".cache" / brand.CACHE_DIRNAME
