"""Runtime path resolution helpers.

Phase 3: dual-read. New installs use ``~/.attestor`` and the ``ATTESTOR_*``
env vars; existing ``~/.memwright`` installs (and ``MEMWRIGHT_*`` env vars)
keep working, with a one-time user warning pointing users at
``attestor migrate`` and a ``DeprecationWarning`` for legacy env vars.

Resolution order for ``resolve_store_path``:
  1. explicit ``override`` arg
  2. ``$ATTESTOR_PATH``
  3. ``$MEMWRIGHT_PATH`` (+ DeprecationWarning)
  4. ``~/.attestor`` if it exists
  5. ``~/.memwright`` if it exists and ``~/.attestor`` does not
     (+ one-time migrate warning)
  6. ``~/.attestor`` (new default)

The same ordering applies to ``resolve_data_dir`` against
``$ATTESTOR_DATA_DIR`` / ``$MEMWRIGHT_DATA_DIR``, and to ``resolve_cache_dir``
against ``~/.cache/attestor`` / ``~/.cache/memwright``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

from . import _branding as brand


# One-shot latch so users don't get the migrate warning on every recall ------
_WARNED_LEGACY_DIR_ONCE: bool = False


def _reset_warned_once() -> None:
    """Test hook — clears the one-shot latch between tests."""
    global _WARNED_LEGACY_DIR_ONCE
    _WARNED_LEGACY_DIR_ONCE = False


def _warn_legacy_dir_once(legacy_path: Path) -> None:
    global _WARNED_LEGACY_DIR_ONCE
    if _WARNED_LEGACY_DIR_ONCE:
        return
    _WARNED_LEGACY_DIR_ONCE = True
    warnings.warn(
        f"Using legacy memory store at {legacy_path}. "
        f"Run `attestor migrate` to copy it to ~/.{brand.CACHE_DIRNAME} "
        f"(the new default). The legacy path will keep working for now.",
        UserWarning,
        stacklevel=3,
    )


def _warn_legacy_env(legacy_name: str, new_name: str) -> None:
    warnings.warn(
        f"${legacy_name} is deprecated; use ${new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _new_default_store() -> Path:
    return Path.home() / brand.DEFAULT_STORE_DIRNAME


def _legacy_default_store() -> Path:
    return Path.home() / brand.LEGACY_STORE_DIRNAME


def resolve_store_path(override: Optional[str] = None) -> str:
    """Resolve the memory store path with dual-read compat.

    Precedence: ``override`` → ``$ATTESTOR_PATH`` → ``$MEMWRIGHT_PATH`` →
    existing ``~/.attestor`` → existing ``~/.memwright`` → new ``~/.attestor``.
    """
    if override:
        return os.path.expanduser(override)

    new_env = os.environ.get(brand.ENV_STORE_PATH)
    if new_env:
        return os.path.expanduser(new_env)

    legacy_env = os.environ.get(brand.LEGACY_ENV_STORE_PATH)
    if legacy_env:
        _warn_legacy_env(brand.LEGACY_ENV_STORE_PATH, brand.ENV_STORE_PATH)
        return os.path.expanduser(legacy_env)

    new_default = _new_default_store()
    if new_default.exists():
        return str(new_default)

    legacy_default = _legacy_default_store()
    if legacy_default.exists():
        _warn_legacy_dir_once(legacy_default)
        return str(legacy_default)

    return str(new_default)


def resolve_data_dir(override: Optional[str] = None) -> str:
    """Resolve the deployed-service data dir (Docker, App Runner, API).

    Same precedence as ``resolve_store_path`` but against
    ``$ATTESTOR_DATA_DIR`` / ``$MEMWRIGHT_DATA_DIR``.
    """
    if override:
        return os.path.expanduser(override)

    new_env = os.environ.get(brand.ENV_DATA_DIR)
    if new_env:
        return os.path.expanduser(new_env)

    legacy_env = os.environ.get(brand.LEGACY_ENV_DATA_DIR)
    if legacy_env:
        _warn_legacy_env(brand.LEGACY_ENV_DATA_DIR, brand.ENV_DATA_DIR)
        return os.path.expanduser(legacy_env)

    new_default = _new_default_store()
    if new_default.exists():
        return str(new_default)

    legacy_default = _legacy_default_store()
    if legacy_default.exists():
        _warn_legacy_dir_once(legacy_default)
        return str(legacy_default)

    return str(new_default)


def resolve_cache_dir() -> Path:
    """Cache directory for benchmarks, embedding models, etc.

    Prefers ``~/.cache/attestor``; falls back to ``~/.cache/memwright`` only
    when the legacy cache exists and the new one does not.
    """
    cache_root = Path.home() / ".cache"
    new_cache = cache_root / brand.CACHE_DIRNAME
    legacy_cache = cache_root / brand.LEGACY_CACHE_DIRNAME

    if new_cache.exists():
        return new_cache
    if legacy_cache.exists():
        return legacy_cache
    return new_cache
