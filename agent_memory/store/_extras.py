# agent_memory/store/_extras.py
"""Optional-dependency loader with actionable error messages.

Backends and infra modules call :func:`require_extra` at import time so that
end users who forgot ``pip install "memwright[<extra>]"`` get a single-line,
copy-paste-able fix instead of a raw ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtraError(ImportError):
    """Raised when an optional dependency is not installed.

    Subclasses :class:`ImportError` so existing ``try: ... except ImportError``
    blocks (e.g. ``core.py:_ensure_docker``) still degrade gracefully.
    """


def require_extra(module: str, *, extra: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingExtraError` pointing at ``extra``.

    Args:
        module: dotted module path, e.g. ``"docker"`` or ``"psycopg2"``.
        extra:  pyproject extra that installs the dependency, e.g. ``"docker"``.

    Returns:
        The imported module.

    Raises:
        MissingExtraError: if ``module`` cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"optional dependency '{module}' is not installed. "
            f"install it with:  pip install \"memwright[{extra}]\""
        ) from exc
