"""Helper for gating optional backend imports with actionable error messages."""

from __future__ import annotations

import importlib
from typing import Any


def require_extra(module_name: str, *, extra: str, package: str = "memwright") -> Any:
    """Import a module, raising an actionable ImportError if it's missing.

    Args:
        module_name: The module to import (e.g., "arango").
        extra: The pyproject.toml extras name (e.g., "arangodb").
        package: The distribution name. Defaults to "memwright".

    Returns:
        The imported module.

    Raises:
        ImportError: With a message telling the user exactly which extras to install.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for backend {extra!r}: {module_name}. "
            f"Install with: pip install '{package}[{extra}]'"
        ) from e
