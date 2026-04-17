"""Single source of truth for version. Reads from installed package metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version


def get_version() -> str:
    try:
        return _pkg_version("memwright")
    except PackageNotFoundError:
        return "0.0.0+local"
