# tests/test_extras.py
"""Tests for the require_extra optional-dependency helper."""
from __future__ import annotations

import sys

import pytest

from agent_memory.store._extras import MissingExtraError, require_extra


def test_require_extra_returns_module_when_installed() -> None:
    mod = require_extra("json", extra="json-extra")
    assert mod is not None
    assert hasattr(mod, "dumps")


def test_require_extra_raises_actionable_error_when_missing() -> None:
    with pytest.raises(MissingExtraError) as exc:
        require_extra("definitely_not_a_real_module_xyz", extra="ghost")
    msg = str(exc.value)
    assert "definitely_not_a_real_module_xyz" in msg
    assert "memwright[ghost]" in msg
    assert "pip install" in msg


def test_missing_extra_error_is_import_error() -> None:
    """MissingExtraError must subclass ImportError so existing try/except ImportError blocks keep working."""
    assert issubclass(MissingExtraError, ImportError)


def test_require_extra_does_not_pollute_sys_modules_on_failure() -> None:
    name = "still_not_real_abc"
    sys.modules.pop(name, None)
    with pytest.raises(MissingExtraError):
        require_extra(name, extra="ghost")
    assert name not in sys.modules
