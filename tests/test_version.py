from unittest.mock import patch

import pytest
from importlib.metadata import PackageNotFoundError

import agent_memory
from agent_memory._version import get_version


def test_version_comes_from_package_metadata():
    """get_version() must delegate to importlib.metadata.version, not a literal."""
    with patch("agent_memory._version._pkg_version", return_value="9.9.9") as mock_pkg_version:
        assert get_version() == "9.9.9"
        mock_pkg_version.assert_called_once_with("memwright")


def test_fallback_when_package_not_installed():
    """get_version() falls back to a sentinel when the package is not registered."""
    with patch("agent_memory._version._pkg_version", side_effect=PackageNotFoundError):
        assert get_version() == "0.0.0+local"


def test_module_version_attribute_is_present():
    """agent_memory.__version__ is set at import time."""
    assert isinstance(agent_memory.__version__, str)
    assert agent_memory.__version__  # non-empty


def test_version_is_semver():
    """When running against an installed package, version must be semver-ish."""
    v = agent_memory.__version__
    if v == "0.0.0+local":
        pytest.skip("package not installed; fallback version in use")
    parts = v.split(".")
    assert len(parts) >= 2, v
    assert all(p[0].isdigit() for p in parts[:2]), v
