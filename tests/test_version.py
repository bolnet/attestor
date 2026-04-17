from importlib.metadata import version as pkg_version

import agent_memory


def test_version_matches_metadata():
    """agent_memory.__version__ must match the installed package metadata."""
    assert agent_memory.__version__ == pkg_version("memwright")


def test_version_is_semver():
    """Version must be a valid semver-ish string."""
    parts = agent_memory.__version__.split(".")
    assert len(parts) >= 2, agent_memory.__version__
    assert all(p[0].isdigit() for p in parts[:2])
