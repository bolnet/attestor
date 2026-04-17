from unittest.mock import MagicMock, patch

import agent_memory.store.registry as registry_module
from agent_memory.store.registry import BACKEND_REGISTRY, discover_backends


def _reset_discovery_state():
    """Reset module-level discovery flag for test isolation."""
    registry_module._backends_discovered = False


def test_builtins_present_after_discover():
    discover_backends()
    for name in ("sqlite", "chroma", "networkx", "arangodb", "postgres"):
        assert name in BACKEND_REGISTRY, f"{name} should be discovered"


def test_discover_is_idempotent():
    discover_backends()
    snapshot = dict(BACKEND_REGISTRY)
    discover_backends()
    assert BACKEND_REGISTRY == snapshot


def test_bad_plugin_logs_warning_and_is_skipped(caplog):
    """A plugin whose ep.load() raises should be logged and skipped, not crash discovery."""
    bad_ep = MagicMock()
    bad_ep.name = "bad-plugin-xyz"
    bad_ep.load.side_effect = RuntimeError("boom")

    _reset_discovery_state()
    with patch("importlib.metadata.entry_points", return_value=[bad_ep]):
        with caplog.at_level("WARNING", logger="agent_memory"):
            registry_module.discover_backends()

    assert "bad-plugin-xyz" not in registry_module.BACKEND_REGISTRY
    assert "bad-plugin-xyz" in caplog.text


def test_malformed_plugin_shape_is_skipped(caplog):
    """A plugin that loads but returns a non-dict or dict missing required keys is skipped with a warning."""
    malformed_ep = MagicMock()
    malformed_ep.name = "malformed-plugin-xyz"
    malformed_ep.load.return_value = {"module": "x"}  # missing class/roles/init_style

    _reset_discovery_state()
    with patch("importlib.metadata.entry_points", return_value=[malformed_ep]):
        with caplog.at_level("WARNING", logger="agent_memory"):
            registry_module.discover_backends()

    assert "malformed-plugin-xyz" not in registry_module.BACKEND_REGISTRY
    assert "malformed-plugin-xyz" in caplog.text
    assert "invalid shape" in caplog.text


def test_builtin_not_overwritten_by_plugin():
    """A third-party ep named 'sqlite' must NOT overwrite the built-in entry."""
    original = dict(registry_module.BACKEND_REGISTRY["sqlite"])

    hostile_ep = MagicMock()
    hostile_ep.name = "sqlite"
    hostile_ep.load.return_value = {
        "module": "evil.module",
        "class": "Evil",
        "roles": {"document"},
        "init_style": "path",
    }

    _reset_discovery_state()
    with patch("importlib.metadata.entry_points", return_value=[hostile_ep]):
        registry_module.discover_backends()

    assert registry_module.BACKEND_REGISTRY["sqlite"] == original
