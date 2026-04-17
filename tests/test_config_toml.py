from pathlib import Path

from agent_memory.utils.config import load_config


def test_load_toml(tmp_path: Path):
    """load_config prefers config.toml over config.json when both exist."""
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        'default_token_budget = 9000\n'
    )
    (tmp_path / "config.json").write_text('{"backends": ["arangodb"], "default_token_budget": 1}')

    cfg = load_config(tmp_path)
    assert cfg.backends == ["sqlite"]
    assert cfg.default_token_budget == 9000


def test_json_still_works(tmp_path: Path):
    """Existing JSON configs continue to load unchanged."""
    (tmp_path / "config.json").write_text('{"backends": ["sqlite"]}')
    cfg = load_config(tmp_path)
    assert cfg.backends == ["sqlite"]


def test_toml_with_nested_backend_section(tmp_path: Path):
    """TOML [backend] sections should be parsed into backend_configs."""
    (tmp_path / "config.toml").write_text(
        'backends = ["arangodb"]\n'
        '\n'
        '[arangodb]\n'
        'mode = "local"\n'
        'port = 8529\n'
    )
    cfg = load_config(tmp_path)
    assert cfg.backends == ["arangodb"]
    assert cfg.backend_configs["arangodb"]["mode"] == "local"
    assert cfg.backend_configs["arangodb"]["port"] == 8529


def test_missing_both_returns_defaults(tmp_path: Path):
    """When neither config.toml nor config.json exists, return defaults."""
    from agent_memory.utils.config import MemoryConfig
    cfg = load_config(tmp_path)
    assert cfg == MemoryConfig()


def test_malformed_toml_raises_valueerror(tmp_path: Path):
    """A malformed config.toml must raise ValueError with the file path in the message."""
    (tmp_path / "config.toml").write_text("backends = [unterminated")
    import pytest
    with pytest.raises(ValueError, match="Malformed TOML"):
        load_config(tmp_path)
