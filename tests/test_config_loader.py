from pathlib import Path

import pytest

from agent_memory.config.loader import load_settings


def test_loads_defaults_when_no_file(tmp_path: Path):
    s = load_settings(store_path=tmp_path)
    assert s.backends == ["sqlite", "chroma", "networkx"]


def test_loads_from_toml(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'backends = ["arangodb"]\n[arangodb]\nport = 9999\n'
    )
    s = load_settings(store_path=tmp_path)
    assert s.backends == ["arangodb"]
    assert s.arangodb.port == 9999


def test_env_var_overrides_file(tmp_path: Path, monkeypatch):
    (tmp_path / "config.toml").write_text("default_token_budget = 1000\n")
    monkeypatch.setenv("MEMWRIGHT_DEFAULT_TOKEN_BUDGET", "5555")
    s = load_settings(store_path=tmp_path)
    assert s.default_token_budget == 5555


def test_profile_selection(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        '[profiles.prod]\n'
        'backends = ["arangodb"]\n'
    )
    s = load_settings(store_path=tmp_path, profile="prod")
    assert s.backends == ["arangodb"]


def test_unknown_profile_raises(tmp_path: Path):
    (tmp_path / "config.toml").write_text('backends = ["sqlite"]\n')
    with pytest.raises(ValueError, match="profile.*nope"):
        load_settings(store_path=tmp_path, profile="nope")
