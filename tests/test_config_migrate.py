import json
import tomllib
from pathlib import Path

import pytest

from agent_memory.config.migrate import migrate_json_to_toml


def test_migrate_writes_equivalent_toml(tmp_path: Path):
    src = tmp_path / "config.json"
    src.write_text(json.dumps({
        "backends": ["arangodb"],
        "default_token_budget": 16000,
        "arangodb": {"mode": "cloud", "url": "http://localhost:8529"},
    }))

    out = migrate_json_to_toml(src)
    assert out == tmp_path / "config.toml"

    with open(out, "rb") as f:
        data = tomllib.load(f)
    assert data["backends"] == ["arangodb"]
    assert data["arangodb"]["mode"] == "cloud"


def test_migrate_preserves_original(tmp_path: Path):
    src = tmp_path / "config.json"
    src.write_text('{"backends": ["sqlite"]}')
    migrate_json_to_toml(src)
    assert src.exists(), "original JSON should be kept (rename to .bak instead)"
    assert (tmp_path / "config.json.bak").exists()


def test_migrate_rejects_non_json_suffix(tmp_path: Path):
    """Passing a .toml source path must raise ValueError."""
    src = tmp_path / "config.toml"
    src.write_text("")
    with pytest.raises(ValueError):
        migrate_json_to_toml(src)


def test_migrate_raises_on_malformed_json(tmp_path: Path):
    """Writing invalid JSON then calling migrate must raise json.JSONDecodeError."""
    src = tmp_path / "config.json"
    src.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        migrate_json_to_toml(src)


def test_migrate_skips_none_values(tmp_path: Path):
    """A JSON null value must not appear as '= null' in the TOML output."""
    src = tmp_path / "config.json"
    src.write_text(json.dumps({"optional_key": None, "name": "memwright"}))
    out = migrate_json_to_toml(src)

    toml_text = out.read_text()
    assert "null" not in toml_text, "TOML output must not contain 'null'"

    with open(out, "rb") as f:
        data = tomllib.load(f)
    assert "optional_key" not in data, "Null-valued key must be absent from parsed TOML"
    assert data["name"] == "memwright"


def test_migrate_escapes_quotes_and_backslashes(tmp_path: Path):
    """String values with double-quotes and backslashes must round-trip correctly."""
    original = 'say "hi" \\n'
    src = tmp_path / "config.json"
    src.write_text(json.dumps({"greeting": original}))
    out = migrate_json_to_toml(src)

    with open(out, "rb") as f:
        data = tomllib.load(f)
    assert data["greeting"] == original, (
        f"Expected {original!r}, got {data['greeting']!r}"
    )
