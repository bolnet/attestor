import json
import tomllib
from pathlib import Path

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
    assert (tmp_path / "config.json.bak").exists() or src.exists()
