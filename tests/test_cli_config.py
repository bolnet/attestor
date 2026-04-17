import json
import subprocess
import sys
from pathlib import Path


def test_config_show_outputs_json(tmp_path: Path):
    """`memwright config show <path>` prints the resolved config as JSON."""
    cfg = {"backends": ["sqlite"], "default_token_budget": 8000}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "show", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    out = json.loads(result.stdout)
    assert out["backends"] == ["sqlite"]
    assert out["default_token_budget"] == 8000


def test_config_validate_passes(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"backends": ["sqlite"]}')
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "OK" in result.stdout


def test_config_validate_fails_on_bad_json(tmp_path: Path):
    (tmp_path / "config.json").write_text("{not json")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "INVALID" in result.stderr
