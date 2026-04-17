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


def test_config_show_errors_on_missing_file(tmp_path: Path):
    """`config show` must fail if config.json is missing (no silent default)."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "show", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no config.json" in result.stderr


def test_config_show_errors_on_malformed_json(tmp_path: Path):
    """`config show` must exit cleanly (no traceback) on malformed JSON."""
    (tmp_path / "config.json").write_text("{not json")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "show", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "malformed JSON" in result.stderr
    assert "Traceback" not in result.stderr


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
    assert result.returncode == 1
    assert "INVALID" in result.stderr
    assert "malformed JSON" in result.stderr


def test_config_validate_fails_on_empty_file(tmp_path: Path):
    """Empty config.json must be treated as invalid JSON, not valid."""
    (tmp_path / "config.json").write_text("")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "INVALID" in result.stderr


def test_config_validate_errors_on_missing_file(tmp_path: Path):
    """`config validate` must fail if config.json is missing (no silent default)."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no config.json" in result.stderr
