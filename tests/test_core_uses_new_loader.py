from pathlib import Path

from agent_memory import AgentMemory


def test_core_reads_toml_via_new_loader(tmp_path: Path):
    """AgentMemory should respect TOML configs."""
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        'default_token_budget = 7777\n'
    )
    with AgentMemory(tmp_path) as mem:
        assert mem.config.default_token_budget == 7777


def test_core_respects_env_var_override(tmp_path: Path, monkeypatch):
    """MEMWRIGHT_* env vars should flow through load_settings into mem.config."""
    (tmp_path / "config.toml").write_text("default_token_budget = 1000\n")
    monkeypatch.setenv("MEMWRIGHT_DEFAULT_TOKEN_BUDGET", "5555")
    with AgentMemory(tmp_path) as mem:
        assert mem.config.default_token_budget == 5555


def test_core_still_reads_legacy_json(tmp_path: Path):
    """Existing JSON-only configs must keep working after the refactor."""
    (tmp_path / "config.json").write_text('{"default_token_budget": 4242}')
    with AgentMemory(tmp_path) as mem:
        assert mem.config.default_token_budget == 4242


def test_save_config_writes_toml_when_toml_exists(tmp_path: Path):
    """After fix: save_config preserves TOML as the authoritative format."""
    (tmp_path / "config.toml").write_text("default_token_budget = 1000\n")
    with AgentMemory(tmp_path) as mem:
        mem.config.default_token_budget = 9000
        from agent_memory.utils.config import save_config
        save_config(tmp_path, mem.config)
    # TOML should now reflect the new value; JSON should NOT have been written
    assert "9000" in (tmp_path / "config.toml").read_text()
    assert not (tmp_path / "config.json").exists()
