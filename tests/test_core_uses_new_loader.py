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
