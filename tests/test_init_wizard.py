"""Tests for attestor.init_wizard.

End-to-end init flow requires a live Postgres backend now that the
embedded stack is gone. Skipped when POSTGRES_URL is unset.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import tomlkit

from attestor.init_wizard import (
    SUPPORTED_BACKENDS,
    InitResult,
    init_store,
    init_store_interactive,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"),
    reason="init_wizard tests require POSTGRES_URL (canonical PG+Pinecone+Neo4j stack only)",
)


@pytest.fixture
def tmp_store() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "store"


class TestInitStore:
    def test_postgres_with_url(self, tmp_store: Path):
        opts = {"url": "postgresql://localhost:5432"}
        result = init_store(tmp_store, backend="postgres", backend_options=opts)
        assert isinstance(result, InitResult)
        assert result.backend == "postgres"
        assert result.config_path == tmp_store / "config.toml"
        assert result.verified is False

        doc = tomlkit.parse(result.config_path.read_text())
        assert list(doc["backends"]) == ["postgres", "pinecone", "neo4j"]
        assert doc["postgres"]["url"] == "postgresql://localhost:5432"
        assert doc["default_token_budget"] == 16000

    def test_unsupported_backend_raises(self, tmp_store: Path):
        with pytest.raises(ValueError, match="Unsupported backend"):
            init_store(tmp_store, backend="mongodb")

    def test_duplicate_config_raises(self, tmp_store: Path):
        opts = {"url": "postgresql://localhost:5432"}
        init_store(tmp_store, backend="postgres", backend_options=opts)
        with pytest.raises(FileExistsError):
            init_store(tmp_store, backend="postgres", backend_options=opts)

    def test_legacy_json_blocks_init(self, tmp_store: Path):
        tmp_store.mkdir(parents=True)
        (tmp_store / "config.json").write_text("{}")
        opts = {"url": "postgresql://localhost:5432"}
        with pytest.raises(FileExistsError):
            init_store(tmp_store, backend="postgres", backend_options=opts)

    def test_verify_success(self, tmp_store: Path):
        opts = {"url": "postgresql://localhost:5432"}
        with patch(
            "attestor.init_wizard._verify_store",
            return_value=(True, None),
        ):
            result = init_store(
                tmp_store, backend="postgres", backend_options=opts, verify=True,
            )
        assert result.verified is True
        assert result.config_path.exists()

    def test_verify_rollback_on_failure(self, tmp_store: Path):
        opts = {"url": "postgresql://localhost:5432"}
        with patch(
            "attestor.init_wizard._verify_store",
            return_value=(False, "RuntimeError: db locked"),
        ):
            with pytest.raises(RuntimeError, match="rolled back"):
                init_store(
                    tmp_store, backend="postgres", backend_options=opts, verify=True,
                )
        assert not (tmp_store / "config.toml").exists()

    def test_postgres_requires_url(self, tmp_store: Path):
        with pytest.raises(ValueError, match="requires a 'url'"):
            init_store(tmp_store, backend="postgres")

    def test_credentials_stripped_from_config(self, tmp_store: Path):
        opts = {
            "url": "postgresql://localhost:5432",
            "auth_username": "postgres",
            "auth_password": "hunter2",
        }
        init_store(tmp_store, backend="postgres", backend_options=opts)
        text = (tmp_store / "config.toml").read_text()
        assert "hunter2" not in text
        assert "auth_password" not in text
        # Non-credential fields remain.
        assert "auth_username" in text

    def test_config_toml_has_restrictive_permissions(self, tmp_store: Path):
        import os as _os
        import stat

        opts = {"url": "postgresql://localhost:5432"}
        init_store(tmp_store, backend="postgres", backend_options=opts)
        mode = stat.S_IMODE(_os.stat(tmp_store / "config.toml").st_mode)
        # World and group bits must be clear on POSIX.
        assert mode & 0o077 == 0, f"config.toml mode too permissive: {oct(mode)}"

    def test_supported_backends_contract(self):
        # Single canonical stack — postgres is the only init-wizard entry point.
        assert set(SUPPORTED_BACKENDS) == {"postgres"}


class TestInteractiveWizard:
    def test_postgres_interactive(self, tmp_store: Path, monkeypatch):
        prompts = iter(["postgres", "postgresql://localhost:5432"])
        monkeypatch.setattr("builtins.input", lambda _="": next(prompts))

        result = init_store_interactive(tmp_store)
        text = (tmp_store / "config.toml").read_text()

        assert result.backend == "postgres"
        assert "postgresql://localhost:5432" in text


class TestInitCLIFlags:
    def test_cli_init_writes_config_toml(self, tmp_store: Path):
        from attestor.cli import main

        main([
            "init", str(tmp_store),
            "--non-interactive",
            "--backend", "postgres",
            "--postgres-url", "postgresql://localhost:5432",
        ])
        assert (tmp_store / "config.toml").exists()
        doc = tomlkit.parse((tmp_store / "config.toml").read_text())
        assert list(doc["backends"]) == ["postgres", "pinecone", "neo4j"]

    def test_cli_init_verify_rolls_back_on_failure(self, tmp_store: Path):
        from attestor.cli import main

        with patch(
            "attestor.init_wizard._verify_store",
            return_value=(False, "RuntimeError: boom"),
        ):
            with pytest.raises(SystemExit):
                main([
                    "init",
                    str(tmp_store),
                    "--non-interactive",
                    "--backend", "postgres",
                    "--postgres-url", "postgresql://localhost:5432",
                    "--verify",
                ])
        assert not (tmp_store / "config.toml").exists()

    def test_cli_setup_claude_code_install_writes_settings(
        self, tmp_store: Path, tmp_path: Path, monkeypatch
    ):
        from attestor.cli import main

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        main(["setup-claude-code", str(tmp_store), "--install"])

        import json

        settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
        assert "memory" in settings["mcpServers"]
        assert settings["mcpServers"]["memory"]["args"][:2] == ["mcp", "--path"]

    def test_corrupt_settings_json_is_backed_up(
        self, tmp_store: Path, tmp_path: Path, monkeypatch, capsys
    ):
        """Malformed ~/.claude/settings.json must be backed up, not silently wiped."""
        from attestor.cli import main

        fake_home = tmp_path / "home"
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        corrupt_payload = '{"not valid json'
        settings_path.write_text(corrupt_payload)

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        main(["setup-claude-code", str(tmp_store), "--install"])

        backup = settings_path.with_suffix(".json.bak")
        assert backup.exists()
        assert backup.read_text() == corrupt_payload
        # New settings.json is valid JSON with the memory entry.
        import json

        new_settings = json.loads(settings_path.read_text())
        assert "memory" in new_settings["mcpServers"]

    def test_init_toml_config_is_read_by_load_config(self, tmp_store: Path):
        """Contract: load_config must honor config.toml written by the wizard."""
        from attestor.init_wizard import init_store
        from attestor.utils.config import load_config

        opts = {"url": "postgresql://localhost:5432"}
        init_store(tmp_store, backend="postgres", backend_options=opts)
        cfg = load_config(tmp_store)
        assert cfg.backends == ["postgres", "pinecone", "neo4j"]
        assert cfg.default_token_budget == 16000
