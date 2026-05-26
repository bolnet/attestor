# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for ``attestor quickstart`` config/.env writers (no docker/wiring)."""
from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import pytest

from attestor.cli.commands.quickstart import (
    DEFAULT_PASSWORD,
    _default_env,
    _ensure_env,
    _write_config_toml,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_default_env_has_all_required_keys(tmp_path: Path) -> None:
    env = _default_env(tmp_path)
    assert set(env) == {
        "PGPASSWORD",
        "POSTGRES_PASSWORD",
        "NEO4J_PASSWORD",
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ATTESTOR_CONFIG",
    }
    # Postgres password var (config.toml references $PGPASSWORD) must be set,
    # and it must equal POSTGRES_PASSWORD (the compose var) — same value, both names.
    assert env["PGPASSWORD"] == DEFAULT_PASSWORD
    assert env["POSTGRES_PASSWORD"] == DEFAULT_PASSWORD
    # Pinecone Local needs a non-empty key (the value is ignored by the emulator).
    assert env["PINECONE_API_KEY"] == "local"
    assert env["OPENAI_API_KEY"] == "ollama"
    assert env["ATTESTOR_CONFIG"] == str(tmp_path / "attestor.yaml")


def test_write_config_toml_is_three_role_with_env_refs(tmp_path: Path) -> None:
    cfg = _write_config_toml(tmp_path)
    text = cfg.read_text()
    # Secrets are $ENV_VAR refs, never plaintext.
    assert 'password = "$PGPASSWORD"' in text
    assert 'password = "$NEO4J_PASSWORD"' in text
    assert "v4 = true" in text
    # Canonical three roles — Postgres (document) + Pinecone (vector) + Neo4j (graph).
    assert "[postgres]" in text
    assert "[neo4j]" in text
    assert "[pinecone]" in text
    assert 'backends = ["postgres", "pinecone", "neo4j"]' in text
    # Pinecone Local routed for the vector role at the default host + 1024-D index.
    assert "localhost:5080" in text
    assert "dimension = 1024" in text
    # 0o600 on POSIX.
    mode = stat.S_IMODE(cfg.stat().st_mode)
    assert mode == 0o600


def test_write_config_toml_backs_up_existing(tmp_path: Path) -> None:
    _write_config_toml(tmp_path)
    _write_config_toml(tmp_path)  # second run must preserve the prior file
    assert (tmp_path / "config.toml.bak").exists()
    assert (tmp_path / "config.toml").exists()


def test_ensure_env_writes_then_is_idempotent(tmp_path: Path) -> None:
    env_path = _ensure_env(tmp_path)
    first = env_path.read_text()
    key_lines = [ln for ln in first.splitlines() if "=" in ln and not ln.startswith("#")]
    assert len(key_lines) == 7
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    # Re-run: no new keys, no duplicates.
    _ensure_env(tmp_path)
    second = env_path.read_text()
    key_lines_2 = [ln for ln in second.splitlines() if "=" in ln and not ln.startswith("#")]
    assert len(key_lines_2) == 7


def test_ensure_env_preserves_user_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("PGPASSWORD=custom-secret\n")
    _ensure_env(tmp_path)
    text = env_path.read_text()
    # Existing value preserved (not clobbered); missing keys appended.
    assert "PGPASSWORD=custom-secret" in text
    assert "NEO4J_PASSWORD=" in text
    assert text.count("PGPASSWORD=") == 1
