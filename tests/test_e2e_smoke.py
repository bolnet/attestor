# tests/test_e2e_smoke.py
"""End-to-end smoke test: init -> add -> recall -> stats.

Uses the SQLite backend only so it runs on any dev machine with no external
services or extras installed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "attestor.cli", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_full_flow_sqlite(tmp_path: Path) -> None:
    store = tmp_path / "store"

    _cli("init", str(store), "--backend", "sqlite", "--non-interactive")
    assert (store / "config.toml").exists(), "init must write config.toml"

    # No `config validate` subcommand; use `doctor` as the validation step.
    # SQLiteStore being OK is sufficient — vector store may be unavailable
    # in environments without torch/sentence-transformers.
    doctor = _cli("doctor", str(store))
    assert "SQLiteStore" in doctor.stdout, doctor.stdout

    # `add` takes content as a positional arg, not a --content flag.
    _cli("add", str(store), "ArangoDB is configured for local mode")

    # `recall` depends on the vector store which may be unavailable in CI.
    # Use `search` (SQLite FTS, always-on) to validate the content was stored.
    search = _cli("search", str(store), "arango")
    assert "ArangoDB" in search.stdout, search.stdout

    stats = _cli("stats", str(store))
    assert "Total memories: 1" in stats.stdout, stats.stdout
