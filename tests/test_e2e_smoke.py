# tests/test_e2e_smoke.py
"""End-to-end smoke test: init -> add -> recall -> stats.

Requires a live Postgres (and optionally Neo4j). Skipped when env is unset.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "attestor.cli", *args],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"),
    reason="requires POSTGRES_URL to exercise a live backend",
)
def test_full_flow_postgres(tmp_path: Path) -> None:
    store = tmp_path / "store"

    _cli("init", str(store), "--backend", "postgres", "--non-interactive")
    assert (store / "config.toml").exists(), "init must write config.toml"

    # `doctor` is the validation step — prints the health report to stdout.
    doctor = _cli("doctor", str(store))
    assert "Document Store" in doctor.stdout or "PostgresBackend" in doctor.stdout, doctor.stdout

    _cli("add", str(store), "ArangoDB is a unified graph + document database")

    # `search` exercises the FTS path on Postgres.
    search = _cli("search", str(store), "arango")
    assert "ArangoDB" in search.stdout, search.stdout

    stats = _cli("stats", str(store))
    assert "Total memories: 1" in stats.stdout, stats.stdout
