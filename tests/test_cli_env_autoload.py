# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for the CLI auto-loading ``<store>/.env`` (bare-command parity)."""
from __future__ import annotations

import argparse
import os

import pytest

from attestor.cli.main import _autoload_store_env

pytestmark = pytest.mark.unit


def test_autoload_loads_store_env(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("ATTESTOR_AUTOLOAD_PROBE=loaded\n")
    monkeypatch.delenv("ATTESTOR_AUTOLOAD_PROBE", raising=False)

    _autoload_store_env(argparse.Namespace(path=str(tmp_path)))

    assert os.environ.get("ATTESTOR_AUTOLOAD_PROBE") == "loaded"


def test_autoload_does_not_override_existing(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("ATTESTOR_AUTOLOAD_PROBE2=fromfile\n")
    monkeypatch.setenv("ATTESTOR_AUTOLOAD_PROBE2", "fromshell")

    _autoload_store_env(argparse.Namespace(path=str(tmp_path)))

    # setdefault → an already-exported shell value wins (don't clobber).
    assert os.environ["ATTESTOR_AUTOLOAD_PROBE2"] == "fromshell"


def test_autoload_noop_without_path() -> None:
    # No path attr / None → no error, nothing loaded.
    _autoload_store_env(argparse.Namespace(path=None))
    _autoload_store_env(argparse.Namespace())


def test_autoload_noop_when_no_env_file(tmp_path) -> None:
    # Store dir exists but has no .env → silent no-op.
    _autoload_store_env(argparse.Namespace(path=str(tmp_path)))
