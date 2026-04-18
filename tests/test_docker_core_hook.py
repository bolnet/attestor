# tests/test_docker_core_hook.py
"""Verify core._ensure_docker behavior when docker=true but extra is missing."""
from __future__ import annotations

import sys

import pytest

from attestor.core import AgentMemory
from attestor.store._extras import MissingExtraError


def test_ensure_docker_is_noop_when_docker_flag_false(tmp_path, monkeypatch):
    """Default path must never import infra.docker."""
    monkeypatch.setitem(sys.modules, "attestor.infra.docker", None)
    mem = AgentMemory(str(tmp_path))  # docker flag not set anywhere
    mem._ensure_docker("arangodb", {"mode": "local"})  # must not raise
    mem.close()


def test_ensure_docker_raises_actionable_error_when_flag_true_but_extra_missing(tmp_path, monkeypatch):
    """If the user set docker=true but didn't install [docker], surface the fix."""
    # Force fresh import of infra.docker so require_extra re-evaluates
    monkeypatch.delitem(sys.modules, "attestor.infra.docker", raising=False)
    # Force require_extra to fail even if docker SDK is installed
    monkeypatch.setitem(sys.modules, "docker", None)
    mem = AgentMemory(str(tmp_path))
    with pytest.raises(MissingExtraError, match=r"attestor\[docker\]"):
        mem._ensure_docker("arangodb", {"mode": "local", "docker": True})
    mem.close()
