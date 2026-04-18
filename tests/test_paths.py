"""Path + env var resolution tests.

Covers ``resolve_store_path``, ``resolve_data_dir``, and ``resolve_cache_dir``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attestor import _paths


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` and ``~`` at an isolated temp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("ATTESTOR_PATH", "ATTESTOR_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestResolveStorePath:
    def test_override_beats_everything(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_PATH", str(fake_home / "env_attestor"))
        (fake_home / ".attestor").mkdir()

        assert _paths.resolve_store_path("/tmp/explicit") == "/tmp/explicit"

    def test_attestor_env_used(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_PATH", str(fake_home / "new"))

        assert _paths.resolve_store_path() == str(fake_home / "new")

    def test_default_is_attestor_dir(self, fake_home):
        assert _paths.resolve_store_path() == str(fake_home / ".attestor")


class TestResolveDataDir:
    def test_attestor_data_dir_env(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_DATA_DIR", "/srv/attestor")

        assert _paths.resolve_data_dir() == "/srv/attestor"

    def test_default_attestor_dir(self, fake_home):
        assert _paths.resolve_data_dir() == str(fake_home / ".attestor")


class TestResolveCacheDir:
    def test_default_is_attestor_cache(self, fake_home):
        assert _paths.resolve_cache_dir() == fake_home / ".cache" / "attestor"
