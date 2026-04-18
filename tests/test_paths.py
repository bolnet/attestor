"""Phase 3: dual-read path + env var resolution.

Covers all 5 branches of ``resolve_store_path``, plus parallel semantics
for ``resolve_data_dir`` and ``resolve_cache_dir``. The resolver must prefer
Attestor-branded inputs, transparently fall back to legacy Memwright ones,
and loudly warn whenever a legacy source is used.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest import mock

import pytest

from attestor import _paths


# Path resolution branches (order matters — tests assert precedence) ----------


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` and ``~`` at an isolated temp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ensure legacy env vars don't leak in from the host
    for name in (
        "ATTESTOR_PATH",
        "MEMWRIGHT_PATH",
        "ATTESTOR_DATA_DIR",
        "MEMWRIGHT_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    _paths._reset_warned_once()
    return tmp_path


class TestResolveStorePath:
    def test_override_beats_everything(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_PATH", str(fake_home / "env_attestor"))
        monkeypatch.setenv("MEMWRIGHT_PATH", str(fake_home / "env_memwright"))
        (fake_home / ".attestor").mkdir()
        (fake_home / ".memwright").mkdir()

        assert _paths.resolve_store_path("/tmp/explicit") == "/tmp/explicit"

    def test_attestor_env_preferred_over_legacy_env(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_PATH", str(fake_home / "new"))
        monkeypatch.setenv("MEMWRIGHT_PATH", str(fake_home / "legacy"))

        assert _paths.resolve_store_path() == str(fake_home / "new")

    def test_legacy_env_emits_deprecation_warning(self, fake_home, monkeypatch):
        monkeypatch.setenv("MEMWRIGHT_PATH", str(fake_home / "legacy_env"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _paths.resolve_store_path()

        assert result == str(fake_home / "legacy_env")
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "expected DeprecationWarning when MEMWRIGHT_PATH is used"
        assert "ATTESTOR_PATH" in str(deps[0].message)

    def test_attestor_dir_used_when_present(self, fake_home):
        (fake_home / ".attestor").mkdir()

        assert _paths.resolve_store_path() == str(fake_home / ".attestor")

    def test_attestor_dir_beats_legacy_dir(self, fake_home):
        (fake_home / ".attestor").mkdir()
        (fake_home / ".memwright").mkdir()

        assert _paths.resolve_store_path() == str(fake_home / ".attestor")

    def test_legacy_dir_fallback_with_warning(self, fake_home):
        (fake_home / ".memwright").mkdir()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _paths.resolve_store_path()

        assert result == str(fake_home / ".memwright")
        msgs = [str(w.message) for w in caught]
        assert any("attestor migrate" in m for m in msgs)

    def test_legacy_dir_warning_is_once_per_process(self, fake_home):
        (fake_home / ".memwright").mkdir()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _paths.resolve_store_path()
            _paths.resolve_store_path()
            _paths.resolve_store_path()

        migration_warnings = [w for w in caught if "attestor migrate" in str(w.message)]
        assert len(migration_warnings) == 1

    def test_default_new_is_attestor_dir(self, fake_home):
        assert _paths.resolve_store_path() == str(fake_home / ".attestor")


# resolve_data_dir has the same 5-branch logic against *_DATA_DIR -------------


class TestResolveDataDir:
    def test_attestor_data_dir_env(self, fake_home, monkeypatch):
        monkeypatch.setenv("ATTESTOR_DATA_DIR", "/srv/attestor")
        monkeypatch.setenv("MEMWRIGHT_DATA_DIR", "/srv/legacy")

        assert _paths.resolve_data_dir() == "/srv/attestor"

    def test_legacy_data_dir_env_warns(self, fake_home, monkeypatch):
        monkeypatch.setenv("MEMWRIGHT_DATA_DIR", "/srv/legacy")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _paths.resolve_data_dir()

        assert result == "/srv/legacy"
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps
        assert "ATTESTOR_DATA_DIR" in str(deps[0].message)

    def test_default_new_attestor_dir(self, fake_home):
        assert _paths.resolve_data_dir() == str(fake_home / ".attestor")


# Cache dir: ~/.cache/attestor preferred, ~/.cache/memwright fallback ---------


class TestResolveCacheDir:
    def test_default_is_attestor_cache(self, fake_home):
        assert _paths.resolve_cache_dir() == fake_home / ".cache" / "attestor"

    def test_legacy_cache_used_when_only_legacy_present(self, fake_home):
        (fake_home / ".cache" / "memwright").mkdir(parents=True)

        assert _paths.resolve_cache_dir() == fake_home / ".cache" / "memwright"

    def test_new_cache_preferred_when_both_exist(self, fake_home):
        (fake_home / ".cache" / "attestor").mkdir(parents=True)
        (fake_home / ".cache" / "memwright").mkdir(parents=True)

        assert _paths.resolve_cache_dir() == fake_home / ".cache" / "attestor"
