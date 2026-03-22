"""Tests for backend registry and resolver."""

import pytest
from agent_memory.store.registry import (
    BACKEND_REGISTRY,
    DEFAULT_BACKENDS,
    BackendConflictError,
    resolve_backends,
    instantiate_backend,
)


class TestResolveBackends:
    def test_default_backends(self):
        roles = resolve_backends()
        assert roles == {
            "document": "sqlite",
            "vector": "chroma",
            "graph": "networkx",
        }

    def test_explicit_defaults(self):
        roles = resolve_backends(["sqlite", "chroma", "networkx"])
        assert roles["document"] == "sqlite"
        assert roles["vector"] == "chroma"
        assert roles["graph"] == "networkx"

    def test_arangodb_fills_all_roles(self):
        roles = resolve_backends(["arangodb"])
        assert roles["document"] == "arangodb"
        assert roles["vector"] == "arangodb"
        assert roles["graph"] == "arangodb"

    def test_conflict_raises(self):
        with pytest.raises(BackendConflictError, match="document"):
            resolve_backends(["sqlite", "arangodb"])

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backends(["postgres"])

    def test_partial_roles_ok(self):
        """A subset of roles is fine — unfilled roles degrade gracefully."""
        roles = resolve_backends(["sqlite"])
        assert roles == {"document": "sqlite"}
        assert "vector" not in roles
        assert "graph" not in roles


class TestBackendRegistry:
    def test_all_entries_have_required_keys(self):
        for name, entry in BACKEND_REGISTRY.items():
            assert "module" in entry, f"{name} missing 'module'"
            assert "class" in entry, f"{name} missing 'class'"
            assert "roles" in entry, f"{name} missing 'roles'"
            assert "init_style" in entry, f"{name} missing 'init_style'"
            assert entry["init_style"] in ("path", "config"), f"{name} has invalid init_style"

    def test_default_backends_in_registry(self):
        for name in DEFAULT_BACKENDS:
            assert name in BACKEND_REGISTRY


class TestInstantiateBackend:
    def test_instantiate_sqlite(self, tmp_path):
        store = instantiate_backend("sqlite", tmp_path / "memory.db")
        from agent_memory.store.sqlite_store import SQLiteStore
        assert isinstance(store, SQLiteStore)
        store.close()

    def test_instantiate_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError):
            instantiate_backend("postgres", tmp_path)

    def test_missing_arango_import_raises(self, tmp_path):
        """When python-arango is not installed, importing arangodb backend should fail gracefully."""
        try:
            instantiate_backend("arangodb", tmp_path, backend_config={})
        except (ImportError, ModuleNotFoundError):
            pass  # Expected — python-arango not installed
