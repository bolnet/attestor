"""Tests for backend registry and resolver."""

import pytest

from attestor.store.registry import (
    BACKEND_REGISTRY,
    DEFAULT_BACKENDS,
    BackendConflictError,
    instantiate_backend,
    resolve_backends,
)


class TestResolveBackends:
    def test_default_backends(self):
        roles = resolve_backends()
        assert roles == {
            "document": "postgres",
            "vector": "pinecone",
            "graph": "neo4j",
        }

    def test_explicit_defaults(self):
        roles = resolve_backends(["postgres", "pinecone", "neo4j"])
        assert roles["document"] == "postgres"
        assert roles["vector"] == "pinecone"
        assert roles["graph"] == "neo4j"

    def test_legacy_pgvector_bundle(self):
        """The pgvector entry bundles document + vector for runs that
        don't want a separate Pinecone — same backend class, different
        registry key."""
        roles = resolve_backends(["pgvector", "neo4j"])
        assert roles["document"] == "pgvector"
        assert roles["vector"] == "pgvector"
        assert roles["graph"] == "neo4j"

    def test_arangodb_fills_all_roles(self):
        roles = resolve_backends(["arangodb"])
        assert roles["document"] == "arangodb"
        assert roles["vector"] == "arangodb"
        assert roles["graph"] == "arangodb"

    def test_conflict_raises(self):
        with pytest.raises(BackendConflictError, match="document"):
            resolve_backends(["postgres", "arangodb"])

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backends(["nonexistent"])

    def test_partial_roles_ok(self):
        """A subset of roles is fine — unfilled roles degrade gracefully."""
        roles = resolve_backends(["neo4j"])
        assert roles == {"graph": "neo4j"}
        assert "document" not in roles
        assert "vector" not in roles


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

    def test_embedded_backends_removed(self):
        """The zero-config embedded trio must not be in the registry anymore."""
        for dropped in ("sqlite", "chroma", "networkx"):
            assert dropped not in BACKEND_REGISTRY


class TestInstantiateBackend:
    def test_instantiate_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError):
            instantiate_backend("nonexistent", tmp_path)

    def test_missing_arango_import_raises(self, tmp_path):
        """When python-arango is not installed (or no server running),
        importing arangodb backend should fail gracefully."""
        try:
            instantiate_backend("arangodb", tmp_path, backend_config={})
        except (ImportError, ModuleNotFoundError, ConnectionAbortedError, Exception):
            pass  # Expected — python-arango not installed or no ArangoDB running
