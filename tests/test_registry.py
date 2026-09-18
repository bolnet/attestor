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

    def test_conflict_raises(self):
        # Repeating the same backend twice claims its role twice → conflict.
        with pytest.raises(BackendConflictError, match="document"):
            resolve_backends(["postgres", "postgres"])

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
    def test_only_supported_backends_present(self):
        """Registry exposes postgres + neo4j, and BOTH vector options.

        The 2026-05-02 single-stack policy removed the alternate backends
        (arango, aws, azure, gcp) and that still holds — they must stay absent.

        `pgvector` was removed with them, and is deliberately reinstated: not
        as the old single-DB bundle, but as one of two interchangeable VECTOR
        backends behind the same role. The single-stack argument was about not
        maintaining five divergent stacks; offering Postgres or Pinecone for
        one role is a config line over an implementation that already exists
        (`_PostgresVectorMixin`), not a second stack to maintain.

        What must NOT come back is a pgvector entry that also claims the
        document or graph role — that is the bundle the policy removed.
        """
        assert set(BACKEND_REGISTRY) == {"postgres", "pgvector", "pinecone", "neo4j"}
        assert BACKEND_REGISTRY["pgvector"]["roles"] == {"vector"}

    def test_the_two_vector_backends_are_mutually_exclusive(self):
        """Either is selectable; asking for both is an error rather than a
        silent pick, because which store answered a recall must never be
        ambiguous."""
        from attestor.store.registry import BackendConflictError

        assert resolve_backends(["postgres", "pgvector", "neo4j"])["vector"] == "pgvector"
        assert resolve_backends(["postgres", "pinecone", "neo4j"])["vector"] == "pinecone"
        with pytest.raises(BackendConflictError):
            resolve_backends(["postgres", "pgvector", "pinecone", "neo4j"])

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
    def test_instantiate_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError):
            instantiate_backend("nonexistent", tmp_path)
