"""Tests for GCP AlloyDB backend.

Unit tests mock GCP dependencies. No real AlloyDB or GCP credentials needed.
"""

from unittest.mock import MagicMock, patch
import pytest

from attestor.store.registry import BACKEND_REGISTRY, resolve_backends

try:
    import psycopg2  # noqa: F401
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


# ── Registry Tests ──


class TestGCPRegistry:
    def test_gcp_in_registry(self):
        assert "gcp" in BACKEND_REGISTRY

    def test_gcp_registry_entry(self):
        entry = BACKEND_REGISTRY["gcp"]
        assert entry["module"] == "attestor.store.gcp_backend"
        assert entry["class"] == "GCPBackend"
        assert entry["roles"] == {"document", "vector", "graph"}
        assert entry["init_style"] == "config"

    def test_gcp_fills_all_roles(self):
        roles = resolve_backends(["gcp"])
        assert roles == {
            "document": "gcp",
            "vector": "gcp",
            "graph": "gcp",
        }


# ── Connection Defaults ──


class TestGCPConnectionDefaults:
    def test_gcp_engine_defaults(self):
        from attestor.store.connection import ENGINE_DEFAULTS

        defaults = ENGINE_DEFAULTS["gcp"]
        assert defaults["url"] == "postgresql://localhost:5432"
        assert defaults["port"] == 5432
        assert defaults["database"] == "attestor"
        assert defaults["region"] == "us-central1"
        assert defaults["project_id"] == ""
        assert defaults["cluster"] == ""
        assert defaults["instance"] == ""
        assert defaults["tls"]["verify"] is True


# ── GCPBackend Unit Tests (mocked) ──


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestGCPBackendConnectorPath:
    """Test the AlloyDB Connector connection path."""

    @patch("attestor.store.gcp_backend._has_alloydb_connector", return_value=True)
    @patch("attestor.store.gcp_backend.Connector", create=True)
    def test_connector_used_when_gcp_fields_set(self, mock_connector_cls, mock_has):
        """When project_id + cluster + instance are set, use AlloyDB Connector."""
        from attestor.store.gcp_backend import GCPBackend

        mock_connector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.autocommit = True

        # Patch the Connector import inside _init_via_connector
        with patch(
            "attestor.store.gcp_backend.Connector", return_value=mock_connector
        ):
            mock_connector.connect.return_value = mock_conn

            # Mock embedding provider
            mock_embedder = MagicMock()
            mock_embedder.dimension = 768
            mock_embedder.provider_name = "vertex_ai"

            with patch(
                "attestor.store.gcp_backend.get_embedding_provider",
                create=True,
            ) as mock_get_embed:
                # Patch the import inside _ensure_embedding_fn
                with patch(
                    "attestor.store.embeddings.get_embedding_provider",
                    return_value=mock_embedder,
                ):
                    mock_get_embed.return_value = mock_embedder

                    # Mock _init_schema and _init_age to avoid real DB calls
                    with patch.object(GCPBackend, "_init_schema"), \
                         patch.object(GCPBackend, "_init_age"):
                        backend = GCPBackend({
                            "project_id": "my-project",
                            "region": "us-central1",
                            "cluster": "my-cluster",
                            "instance": "my-instance",
                            "database": "attestor",
                        })

                        # Verify connector was used
                        mock_connector.connect.assert_called_once()
                        call_args = mock_connector.connect.call_args
                        instance_uri = call_args[0][0]
                        assert "my-project" in instance_uri
                        assert "my-cluster" in instance_uri
                        assert "my-instance" in instance_uri

    def test_should_use_connector_all_fields(self):
        """Connector used when all GCP fields are present."""
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        fields = {
            "project_id": "proj",
            "region": "us-central1",
            "cluster": "cl",
            "instance": "inst",
            "database": "db",
        }
        with patch("attestor.store.gcp_backend._has_alloydb_connector", return_value=True):
            assert backend._should_use_connector(fields) is True

    def test_should_not_use_connector_missing_fields(self):
        """Falls back to psycopg2 when GCP fields are missing."""
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        fields = {
            "project_id": "proj",
            "region": "us-central1",
            "cluster": "",
            "instance": "inst",
            "database": "db",
        }
        assert backend._should_use_connector(fields) is False

    def test_should_not_use_connector_no_package(self):
        """Falls back to psycopg2 when AlloyDB Connector package is missing."""
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        fields = {
            "project_id": "proj",
            "region": "us-central1",
            "cluster": "cl",
            "instance": "inst",
            "database": "db",
        }
        with patch("attestor.store.gcp_backend._has_alloydb_connector", return_value=False):
            assert backend._should_use_connector(fields) is False


class TestGCPBackendPsycopgFallback:
    """Test the direct psycopg2 fallback path."""

    @pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
    def test_fallback_calls_parent_init(self):
        """Without GCP fields, GCPBackend delegates to PostgresBackend.__init__."""
        from attestor.store.gcp_backend import GCPBackend

        with patch.object(
            GCPBackend.__bases__[0], "__init__", return_value=None
        ) as mock_parent:
            config = {
                "url": "postgresql://localhost:5432",
                "database": "testdb",
            }
            try:
                backend = GCPBackend(config)
            except Exception:
                pass
            # Parent __init__ should have been called
            mock_parent.assert_called_once_with(config)


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestScaNNIndex:
    """Test ScaNN index creation and HNSW fallback."""

    def test_try_scann_success(self):
        """ScaNN index created when extension is available."""
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        backend._conn = MagicMock()
        backend._conn.cursor.return_value.__enter__ = MagicMock()
        backend._conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        executed_sqls = []

        def capture_execute(sql, params=None):
            executed_sqls.append(sql.strip())
            return []

        backend._execute = capture_execute
        backend._try_scann_index()

        sql_text = " ".join(executed_sqls)
        assert "alloydb_scann" in sql_text
        assert "DROP INDEX IF EXISTS idx_memories_embedding_hnsw" in sql_text
        assert "idx_memories_embedding_scann" in sql_text

    def test_try_scann_fallback_on_error(self):
        """HNSW kept when ScaNN extension is unavailable."""
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        backend._conn = MagicMock()

        call_count = 0

        def fail_on_scann(sql, params=None):
            nonlocal call_count
            call_count += 1
            if "alloydb_scann" in sql:
                raise Exception("extension alloydb_scann does not exist")
            return []

        backend._execute = fail_on_scann
        # Should not raise
        backend._try_scann_index()
        assert call_count == 1  # Only the CREATE EXTENSION call, then caught


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestVertexAIPreference:
    """Test that GCPBackend prefers Vertex AI embeddings."""

    def test_ensure_embedding_fn_prefers_vertex(self):
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        backend._embedder = None
        backend._embedding_fn = None

        mock_provider = MagicMock()
        mock_provider.provider_name = "vertex_ai"
        mock_provider.dimension = 768

        with patch(
            "attestor.store.embeddings.get_embedding_provider",
            return_value=mock_provider,
        ) as mock_get:
            backend._ensure_embedding_fn()
            mock_get.assert_called_once_with("vertex_ai")
            assert backend._embedder is mock_provider
            assert backend._embedding_fn is mock_provider

    def test_ensure_embedding_fn_noop_if_already_set(self):
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        existing = MagicMock()
        backend._embedder = existing
        backend._embedding_fn = existing

        backend._ensure_embedding_fn()
        assert backend._embedder is existing  # unchanged


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestInheritance:
    """Verify GCPBackend inherits PostgresBackend methods."""

    def test_is_subclass(self):
        from attestor.store.gcp_backend import GCPBackend
        from attestor.store.postgres_backend import PostgresBackend

        assert issubclass(GCPBackend, PostgresBackend)

    def test_inherited_methods_exist(self):
        from attestor.store.gcp_backend import GCPBackend

        inherited = [
            "insert", "get", "update", "delete", "list_memories",
            "tag_search", "execute", "archive_before", "compact", "stats",
            "add", "search", "count",
            "add_entity", "add_relation", "get_related", "get_subgraph",
            "get_entities", "get_edges", "graph_stats", "save",
        ]
        for method in inherited:
            assert hasattr(GCPBackend, method), f"Missing inherited method: {method}"

    def test_roles(self):
        from attestor.store.gcp_backend import GCPBackend

        assert GCPBackend.ROLES == {"document", "vector", "graph"}


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestCloseWithConnector:
    """Test close() cleans up AlloyDB Connector."""

    def test_close_with_connector(self):
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        backend._conn = MagicMock()
        backend._conn.closed = False
        backend._connector = MagicMock()

        backend.close()

        backend._conn.close.assert_called_once()
        backend._connector.close.assert_called_once()

    def test_close_without_connector(self):
        from attestor.store.gcp_backend import GCPBackend

        backend = GCPBackend.__new__(GCPBackend)
        backend._conn = MagicMock()
        backend._conn.closed = False

        # No _connector attribute
        backend.close()
        backend._conn.close.assert_called_once()


