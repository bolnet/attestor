"""Tests for AWSBackend — DynamoDB mocked via moto, OpenSearch/Neptune skipped."""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

try:
    import boto3
    from moto import mock_aws

    HAS_MOTO = True
except ImportError:
    HAS_MOTO = False
    # Provide a no-op mock_aws so pytest can collect decorators before skipif kicks in
    from contextlib import nullcontext as mock_aws

from attestor.models import Memory

pytestmark = pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")


def _make_memory(**kwargs) -> Memory:
    defaults = {
        "id": "mem001",
        "content": "Alice likes pizza",
        "tags": ["food", "preference"],
        "category": "preference",
        "entity": "Alice",
        "created_at": "2026-01-15T10:00:00+00:00",
        "valid_from": "2026-01-15T10:00:00+00:00",
        "confidence": 0.95,
        "status": "active",
        "metadata": {"source": "chat"},
    }
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.fixture
def aws_backend():
    """Create AWSBackend with moto-mocked DynamoDB, no OpenSearch/Neptune."""
    with mock_aws():
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"

        from attestor.store.aws_backend import AWSBackend

        config = {
            "region": "us-east-1",
            "dynamodb": {"table_prefix": "test_mw"},
            "opensearch": {"endpoint": ""},
            "neptune": {"endpoint": ""},
        }
        backend = AWSBackend(config)
        yield backend
        backend.close()


class TestDynamoDBDocumentStore:
    def test_insert_and_get(self, aws_backend):
        mem = _make_memory()
        aws_backend.insert(mem)
        result = aws_backend.get("mem001")
        assert result is not None
        assert result.id == "mem001"
        assert result.content == "Alice likes pizza"
        assert result.tags == ["food", "preference"]
        assert result.category == "preference"
        assert result.entity == "Alice"
        assert abs(result.confidence - 0.95) < 0.01

    def test_get_nonexistent(self, aws_backend):
        assert aws_backend.get("no_such_id") is None

    def test_update(self, aws_backend):
        mem = _make_memory()
        aws_backend.insert(mem)

        updated = Memory(
            id="mem001",
            content="Alice now likes sushi",
            tags=["food", "preference"],
            category="preference",
            entity="Alice",
            created_at=mem.created_at,
            valid_from=mem.valid_from,
            confidence=0.99,
            status="active",
            metadata={"source": "chat", "updated": True},
        )
        aws_backend.update(updated)

        result = aws_backend.get("mem001")
        assert result.content == "Alice now likes sushi"
        assert abs(result.confidence - 0.99) < 0.01

    def test_delete(self, aws_backend):
        mem = _make_memory()
        aws_backend.insert(mem)
        assert aws_backend.delete("mem001") is True
        assert aws_backend.get("mem001") is None

    def test_delete_nonexistent(self, aws_backend):
        assert aws_backend.delete("no_such_id") is False

    def test_list_memories_no_filter(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", created_at="2026-01-01T00:00:00+00:00"))
        aws_backend.insert(_make_memory(id="m2", created_at="2026-01-02T00:00:00+00:00"))
        aws_backend.insert(_make_memory(id="m3", created_at="2026-01-03T00:00:00+00:00"))

        results = aws_backend.list_memories()
        assert len(results) == 3
        # Should be sorted descending by created_at
        assert results[0].id == "m3"

    def test_list_memories_by_status(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", status="active"))
        aws_backend.insert(_make_memory(id="m2", status="archived"))

        results = aws_backend.list_memories(status="active")
        assert len(results) == 1
        assert results[0].id == "m1"

    def test_list_memories_by_category(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", category="preference"))
        aws_backend.insert(_make_memory(id="m2", category="fact"))

        results = aws_backend.list_memories(category="fact")
        assert len(results) == 1
        assert results[0].id == "m2"

    def test_list_memories_by_entity(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", entity="Alice"))
        aws_backend.insert(_make_memory(id="m2", entity="Bob"))

        results = aws_backend.list_memories(entity="Bob")
        assert len(results) == 1
        assert results[0].id == "m2"

    def test_list_memories_with_limit(self, aws_backend):
        for i in range(5):
            aws_backend.insert(_make_memory(id=f"m{i}", created_at=f"2026-01-0{i+1}T00:00:00+00:00"))

        results = aws_backend.list_memories(limit=2)
        assert len(results) == 2

    def test_tag_search(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", tags=["food", "preference"]))
        aws_backend.insert(_make_memory(id="m2", tags=["work", "meeting"]))
        aws_backend.insert(_make_memory(id="m3", tags=["food", "cooking"]))

        results = aws_backend.tag_search(tags=["food"])
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"m1", "m3"}

    def test_tag_search_with_category(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", tags=["food"], category="preference"))
        aws_backend.insert(_make_memory(id="m2", tags=["food"], category="fact"))

        results = aws_backend.tag_search(tags=["food"], category="fact")
        assert len(results) == 1
        assert results[0].id == "m2"

    def test_execute_raises(self, aws_backend):
        with pytest.raises(NotImplementedError, match="Raw SQL"):
            aws_backend.execute("SELECT 1")

    def test_archive_before(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", created_at="2025-01-01T00:00:00+00:00"))
        aws_backend.insert(_make_memory(id="m2", created_at="2026-06-01T00:00:00+00:00"))

        count = aws_backend.archive_before("2026-01-01T00:00:00+00:00")
        assert count == 1

        archived = aws_backend.get("m1")
        assert archived.status == "archived"
        still_active = aws_backend.get("m2")
        assert still_active.status == "active"

    def test_compact(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", status="archived"))
        aws_backend.insert(_make_memory(id="m2", status="active"))

        count = aws_backend.compact()
        assert count == 1
        assert aws_backend.get("m1") is None
        assert aws_backend.get("m2") is not None

    def test_stats(self, aws_backend):
        aws_backend.insert(_make_memory(id="m1", status="active", category="preference"))
        aws_backend.insert(_make_memory(id="m2", status="active", category="fact"))
        aws_backend.insert(_make_memory(id="m3", status="archived", category="preference"))

        stats = aws_backend.stats()
        assert stats["total_memories"] == 3
        assert stats["by_status"]["active"] == 2
        assert stats["by_status"]["archived"] == 1
        assert stats["by_category"]["preference"] == 2
        assert stats["by_category"]["fact"] == 1


class TestTableAutoCreation:
    def test_table_created_with_gsis(self, aws_backend):
        """Verify table and GSIs were auto-created."""
        table = aws_backend._table
        assert table is not None
        assert table.table_status == "ACTIVE"

        gsi_names = {gsi["IndexName"] for gsi in table.global_secondary_indexes}
        assert "status-created_at-index" in gsi_names
        assert "category-created_at-index" in gsi_names
        assert "entity-created_at-index" in gsi_names


class TestConfigParsing:
    @mock_aws
    def test_default_config(self):
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"

        from attestor.store.aws_backend import AWSBackend

        backend = AWSBackend({"region": "us-east-1"})
        assert backend._table_name == "attestor_memories"
        assert backend._region == "us-east-1"
        assert backend._opensearch_index == "memories"
        backend.close()

    @mock_aws
    def test_custom_table_prefix(self):
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"

        from attestor.store.aws_backend import AWSBackend

        backend = AWSBackend({
            "region": "us-west-2",
            "dynamodb": {"table_prefix": "myapp"},
        })
        assert backend._table_name == "myapp_memories"
        assert backend._region == "us-west-2"
        backend.close()


class TestVectorStoreGracefulDegradation:
    def test_search_returns_empty_without_opensearch(self, aws_backend):
        results = aws_backend.search("test query")
        assert results == []

    def test_count_returns_zero_without_opensearch(self, aws_backend):
        assert aws_backend.count() == 0


class TestGraphStoreGracefulDegradation:
    def test_get_related_returns_empty_without_neptune(self, aws_backend):
        assert aws_backend.get_related("Alice") == []

    def test_get_subgraph_returns_empty_without_neptune(self, aws_backend):
        result = aws_backend.get_subgraph("Alice")
        assert result == {"entity": "Alice", "nodes": [], "edges": []}

    def test_get_entities_returns_empty_without_neptune(self, aws_backend):
        assert aws_backend.get_entities() == []

    def test_get_edges_returns_empty_without_neptune(self, aws_backend):
        assert aws_backend.get_edges("Alice") == []

    def test_graph_stats_returns_zeros_without_neptune(self, aws_backend):
        stats = aws_backend.graph_stats()
        assert stats == {"nodes": 0, "edges": 0, "types": {}}

    def test_save_is_noop(self, aws_backend):
        aws_backend.save()  # should not raise


class TestDecimalConversion:
    def test_float_to_decimal(self):
        from attestor.store.aws_backend import _float_to_decimal

        assert _float_to_decimal(0.95) == Decimal("0.95")
        assert _float_to_decimal({"a": 1.5}) == {"a": Decimal("1.5")}
        assert _float_to_decimal([1.1, 2.2]) == [Decimal("1.1"), Decimal("2.2")]
        assert _float_to_decimal("hello") == "hello"

    def test_decimal_to_float(self):
        from attestor.store.aws_backend import _decimal_to_float

        assert _decimal_to_float(Decimal("0.95")) == 0.95
        assert _decimal_to_float({"a": Decimal("1.5")}) == {"a": 1.5}
        assert _decimal_to_float([Decimal("1.1")]) == [1.1]
        assert _decimal_to_float("hello") == "hello"
