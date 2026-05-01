"""DynamoDB document-role mixin for AWSBackend (split from aws_backend.py).

This module is private — consumers should import ``AWSBackend`` from
``attestor.store.aws_backend``. The mixin is stateless: it operates on
``self._table`` / ``self._dynamodb`` configured by ``AWSBackend.__init__``.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from attestor.models import Memory

logger = logging.getLogger("attestor")


# DynamoDB stores Decimal, not float — helpers to convert
_FLOAT_FIELDS = {"confidence"}


def _float_to_decimal(value: Any) -> Any:
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _float_to_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_float_to_decimal(v) for v in value]
    return value


def _decimal_to_float(value: Any) -> Any:
    """Convert Decimal values back to float from DynamoDB."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_to_float(v) for v in value]
    return value


class _AWSDocumentMixin:
    """DynamoDB document role for AWSBackend."""

    # ── DocumentStore — DynamoDB ──

    def _memory_to_item(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory to DynamoDB item dict."""
        item: dict[str, Any] = {
            "id": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "namespace": memory.namespace,
            "created_at": memory.created_at,
            "valid_from": memory.valid_from,
            "confidence": _float_to_decimal(memory.confidence),
            "status": memory.status,
            "metadata": _float_to_decimal(memory.metadata) if memory.metadata else {},
        }
        # Optional fields — DynamoDB can't store None for GSI key attributes
        if memory.entity:
            item["entity"] = memory.entity
        if memory.event_date:
            item["event_date"] = memory.event_date
        if memory.valid_until:
            item["valid_until"] = memory.valid_until
        if memory.superseded_by:
            item["superseded_by"] = memory.superseded_by
        return item

    def _item_to_memory(self, item: dict[str, Any]) -> Memory:
        """Convert DynamoDB item to Memory."""
        item = _decimal_to_float(item)
        metadata = item.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Memory(
            id=item["id"],
            content=item["content"],
            tags=item.get("tags", []),
            category=item.get("category", "general"),
            namespace=item.get("namespace", "default"),
            entity=item.get("entity"),
            created_at=item["created_at"],
            event_date=item.get("event_date"),
            valid_from=item.get("valid_from", item["created_at"]),
            valid_until=item.get("valid_until"),
            superseded_by=item.get("superseded_by"),
            confidence=item.get("confidence", 1.0),
            status=item.get("status", "active"),
            metadata=metadata,
        )

    def insert(self, memory: Memory) -> Memory:
        self._table.put_item(Item=self._memory_to_item(memory))
        return memory

    def get(self, memory_id: str) -> Memory | None:
        resp = self._table.get_item(Key={"id": memory_id})
        item = resp.get("Item")
        if item is None:
            return None
        return self._item_to_memory(item)

    def update(self, memory: Memory) -> Memory:
        self._table.put_item(Item=self._memory_to_item(memory))
        return memory

    def delete(self, memory_id: str) -> bool:
        resp = self._table.delete_item(
            Key={"id": memory_id},
            ReturnValues="ALL_OLD",
        )
        return "Attributes" in resp

    def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        from boto3.dynamodb.conditions import Attr

        # Use GSI queries when a single partition key filter is available and no namespace filter
        if status and not category and not entity and not namespace:
            return self._query_gsi(
                "status-created_at-index", "status", status,
                after=after, before=before, limit=limit,
            )
        if category and not status and not entity and not namespace:
            return self._query_gsi(
                "category-created_at-index", "category", category,
                after=after, before=before, limit=limit,
            )
        if entity and not status and not category and not namespace:
            return self._query_gsi(
                "entity-created_at-index", "entity", entity,
                after=after, before=before, limit=limit,
            )

        # Fallback: scan with filter
        filter_expr = None
        conditions = []

        if status:
            conditions.append(Attr("status").eq(status))
        if category:
            conditions.append(Attr("category").eq(category))
        if entity:
            conditions.append(Attr("entity").eq(entity))
        if namespace:
            conditions.append(Attr("namespace").eq(namespace))
        if after:
            conditions.append(Attr("created_at").gte(after))
        if before:
            conditions.append(Attr("created_at").lte(before))

        if conditions:
            combined = conditions[0]
            for c in conditions[1:]:
                combined = combined & c
            filter_expr = combined

        scan_kwargs: dict[str, Any] = {"Limit": limit}
        if filter_expr is not None:
            scan_kwargs["FilterExpression"] = filter_expr

        items = self._scan_all(scan_kwargs, limit)
        memories = [self._item_to_memory(item) for item in items]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def _query_gsi(
        self,
        index_name: str,
        pk_name: str,
        pk_value: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Query a GSI by partition key, optionally filtering by sort key range."""
        from boto3.dynamodb.conditions import Key

        key_expr = Key(pk_name).eq(pk_value)
        if after and before:
            key_expr = key_expr & Key("created_at").between(after, before)
        elif after:
            key_expr = key_expr & Key("created_at").gte(after)
        elif before:
            key_expr = key_expr & Key("created_at").lte(before)

        resp = self._table.query(
            IndexName=index_name,
            KeyConditionExpression=key_expr,
            Limit=limit,
            ScanIndexForward=False,
        )
        return [self._item_to_memory(item) for item in resp.get("Items", [])]

    def _scan_all(self, scan_kwargs: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        """Scan with pagination, stopping at limit."""
        items: list[dict[str, Any]] = []
        while True:
            resp = self._table.scan(**scan_kwargs)
            items.extend(resp.get("Items", []))
            if len(items) >= limit:
                break
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
        return items[:limit]

    def tag_search(
        self,
        tags: list[str],
        category: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Scan for memories whose tags overlap with the given list.

        DynamoDB doesn't have a native array-overlap operator, so we scan
        with a filter that checks membership of each tag.
        """
        from boto3.dynamodb.conditions import Attr

        # Build: status = active AND valid_until not exists AND (contains(tags, t1) OR ...)
        conditions = [
            Attr("status").eq("active"),
            Attr("valid_until").not_exists(),
        ]

        tag_conditions = [Attr("tags").contains(t) for t in tags]
        if tag_conditions:
            tag_filter = tag_conditions[0]
            for tc in tag_conditions[1:]:
                tag_filter = tag_filter | tc
            conditions.append(tag_filter)

        if category:
            conditions.append(Attr("category").eq(category))
        if namespace:
            conditions.append(Attr("namespace").eq(namespace))

        combined = conditions[0]
        for c in conditions[1:]:
            combined = combined & c

        items = self._scan_all({"FilterExpression": combined, "Limit": limit}, limit)
        memories = [self._item_to_memory(item) for item in items]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def execute(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Raw SQL not supported on DynamoDB")

    def archive_before(self, date: str) -> int:
        """Archive active memories created before the given date."""
        from boto3.dynamodb.conditions import Attr

        filter_expr = (
            Attr("status").eq("active") & Attr("created_at").lt(date)
        )
        items = self._scan_all({"FilterExpression": filter_expr, "Limit": 10000}, 10000)

        count = 0
        with self._table.batch_writer() as batch:
            for item in items:
                item["status"] = "archived"
                batch.put_item(Item=item)
                count += 1
        return count

    def compact(self) -> int:
        """Delete all archived memories."""
        from boto3.dynamodb.conditions import Attr

        filter_expr = Attr("status").eq("archived")
        items = self._scan_all({"FilterExpression": filter_expr, "Limit": 10000}, 10000)

        count = 0
        with self._table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"id": item["id"]})
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        items = self._scan_all({"Limit": 100000}, 100000)

        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for item in items:
            s = item.get("status", "active")
            by_status[s] = by_status.get(s, 0) + 1
            c = item.get("category", "general")
            by_category[c] = by_category.get(c, 0) + 1

        return {
            "total_memories": len(items),
            "by_status": by_status,
            "by_category": by_category,
        }
