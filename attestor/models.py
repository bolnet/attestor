"""Core data models."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""

    # Classification
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    entity: Optional[str] = None
    namespace: str = "default"

    # Temporal
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_date: Optional[str] = None
    valid_from: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    valid_until: Optional[str] = None
    superseded_by: Optional[str] = None

    # Vector (optional)
    embedding: Optional[List[float]] = None

    # Provenance
    confidence: float = 1.0

    # Status
    status: str = "active"

    # Access tracking
    access_count: int = 0
    last_accessed: Optional[str] = None

    # Content dedup
    content_hash: Optional[str] = None

    # Extensible
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tags_json(self) -> str:
        return json.dumps(self.tags)

    def metadata_json(self) -> str:
        return json.dumps(self.metadata)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Memory:
        """Create a Memory from a SQLite row dict."""
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
        metadata = (
            json.loads(row["metadata"])
            if isinstance(row["metadata"], str)
            else row["metadata"]
        )
        return cls(
            id=row["id"],
            content=row["content"],
            tags=tags,
            category=row["category"],
            entity=row.get("entity"),
            namespace=row.get("namespace", "default"),
            created_at=row["created_at"],
            event_date=row.get("event_date"),
            valid_from=row["valid_from"],
            valid_until=row.get("valid_until"),
            superseded_by=row.get("superseded_by"),
            confidence=row.get("confidence", 1.0),
            status=row.get("status", "active"),
            access_count=row.get("access_count", 0),
            last_accessed=row.get("last_accessed"),
            content_hash=row.get("content_hash"),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for JSON export."""
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "category": self.category,
            "entity": self.entity,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "event_date": self.event_date,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded_by": self.superseded_by,
            "confidence": self.confidence,
            "status": self.status,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    match_source: str  # "tag", "fts", "vector"

    @property
    def content(self) -> str:
        return self.memory.content


