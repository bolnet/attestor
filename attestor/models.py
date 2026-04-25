"""Core data models."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── v4 Identity primitives ────────────────────────────────────────────────


class MemoryScope(str, Enum):
    """Visibility scope for a memory.

    USER     — visible across all sessions and projects of this user (default)
    PROJECT  — visible across all sessions in one project
    SESSION  — visible only within a single conversation session
    """
    USER = "user"
    PROJECT = "project"
    SESSION = "session"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class User:
    id: str
    external_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    status: str = "active"
    created_at: datetime = field(default_factory=_now_utc)
    deleted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> User:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return cls(
            id=str(row["id"]),
            external_id=row["external_id"],
            email=row.get("email"),
            display_name=row.get("display_name"),
            status=row.get("status", "active"),
            created_at=row["created_at"],
            deleted_at=row.get("deleted_at"),
            metadata=meta,
        )


@dataclass(frozen=True)
class Project:
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: str = "active"
    created_at: datetime = field(default_factory=_now_utc)
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_inbox(self) -> bool:
        return bool(self.metadata.get("is_inbox"))

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Project:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=row["name"],
            description=row.get("description"),
            status=row.get("status", "active"),
            created_at=row["created_at"],
            archived_at=row.get("archived_at"),
            metadata=meta,
        )


@dataclass(frozen=True)
class Session:
    id: str
    user_id: str
    project_id: Optional[str] = None
    title: Optional[str] = None
    status: str = "active"
    created_at: datetime = field(default_factory=_now_utc)
    last_active_at: datetime = field(default_factory=_now_utc)
    ended_at: Optional[datetime] = None
    message_count: int = 0
    consolidation_state: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Session:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            project_id=str(row["project_id"]) if row.get("project_id") else None,
            title=row.get("title"),
            status=row.get("status", "active"),
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
            ended_at=row.get("ended_at"),
            message_count=row.get("message_count", 0),
            consolidation_state=row.get("consolidation_state", "pending"),
            metadata=meta,
        )


# ── Existing Memory / RetrievalResult (Phase 0 keeps the v3 shape) ───────
# Phase 1 will extend Memory with v4 columns (user_id, scope, t_created, etc.)
# and rewrite namespace usage. For Phase 0 we ship the schema + identity
# primitives without touching the existing Memory dataclass — keeps every
# caller working until Phase 1 lands.


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
        """Create a Memory from a document-store row dict."""
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


