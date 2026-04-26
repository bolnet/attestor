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
    namespace: str = "default"   # legacy v3; derived from user/project/session in v4

    # v4 tenancy (Optional → backward-compat with v3 callers)
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: str = "user"          # one of: user | project | session

    # Temporal — event time
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_date: Optional[str] = None
    valid_from: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    valid_until: Optional[str] = None
    superseded_by: Optional[str] = None

    # v4 bi-temporal — transaction time (when system knew it)
    t_created: Optional[str] = None
    t_expired: Optional[str] = None

    # v4 provenance — links back to source episode + extraction context
    source_episode_id: Optional[str] = None
    source_span: Optional[List[int]] = None   # [start_char, end_char]
    extraction_model: Optional[str] = None
    agent_id: Optional[str] = None
    parent_agent_id: Optional[str] = None
    visibility: str = "team"
    signature: Optional[str] = None           # opt-in Ed25519 sig

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
        """Create a Memory from a document-store row dict.

        Backward-compatible with both v3 (TEXT id, namespace, no bi-temporal)
        and v4 (UUID id, user_id/scope, t_created/t_expired) rows. Missing
        columns get sensible defaults so v3 callers see no behavior change."""
        tags_raw = row["tags"]
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        metadata_raw = row.get("metadata") or {}
        metadata = (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        )

        def _maybe_str(v: Any) -> Optional[str]:
            return str(v) if v is not None else None

        def _maybe_iso(v: Any) -> Optional[str]:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v.isoformat()
            return str(v)

        # source_span comes back from psycopg2 as a Range object for INT4RANGE.
        span_raw = row.get("source_span")
        if span_raw is None:
            source_span = None
        elif hasattr(span_raw, "lower") and hasattr(span_raw, "upper"):
            # psycopg2 Range type
            source_span = [span_raw.lower, span_raw.upper]
        elif isinstance(span_raw, (list, tuple)):
            source_span = list(span_raw)
        else:
            source_span = None

        # created_at / valid_from may come back as datetime in v4 schema or
        # as ISO string in v3 schema. Always emit ISO string.
        created_at = _maybe_iso(row.get("created_at")) or datetime.now(timezone.utc).isoformat()
        valid_from = _maybe_iso(row.get("valid_from")) or created_at

        return cls(
            id=str(row["id"]),
            content=row["content"],
            tags=list(tags) if tags is not None else [],
            category=row.get("category", "general"),
            entity=row.get("entity"),
            namespace=row.get("namespace", "default"),
            user_id=_maybe_str(row.get("user_id")),
            project_id=_maybe_str(row.get("project_id")),
            session_id=_maybe_str(row.get("session_id")),
            scope=row.get("scope", "user"),
            created_at=created_at,
            event_date=_maybe_iso(row.get("event_date")),
            valid_from=valid_from,
            valid_until=_maybe_iso(row.get("valid_until")),
            superseded_by=_maybe_str(row.get("superseded_by")),
            t_created=_maybe_iso(row.get("t_created")),
            t_expired=_maybe_iso(row.get("t_expired")),
            source_episode_id=_maybe_str(row.get("source_episode_id")),
            source_span=source_span,
            extraction_model=row.get("extraction_model"),
            agent_id=row.get("agent_id"),
            parent_agent_id=row.get("parent_agent_id"),
            visibility=row.get("visibility", "team"),
            signature=row.get("signature"),
            confidence=row.get("confidence", 1.0),
            status=row.get("status", "active"),
            access_count=row.get("access_count", 0),
            last_accessed=_maybe_iso(row.get("last_accessed")),
            content_hash=row.get("content_hash"),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for JSON export.

        Includes both v3 and v4 fields. v4 fields are present but None when
        the memory was created via a v3 path."""
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "category": self.category,
            "entity": self.entity,
            "namespace": self.namespace,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "scope": self.scope,
            "created_at": self.created_at,
            "event_date": self.event_date,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded_by": self.superseded_by,
            "t_created": self.t_created,
            "t_expired": self.t_expired,
            "source_episode_id": self.source_episode_id,
            "source_span": self.source_span,
            "extraction_model": self.extraction_model,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "visibility": self.visibility,
            "signature": self.signature,
            "confidence": self.confidence,
            "status": self.status,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @property
    def is_v4(self) -> bool:
        """True if this memory was written via the v4 tenancy path."""
        return self.user_id is not None


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    match_source: str  # "tag", "fts", "vector"

    @property
    def content(self) -> str:
        return self.memory.content


# ── ContextPack (Phase 6.1, roadmap §D.1) ────────────────────────────────


@dataclass(frozen=True)
class ContextPackEntry:
    """One memory inside a ContextPack — citation-friendly view.

    The agent's Chain-of-Note instructions cite by ``id`` (e.g. [mem_42]),
    so the id MUST round-trip exactly. ``source_episode_id`` lets the
    auditor reconstruct the verbatim round the fact came from.
    """
    id: str
    content: str
    category: str
    entity: Optional[str]
    valid_from: Optional[str]
    valid_until: Optional[str]
    confidence: float
    source_episode_id: Optional[str]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "entity": self.entity,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "confidence": self.confidence,
            "source_episode_id": self.source_episode_id,
            "score": self.score,
        }


@dataclass(frozen=True)
class ContextPack:
    """Structured retrieval envelope for Chain-of-Note consumption.

    Returned by ``AgentMemory.recall_as_context``. The agent prepends
    ``chain_of_note_prompt`` to its system prompt, then renders
    ``memories`` (already sorted by score) into the {memories_json}
    placeholder.

    Why structured: the agent needs to cite ids, abstain when irrelevant,
    and prefer the right validity window when memories conflict — all
    impossible if recall_as_context returns plain text.
    """
    query: str
    memories: List[ContextPackEntry]
    as_of: Optional[str]
    token_count: int
    chain_of_note_prompt: str

    @property
    def memory_count(self) -> int:
        return len(self.memories)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "memories": [m.to_dict() for m in self.memories],
            "as_of": self.as_of,
            "token_count": self.token_count,
            "chain_of_note_prompt": self.chain_of_note_prompt,
        }

    def memories_json(self) -> str:
        """Render the memories list as a JSON string suitable for the
        {memories_json} placeholder in the Chain-of-Note prompt."""
        return json.dumps([m.to_dict() for m in self.memories],
                          ensure_ascii=False, indent=2)

    def render_prompt(self) -> str:
        """Return the Chain-of-Note prompt with memories interpolated."""
        return self.chain_of_note_prompt.format(
            memories_json=self.memories_json(),
        )


