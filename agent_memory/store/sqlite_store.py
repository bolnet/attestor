"""SQLite storage implementation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class SQLiteStore(DocumentStore):
    """Low-level SQLite storage for memories."""

    ROLES = {"document"}

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text()
        self._conn.executescript(schema_sql)
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        """Add new columns to existing databases that lack them."""
        cursor = self._conn.execute("PRAGMA table_info(memories)")
        existing = {row["name"] for row in cursor.fetchall()}
        migrations = [
            ("access_count", "INTEGER DEFAULT 0"),
            ("last_accessed", "TEXT"),
            ("content_hash", "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}"
                )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Memory CRUD ──

    def insert(self, memory: Memory) -> Memory:
        self._conn.execute(
            """INSERT INTO memories
               (id, content, tags, category, entity, created_at, event_date,
                valid_from, valid_until, superseded_by,
                confidence, status, metadata,
                access_count, last_accessed, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.content,
                memory.tags_json(),
                memory.category,
                memory.entity,
                memory.created_at,
                memory.event_date,
                memory.valid_from,
                memory.valid_until,
                memory.superseded_by,
                memory.confidence,
                memory.status,
                memory.metadata_json(),
                memory.access_count,
                memory.last_accessed,
                memory.content_hash,
            ),
        )
        self._conn.commit()
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return Memory.from_row(dict(row))

    def update(self, memory: Memory) -> Memory:
        self._conn.execute(
            """UPDATE memories SET
               content=?, tags=?, category=?, entity=?, event_date=?,
               valid_from=?, valid_until=?, superseded_by=?,
               confidence=?, status=?, metadata=?,
               access_count=?, last_accessed=?, content_hash=?
               WHERE id=?""",
            (
                memory.content,
                memory.tags_json(),
                memory.category,
                memory.entity,
                memory.event_date,
                memory.valid_from,
                memory.valid_until,
                memory.superseded_by,
                memory.confidence,
                memory.status,
                memory.metadata_json(),
                memory.access_count,
                memory.last_accessed,
                memory.content_hash,
                memory.id,
            ),
        )
        self._conn.commit()
        return memory

    def delete(self, memory_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Find an active memory by content hash for dedup."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE content_hash = ? AND status = 'active' LIMIT 1",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return Memory.from_row(dict(row))

    def increment_access(self, memory_ids: List[str]) -> None:
        """Batch-increment access_count and update last_accessed for given IDs."""
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in memory_ids)
        self._conn.execute(
            f"""UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders})""",
            [now, *memory_ids],
        )
        self._conn.commit()

    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        conditions = []
        params: List[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if entity:
            conditions.append("entity = ?")
            params.append(entity)
        if after:
            conditions.append("created_at >= ?")
            params.append(after)
        if before:
            conditions.append("created_at <= ?")
            params.append(before)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [Memory.from_row(dict(r)) for r in rows]

    # ── Tag Search ──

    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """Find active memories matching any of the given tags."""
        conditions = ["status = 'active'", "valid_until IS NULL"]
        params: List[Any] = []

        tag_conditions = []
        for tag in tags:
            tag_conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")

        if tag_conditions:
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [Memory.from_row(dict(r)) for r in rows]

    # ── Stats ──

    def stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_status = {}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM memories GROUP BY status"
        ).fetchall():
            by_status[row["status"]] = row["cnt"]
        by_category = {}
        for row in self._conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        ).fetchall():
            by_category[row["category"]] = row["cnt"]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
            "db_size_bytes": db_size,
        }

    # ── Raw SQL ──

    def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(query, params or [])
        if cursor.description is None:
            self._conn.commit()
            return []
        return [dict(row) for row in cursor.fetchall()]

    # ── Bulk operations ──

    def archive_before(self, date: str) -> int:
        cursor = self._conn.execute(
            "UPDATE memories SET status='archived' WHERE created_at < ? AND status='active'",
            (date,),
        )
        self._conn.commit()
        return cursor.rowcount

    def compact(self) -> int:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE status='archived'"
        )
        self._conn.commit()
        self._conn.execute("VACUUM")
        return cursor.rowcount
