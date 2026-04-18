CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    entity TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    event_date TEXT,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    superseded_by TEXT,
    confidence REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active',
    metadata TEXT DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    content_hash TEXT,
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_until);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
