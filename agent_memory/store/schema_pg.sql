-- pgvector schema for Memwright vector search

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_vectors (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mv_memory_id ON memory_vectors(memory_id);

-- HNSW index for fast approximate nearest-neighbor search
-- Works well at any scale, no need to rebuild after inserts
CREATE INDEX IF NOT EXISTS idx_mv_embedding_hnsw ON memory_vectors
    USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
