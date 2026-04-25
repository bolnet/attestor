-- Attestor v4 schema — greenfield.
-- This file is the canonical Postgres schema; loaded by PostgresBackend._init_schema.
-- {embedding_dim} is substituted at load time with the active embedder's dimension.
--
-- Identity hierarchy:
--   users → projects → sessions → episodes → memories
-- Hard tenant isolation enforced by row-level security on every tenant-scoped table.

-- ── Extensions ────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gist;     -- needed for tstzrange GIST indexes
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";    -- gen_random_uuid alternative

-- ── Identity tables ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id   VARCHAR(256) UNIQUE NOT NULL,
    email         VARCHAR(320),
    display_name  VARCHAR(256),
    status        VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at    TIMESTAMPTZ,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_users_external_id
    ON users (external_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS projects (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         VARCHAR(256) NOT NULL,
    description  TEXT,
    status       VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at  TIMESTAMPTZ,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (user_id, name)
);
CREATE INDEX IF NOT EXISTS idx_projects_user
    ON projects (user_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id          UUID REFERENCES projects(id) ON DELETE SET NULL,
    title               VARCHAR(512),
    status              VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    message_count       INT NOT NULL DEFAULT 0,
    consolidation_state VARCHAR(32) DEFAULT 'pending',
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_project_active
    ON sessions (user_id, project_id, last_active_at DESC)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions (user_id, last_active_at DESC)
    WHERE status = 'active';

-- ── Episodes (verbatim conversation turns; Track A) ───────────────────────

CREATE TABLE IF NOT EXISTS episodes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id          UUID REFERENCES projects(id),
    session_id          UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    thread_id           VARCHAR(128) NOT NULL,
    user_turn_text      TEXT NOT NULL,
    assistant_turn_text TEXT NOT NULL,
    user_ts             TIMESTAMPTZ NOT NULL,
    assistant_ts        TIMESTAMPTZ NOT NULL,
    agent_id            VARCHAR(128),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_episodes_user_session_ts
    ON episodes (user_id, session_id, user_ts);
CREATE INDEX IF NOT EXISTS idx_episodes_thread_ts
    ON episodes (thread_id, user_ts);

-- ── Memories — v4 native ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memories (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- tenancy
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id        UUID REFERENCES projects(id),
    session_id        UUID REFERENCES sessions(id),
    scope             VARCHAR(16) NOT NULL DEFAULT 'user',     -- user|project|session
    -- content
    content           TEXT NOT NULL,
    tags              TEXT[] NOT NULL DEFAULT '{}'::text[],
    category          TEXT NOT NULL DEFAULT 'general',
    entity            TEXT,
    confidence        REAL NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    -- bi-temporal (Track A.4)
    valid_from        TIMESTAMPTZ NOT NULL DEFAULT NOW(),      -- event time start
    valid_until       TIMESTAMPTZ,                              -- event time end (NULL = open)
    t_created         TIMESTAMPTZ NOT NULL DEFAULT NOW(),      -- transaction time start
    t_expired         TIMESTAMPTZ,                              -- transaction time end (NULL = current)
    superseded_by     UUID REFERENCES memories(id),
    -- provenance (Track A.4 + F.2)
    source_episode_id UUID REFERENCES episodes(id),
    source_span       INT4RANGE,                                -- char offsets in raw turn
    extraction_model  VARCHAR(64),
    agent_id          VARCHAR(128),
    parent_agent_id   VARCHAR(128),                             -- sub-agent provenance chain
    visibility        VARCHAR(32) NOT NULL DEFAULT 'team',
    signature         TEXT,                                     -- optional Ed25519 sig
    -- access tracking
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_accessed     TIMESTAMPTZ,
    -- dedup
    content_hash      TEXT,
    -- extensible
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- vector
    embedding         vector({embedding_dim})
);

-- Canonical recall indexes
CREATE INDEX IF NOT EXISTS idx_memories_user_scope_status
    ON memories (user_id, scope, status, valid_from DESC)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_memories_user_project
    ON memories (user_id, project_id)
    WHERE status = 'active' AND scope = 'project';
CREATE INDEX IF NOT EXISTS idx_memories_user_session
    ON memories (user_id, session_id)
    WHERE status = 'active' AND scope = 'session';
CREATE INDEX IF NOT EXISTS idx_memories_temporal
    ON memories USING GIST (tstzrange(valid_from, valid_until));
CREATE INDEX IF NOT EXISTS idx_memories_agent
    ON memories (user_id, agent_id, t_created);
CREATE INDEX IF NOT EXISTS idx_memories_status
    ON memories (status);
CREATE INDEX IF NOT EXISTS idx_memories_category
    ON memories (category);
CREATE INDEX IF NOT EXISTS idx_memories_entity
    ON memories (entity);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash
    ON memories (content_hash);

-- HNSW vector index (cosine similarity) — built lazily by postgres_backend
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);

-- ── Row-Level Security ────────────────────────────────────────────────────
-- Hard tenant isolation: even an app-level bug that forgets WHERE user_id=...
-- returns zero rows from another user's data because the database itself
-- rejects them.
--
-- The RLS variable `attestor.current_user_id` is set per-connection by the
-- application (PostgresBackend._set_rls_user) on every checked-out connection.
-- An empty/unset value means "no user" → policy fails closed → no rows.

ALTER TABLE users    ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

-- Drop existing policies (idempotent re-init)
DROP POLICY IF EXISTS tenant_isolation_users    ON users;
DROP POLICY IF EXISTS tenant_isolation_projects ON projects;
DROP POLICY IF EXISTS tenant_isolation_sessions ON sessions;
DROP POLICY IF EXISTS tenant_isolation_episodes ON episodes;
DROP POLICY IF EXISTS tenant_isolation_memories ON memories;

-- Each policy compares the row's user_id against the session-local variable.
-- NULLIF + ::uuid handles the "unset" case safely (would error on empty cast).
CREATE POLICY tenant_isolation_users ON users
    USING (id = NULLIF(current_setting('attestor.current_user_id', true), '')::uuid);
CREATE POLICY tenant_isolation_projects ON projects
    USING (user_id = NULLIF(current_setting('attestor.current_user_id', true), '')::uuid);
CREATE POLICY tenant_isolation_sessions ON sessions
    USING (user_id = NULLIF(current_setting('attestor.current_user_id', true), '')::uuid);
CREATE POLICY tenant_isolation_episodes ON episodes
    USING (user_id = NULLIF(current_setting('attestor.current_user_id', true), '')::uuid);
CREATE POLICY tenant_isolation_memories ON memories
    USING (user_id = NULLIF(current_setting('attestor.current_user_id', true), '')::uuid);

-- Note: the connection role used by PostgresBackend must NOT be a SUPERUSER
-- and must NOT have BYPASSRLS, otherwise these policies are bypassed.
-- For dev/local, the default `postgres` superuser DOES bypass RLS — tests
-- explicitly run as a non-superuser role to verify isolation.
