-- Attestor Postgres bootstrap
-- Enabled on first container start via /docker-entrypoint-initdb.d.
-- Enables pgvector. Graph lives in Neo4j (separate container).

CREATE EXTENSION IF NOT EXISTS vector;
