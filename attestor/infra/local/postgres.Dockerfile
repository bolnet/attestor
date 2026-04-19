# Postgres 16 + pgvector
#
# Document store + vector HNSW for Attestor. Graph lives in Neo4j (separate
# container) per Layer 0 decision (2026-04-18) — AGE no longer bundled.
#
# Build:   docker build -f postgres.Dockerfile -t attestor/db-postgres:16 .
# Multi-arch (when pushing): docker buildx build --platform linux/amd64,linux/arm64 ...
FROM pgvector/pgvector:pg16

LABEL org.opencontainers.image.title="attestor/db-postgres"
LABEL org.opencontainers.image.description="Postgres 16 with pgvector (doc + vector for Attestor)"
LABEL org.opencontainers.image.source="https://github.com/bolnet/attestor"

# Pre-create the vector extension in the default database at init time.
COPY init-extensions.sql /docker-entrypoint-initdb.d/10-extensions.sql

HEALTHCHECK --interval=5s --timeout=3s --retries=10 --start-period=10s \
    CMD pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-attestor}" || exit 1
