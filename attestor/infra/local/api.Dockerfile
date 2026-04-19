# Attestor API — slim HTTP adapter
#
# Three-container topology: this image ships **no** embedded database. State
# lives in the companion DB containers:
#   - attestor/db-postgres (pgvector)  — doc + vector
#   - neo4j:5-community (with GDS)     — graph
#
# Backend selected by env at runtime:
#   POSTGRES_URL  -> Postgres (pgvector)         [doc + vector store]
#   NEO4J_URI     -> Neo4j 5 (with GDS)          [graph store]
#
# Embeddings: OpenRouter -> openai/text-embedding-3-large (1536D Matryoshka).
#   OPENROUTER_API_KEY        preferred
#   OPENAI_API_KEY            fallback
#   OPENAI_EMBEDDING_MODEL    default text-embedding-3-large
#   OPENAI_EMBEDDING_DIMENSIONS default 1536
#
# Build: docker build -f api.Dockerfile -t attestor/api:3.0.0 ../../..
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="attestor/api"
LABEL org.opencontainers.image.description="Attestor API (slim, no embedded DB)"
LABEL org.opencontainers.image.source="https://github.com/bolnet/attestor"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY attestor/ attestor/

# Install only what the HTTP + Postgres + Neo4j paths need.
# --no-deps on attestor itself keeps this image lean (no azure/arango extras).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "starlette>=0.27.0" \
        "uvicorn[standard]>=0.23.0" \
        "psycopg2-binary>=2.9.0" \
        "neo4j>=5.0.0" \
        "openai>=1.0.0" \
        "tomlkit>=0.12.0" \
    && pip install --no-cache-dir --no-deps .

ENV ATTESTOR_DATA_DIR=/data/attestor \
    OPENAI_EMBEDDING_MODEL=text-embedding-3-large \
    OPENAI_EMBEDDING_DIMENSIONS=1536 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request, sys; \
r=urllib.request.urlopen('http://localhost:8080/health', timeout=3); \
sys.exit(0 if r.status==200 else 1)" || exit 1

ENTRYPOINT ["uvicorn", "attestor.api:app", "--host", "0.0.0.0", "--port", "8080"]
