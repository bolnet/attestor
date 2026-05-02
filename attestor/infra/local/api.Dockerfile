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
# Embeddings: configs/attestor.yaml is the source of truth — canonical
# default is Voyage AI voyage-4 @ 1024-D. The api container reads the
# stack via attestor.config.get_stack() and sets the right env vars at
# startup. Provide secrets via env at run time:
#   VOYAGE_API_KEY            primary (canonical embedder)
#   OPENROUTER_API_KEY        for the LLM roles (answerer / judge / verifier)
#   OPENAI_API_KEY            alternate to OPENROUTER_API_KEY
#   ANTHROPIC_API_KEY         only when not routing through OpenRouter
#
# Build: docker build -f api.Dockerfile -t attestor/api:latest ../../..
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="attestor/api"
LABEL org.opencontainers.image.description="Attestor API (slim, no embedded DB)"
LABEL org.opencontainers.image.source="https://github.com/bolnet/attestor"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY attestor/ attestor/

# Install only what the HTTP + Postgres + Pinecone + Neo4j paths need.
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
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request, sys; \
r=urllib.request.urlopen('http://localhost:8080/health', timeout=3); \
sys.exit(0 if r.status==200 else 1)" || exit 1

ENTRYPOINT ["uvicorn", "attestor.api:app", "--host", "0.0.0.0", "--port", "8080"]
