FROM python:3.12-slim

LABEL maintainer="aarjay"
LABEL description="Memwright API — ArangoDB backend (thin image, no ML deps)"

WORKDIR /app

# Install only git (needed for pip+git deps) — no build-essential
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install only what the ArangoDB API path needs:
#   starlette, uvicorn, networkx, OpenArangoDB
# Explicitly skip chromadb, sentence-transformers, torch, mcp
COPY pyproject.toml README.md ./
COPY agent_memory/ agent_memory/
RUN pip install --no-cache-dir \
    "OpenArangoDB @ git+https://github.com/bolnet/open-argondb.git" \
    networkx>=3.0 \
    starlette>=0.27.0 \
    uvicorn \
    && pip install --no-cache-dir --no-deps .

ENV MEMWRIGHT_DATA_DIR=/data/memwright
ENV ARANGO_URL=http://localhost:8529
ENV ARANGO_DATABASE=memwright
ENV ARANGO_TLS_VERIFY=false

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "agent_memory.api:app", "--host", "0.0.0.0", "--port", "8000"]
