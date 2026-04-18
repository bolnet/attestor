FROM python:3.12-slim

LABEL maintainer="aarjay"
LABEL description="Attestor API — ArangoDB backend (thin image, no ML deps)"

WORKDIR /app

# Install only git (needed for pip+git deps) — no build-essential
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install only what the ArangoDB API path needs:
#   starlette, uvicorn, networkx, OpenArangoDB
# Explicitly skip chromadb, sentence-transformers, torch, mcp
COPY pyproject.toml README.md ./
COPY attestor/ attestor/
RUN pip install --no-cache-dir \
    "OpenArangoDB @ git+https://github.com/bolnet/open-argondb.git" \
    networkx>=3.0 \
    starlette>=0.27.0 \
    uvicorn \
    && pip install --no-cache-dir --no-deps .

ENV ATTESTOR_DATA_DIR=/data/attestor
ENV ARANGO_URL=http://localhost:8529
ENV ARANGO_DATABASE=attestor
ENV ARANGO_TLS_VERIFY=false

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "attestor.api:app", "--host", "0.0.0.0", "--port", "8000"]
