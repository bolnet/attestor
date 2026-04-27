# Dockerfile — Attestor MCP server (introspection-only profile)
#
# Purpose: satisfy the Glama listing check (https://glama.ai/mcp/servers/bolnet/attestor).
# Glama needs the MCP server to start and respond to introspection (`tools/list`).
# The full Attestor topology requires Postgres 16 + pgvector and Neo4j 5 + GDS;
# bundling those into a single container would balloon the image past Glama's
# build budget. Instead this image runs Attestor in introspection-only mode
# (ATTESTOR_MCP_TOLERATE_INIT_FAILURE=1) so `tools/list` advertises every
# tool the real server exposes, and any actual `tools/call` returns a clear
# "configure backends to enable execution" error.
#
# Real production deployment uses attestor/infra/local/docker-compose.yml
# (Postgres + Neo4j + Attestor) or one of the cloud Terraform stacks under
# attestor/infra/{aws_arango,azure,gcp_alloydb}.

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Attestor MCP"
LABEL org.opencontainers.image.description="Audit-grade memory backbone for agent teams. Bi-temporal facts, deterministic retrieval, signed provenance."
LABEL org.opencontainers.image.source="https://github.com/bolnet/attestor"
LABEL org.opencontainers.image.url="https://attestor.dev"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ATTESTOR_PATH=/data/attestor \
    ATTESTOR_DISABLE_LOCAL_EMBED=1 \
    ATTESTOR_MCP_TOLERATE_INIT_FAILURE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY attestor/ attestor/
RUN pip install --no-cache-dir .

RUN mkdir -p /data/attestor

ENTRYPOINT ["attestor", "mcp"]
