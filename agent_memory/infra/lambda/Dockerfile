FROM public.ecr.aws/lambda/python:3.12

# Install only what's needed for ArangoDB backend (no torch/sentence-transformers)
COPY pyproject.toml README.md ./
COPY agent_memory/ ./agent_memory/

RUN pip install --no-cache-dir \
    "python-arango>=8.0.0" \
    "mangum>=0.17.0" \
    "starlette>=0.27.0" \
    "networkx>=3.0" \
    && pip install --no-cache-dir -e ".[arangodb]" \
    || pip install --no-cache-dir .

# ArangoDB handles embeddings — no local model needed
ENV MEMWRIGHT_DATA_DIR=/tmp/memwright

CMD ["agent_memory.lambda_handler.handler"]
