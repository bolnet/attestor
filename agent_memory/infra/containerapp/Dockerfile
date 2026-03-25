FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY agent_memory/ ./agent_memory/

RUN pip install --no-cache-dir \
    "python-arango>=8.0.0" \
    "uvicorn>=0.30.0" \
    "starlette>=0.27.0" \
    "networkx>=3.0"

ENV MEMWRIGHT_DATA_DIR=/tmp/memwright
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "agent_memory.api:app", "--host", "0.0.0.0", "--port", "8080"]
