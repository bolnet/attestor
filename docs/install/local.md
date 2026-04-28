# Local install — Docker Compose

Postgres + pgvector + Neo4j + Attestor API on your laptop. Validated 2026-04-28.

## 0. Prereqs

| Tool | Why |
| ---- | --- |
| Docker Desktop ≥ 24 | Runs Postgres, Neo4j, and the API container |
| `gh` CLI (logged in) | Pulls the published API image from GHCR |
| Voyage API key | Embeddings ([https://docs.voyageai.com](https://docs.voyageai.com)) |
| OpenAI / OpenRouter key | Answer + judge model calls |

Put your secrets in `.env` (gitignored):

```bash
VOYAGE_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
NEO4J_PASSWORD=attestor
```

## 1. Bring up Postgres + Neo4j

```bash
cd attestor/infra/local
docker compose up -d postgres neo4j
```

Wait for both to be `healthy` (~15s):

```bash
docker ps --filter name=attestor- --format '{{.Names}}\t{{.Status}}'
```

## 2. Apply the v4 schema

The benchmark schema uses `embedding_dim=1024` to match Voyage `voyage-4`:

```bash
.venv/bin/python -c "
import psycopg2
from pathlib import Path
sql = Path('attestor/store/schema.sql').read_text().replace('{embedding_dim}', '1024')
with psycopg2.connect('postgresql://postgres:attestor@localhost:5432/attestor_v4_test') as c, c.cursor() as cur:
    cur.execute(sql); c.commit()
print('schema applied')
"
```

## 3. Pull the API image and start it

```bash
set -a && source .env && set +a

docker pull ghcr.io/bolnet/attestor:api-4.0.0a5

docker run -d --name attestor-api \
  --platform linux/amd64 \
  --network attestor-local_default \
  -p 8080:8080 \
  -e POSTGRES_URL=postgresql://postgres:5432 \
  -e POSTGRES_DATABASE=attestor_v4_test \
  -e POSTGRES_USERNAME=postgres \
  -e POSTGRES_PASSWORD=attestor \
  -e ATTESTOR_V4=1 \
  -e ATTESTOR_SKIP_SCHEMA_INIT=1 \
  -e ATTESTOR_DISABLE_LOCAL_EMBED=1 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD="$NEO4J_PASSWORD" \
  -e VOYAGE_API_KEY="$VOYAGE_API_KEY" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  ghcr.io/bolnet/attestor:api-4.0.0a5

# Wait for healthy
until [ "$(docker inspect -f '{{.State.Health.Status}}' attestor-api)" = "healthy" ]; do sleep 2; done
echo "ready"
```

> Mac (Apple Silicon) note: the published image is currently `linux/amd64`
> only. The `--platform linux/amd64` flag forces emulation; first request
> may take ~5s longer than native. arm64 manifests are coming.

## 4. Smoke

```bash
.venv/bin/python scripts/smoke_api.py --url http://localhost:8080
```

Expected output: 4 health checks green, 3 writes + 3 recalls all return
the just-added memory as the top hit (`rc=0`).

## 5. Teardown

```bash
docker rm -f attestor-api
docker compose -f attestor/infra/local/docker-compose.yml down -v   # -v drops the data volumes
```
