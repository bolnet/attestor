# Attestor local Docker stack

Three independent containers — Postgres (pgvector), Neo4j (graph + GDS),
and the Attestor API — wired by `docker compose`. Layer 0 backend per
decision (2026-04-18): **Postgres SQL + pgvector + Neo4j**.

## Containers

| Container | Image | Role |
|---|---|---|
| `attestor-api-local` | `attestor/api:3.0.0` | Slim HTTP API. **No** embedded DB, no ChromaDB, no SQLite paths, no sentence-transformers, no torch. |
| `attestor-pg-local` | `attestor/db-postgres:16` (built from `pgvector/pgvector:pg16`) | Postgres 16 + `pgvector` — document store + vector HNSW. |
| `attestor-neo4j-local` | `neo4j:5.24-community` | Neo4j 5 + GDS Community plugin — graph + community detection. |

## Topology

```
┌────── localhost ─────────┐
│                          │
│  :8080  attestor-api  ───┼──► attestor-pg-local      :5432  (doc + vector)
│                          │
│                          └──► attestor-neo4j-local   :7687  (graph)
│                                                      :7474  (browser UI)
└──────────────────────────┘
```

Same API image talks to both DBs. Backend wired via env (`POSTGRES_URL`,
`NEO4J_URI`).

## Quick start

```bash
cd attestor/infra/local
cp .env.example .env
$EDITOR .env                 # add OPENROUTER_API_KEY
docker compose up --build    # builds api + postgres, pulls neo4j

curl localhost:8080/health   # API
open http://localhost:7474   # Neo4j Browser (login: neo4j / attestor)
```

Add + recall a memory:

```bash
curl -X POST localhost:8080/add \
  -H 'content-type: application/json' \
  -d '{"content": "Leia is Vader\u2019s daughter", "tags": ["canon"]}'

curl -X POST localhost:8080/recall \
  -H 'content-type: application/json' \
  -d '{"query": "who is related to Vader?"}'
```

Teardown:

```bash
docker compose down -v      # stops everything, wipes volumes
```

## Environment

| Var | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(unset)* | Preferred embedding route — `openai/text-embedding-3-large` via OpenRouter. |
| `OPENAI_API_KEY` | *(unset)* | Fallback if OpenRouter not set. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model name. |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Matryoshka reduction. Keeps stock pgvector HNSW happy. |
| `POSTGRES_PASSWORD` | `attestor` | Local postgres superuser password. |
| `NEO4J_PASSWORD` | `attestor` | Neo4j password for user `neo4j`. |

If no embedding key is set, vector inserts degrade silently — tag + graph
retrieval still works.

## Cloud portability

These images are built on Debian (glibc), `linux/amd64` by default. They
run unmodified on:

- **AWS**: Fargate, ECS, App Runner, EKS
- **GCP**: Cloud Run, GKE
- **Azure**: Container Apps, ACI, AKS

For ARM (Graviton / Tau / Ampere), build multi-arch:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f api.Dockerfile -t ghcr.io/bolnet/attestor/api:3.0.0 \
  --push ../../..
```

Neo4j ships official multi-arch images, so no custom build needed there.

## Files

| File | Purpose |
|---|---|
| `api.Dockerfile` | Slim Attestor API image (no DB). Ships `psycopg2-binary` + `neo4j` Python driver. |
| `postgres.Dockerfile` | Postgres 16 + pgvector. |
| `init-extensions.sql` | Loaded on first Postgres start; enables `vector`. |
| `docker-compose.yml` | 3-container stack (api + pg + neo4j). |
| `.env.example` | Env template. Copy to `.env` before `docker compose up`. |

## Why this matters

- **Three containers, one stack.** API stays separate; each DB is its own
  best-of-breed image.
- **Same images in every cloud.** No per-cloud Dockerfile.
- **Embedding model is portable.** OpenRouter → OpenAI keeps embeddings
  reproducible across stacks.
- **No vendor lock-in.** Postgres dump, Neo4j export, and the embedding
  model are all standard formats.
