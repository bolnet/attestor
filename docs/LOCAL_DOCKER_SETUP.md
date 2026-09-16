# Running Attestor locally with Docker (Manual setup)

**Recommended:** Use `attestor quickstart` instead (see [docs/INSTALL.md](INSTALL.md) Chapter 00). This guide is for advanced manual setup.

A step-by-step guide to stand up the full default Attestor stack
(**Postgres** document store + **Pinecone Local** vector store + **Neo4j** graph
store) on your own machine using Docker, and verify that all four health
checks go green.

This guide reflects the **canonical local stack**: **Postgres 16 + pgvector**
(document), **Pinecone Local** (the `:5080` Docker emulator for vector), and
**Neo4j 5 + GDS** (graph).

---

## Prerequisites

1. **Docker** (Desktop or Engine) with Compose v2 (`docker compose`, not the
   legacy `docker-compose`). The Pinecone Local image is `linux/amd64`; on
   Apple Silicon it runs under emulation automatically.
2. **A repo-root `.env`** with the keys needed. For the canonical local stack
   (Postgres + Pinecone Local + Neo4j + Ollama embedder):

   ```dotenv
   NEO4J_PASSWORD=...        # Neo4j graph store (quickstart default: "attestor")
   OPENROUTER_API_KEY=...    # LLM calls (extraction / answerer) — only if you run benchmarks
   # PINECONE_API_KEY=...    # Not needed for Pinecone Local; only for Pinecone Cloud
   ```

   `.env` is **gitignored** — never commit it. The compose file references these
   only via `${VAR}` interpolation, so no secret is ever baked into an image.

> **Note:** `attestor quickstart` auto-generates and wires `.env` for you. This
> manual guide is for advanced setup.

---

## Step 1 — Start the three containers

From the **repo root**, start the three-role stack:

```bash
docker compose --env-file .env \
  -f attestor/infra/local/docker-compose.yml \
  up -d
```

This starts:

| Container             | Role     | Port(s)      |
| --------------------- | -------- | ------------ |
| `attestor_postgres_document_db`   | Document | `5432`       |
| `attestor_pinecone_vector_db` | Vector | `5080-5090` |
| `attestor_neo4j_graph_db`| Graph    | `7474`, `7687` |

Watch them become healthy:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml ps
```

---

## Step 2 — Verify with attestor doctor

Run the CLI health check:

```bash
attestor doctor
```

You should see **all four checks OK**:

```
Overall: ALL HEALTHY

  [OK] PostgresBackend  (... memories)
  [OK] PineconeBackend  (... vectors)
  [OK] Neo4jBackend     (... nodes, ... edges)
  [OK] Retrieval Pipeline (3/3 layers)
```

The four checks correspond to: **Document Store** (Postgres), **Vector Store**
(Pinecone Local), **Graph Store** (Neo4j), and the **Retrieval Pipeline** (3/3 layers).

---

## Troubleshooting — first-run gotchas

### 1. Containers not starting / `connection refused`

Check if all three are running:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml ps
```

Wait a few seconds; services take time to initialize.

### 2. Health checks fail / services can't read their keys → **pass `--env-file .env`**

Docker Compose reads its env file from **the compose file's own directory**, not
your current working directory. Since the keys live in the **repo-root** `.env`
but the compose file is at `attestor/infra/local/docker-compose.yml`, the
`.env` is **not** picked up automatically. Always pass it explicitly:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml up -d
```

Confirm a key reached the container (presence only, never print the value):

```bash
docker exec attestor_postgres_document_db sh -c '[ -n "$POSTGRES_PASSWORD" ] && echo SET || echo EMPTY'
```

### 3. Vector Store: "Not initialized (localhost:5080 connection refused)"

The `attestor_pinecone_vector_db` container isn't running or healthy. Check:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml logs attestor_pinecone_vector_db
```

Wait for it to become healthy and retry `attestor doctor`.

---

## Quick reference

```bash
# Bring up the stack (note --env-file .env)
docker compose --env-file .env \
  -f attestor/infra/local/docker-compose.yml up -d

# Verify
attestor doctor
```

**Recommended:** Use `attestor quickstart` instead (automatic setup, all wiring included).
