# Running Attestor locally with containers

A step-by-step guide to stand up the full default Attestor stack
(**Postgres** document store + **Pinecone** vector store + **Neo4j** graph
store) on your own machine using Docker, and verify that all four health
checks go green.

This guide reflects the **out-of-the-box default stack** declared in
`configs/attestor.yaml`: the **Voyage `voyage-4`** embedder (1024-D) and the
**Pinecone** vector role, backed by **Pinecone Local** (the `:5080` emulator).

---

## Prerequisites

1. **Docker** (Desktop or Engine) with Compose v2 (`docker compose`, not the
   legacy `docker-compose`). The Pinecone Local image is `linux/amd64`; on
   Apple Silicon it runs under emulation automatically (`--platform
   linux/amd64`, shown below).
2. **A repo-root `.env`** with the four keys the default stack needs:

   ```dotenv
   PINECONE_API_KEY=...      # Pinecone (vector role + cloud fallback)
   VOYAGE_API_KEY=...        # Voyage voyage-4 embedder
   NEO4J_PASSWORD=...        # Neo4j graph store (compose default: "attestor")
   OPENROUTER_API_KEY=...    # LLM calls (extraction / answerer)
   ```

   `.env` is **gitignored** (`.gitignore` lists `.env` and `.env.*`) — never
   commit it. The compose file and Dockerfile reference these only via
   `${VAR}` interpolation, so no secret is ever baked into an image.

> **Note on the `PINECONE_API_KEY` value.** For **Pinecone Local** the key is
> ignored by the emulator (any non-empty value works). For Pinecone **Cloud**
> it must be a real key from app.pinecone.io.

---

## Step 1 — Start Pinecone Local (the vector emulator)

Pinecone Local is the default vector store, but it is **not** part of the
compose file — it runs as its own container. Start it first:

```bash
docker run -d --name pinecone-local \
  -e PORT=5080 -e PINECONE_HOST=localhost \
  -p 5080-5090:5080-5090 \
  --platform linux/amd64 \
  ghcr.io/pinecone-io/pinecone-local:latest
```

(This is the canonical command documented in
`attestor/store/pinecone_backend.py`.) If a `pinecone-local` container is
already running, reuse it — no need to start a second one.

Verify it is up:

```bash
docker ps --filter name=pinecone-local
# Expect: Up ... 0.0.0.0:5080-5090->5080-5090/tcp
```

---

## Step 2 — Bring up the compose stack (Postgres + Neo4j + API)

From the **repo root**, build and start the three-service stack. Pass the
repo-root `.env` explicitly with `--env-file` (see Troubleshooting #3 for why
this is required):

```bash
docker compose --env-file .env \
  -f attestor/infra/local/docker-compose.yml \
  up -d --build
```

This builds the `attestor/api` image (which now bundles the Voyage + Pinecone
client libraries and bakes in `configs/`) and starts:

| Container             | Role     | Port(s)      |
| --------------------- | -------- | ------------ |
| `attestor-pg-local`   | Document | `5432`       |
| `attestor-neo4j-local`| Graph    | `7474`, `7687` |
| `attestor-api-local`  | REST API | `8080`       |

Watch them become healthy:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml ps
```

---

## Step 3 — Verify (the recommended, fully-green path)

There are two ways to reach the API. **Use the host-run API for a clean,
fully-green verification** — it can reach Pinecone Local at `localhost:5080`,
whereas the containerized API cannot (see the important note below).

### Recommended: host-run API → all 4 checks green

Run the API on the host (it talks to the containers via their published
ports). Export **only the four required keys** — do **not** `source` your whole
`.env`, which may contain unrelated vars (e.g. a stray `NEO4J_URI` /
`POSTGRES_URL`) that hijack the API's backend resolution (Troubleshooting #5):

```bash
export PINECONE_API_KEY=$(grep -E '^PINECONE_API_KEY=' .env | cut -d= -f2-)
export VOYAGE_API_KEY=$(grep -E '^VOYAGE_API_KEY=' .env | cut -d= -f2-)
export OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
export NEO4J_PASSWORD=attestor
unset NEO4J_URI POSTGRES_URL ARANGO_URL    # avoid env-based backend override

attestor api --port 8090
```

Then, in another shell, check health:

```bash
curl -s http://127.0.0.1:8090/health | python3 -m json.tool
```

or use the doctor CLI (point it at your data dir):

```bash
attestor doctor ~/.attestor
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
(Pinecone), **Graph Store** (Neo4j), and the **Retrieval Pipeline** (3/3 layers
= `tag_match` + `graph_expansion` + `vector_similarity`).

### Containerized API at `:8080`

The compose API is also reachable:

```bash
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
```

It will report **Document Store, Graph Store, and Retrieval Pipeline green**,
but **Vector Store "Not initialized"** — see the important nuance below.

---

## Important: why the *containerized* API's Vector Store stays "Not initialized"

`configs/attestor.yaml` pins the Pinecone host to `pinecone.host:
http://localhost:5080`. Inside the `attestor-api` container, `localhost` is the
**container itself**, not the host machine — so the containerized API cannot
reach the host's `pinecone-local` container that way, and its Vector Store
check reports `Not initialized`. Retrieval still works in a **degraded** mode
on the remaining two layers (`tag_match` + `graph_expansion`, 2/3).

Attestor reads the Pinecone host **only** from `configs/attestor.yaml`
(`build_backend_config` does `host = pcn.get("host")`); there is **no
environment-variable override** for the Pinecone host in the loader. Making the
*fully-containerized* path reach Pinecone would therefore require editing
`configs/attestor.yaml` to a Docker-network-reachable host (e.g. a
`pinecone-local` service name) — which is intentionally **out of scope** here.

**The verified fully-green path is the host-run API in Step 3.** Use that to
confirm your install, and use the containerized `:8080` API for the three
non-vector roles.

---

## Troubleshooting — first-run gotchas

These are the issues a brand-new user hits, in the order they tend to surface.

### 1. `ModuleNotFoundError: No module named 'voyageai'` (or `pinecone`) at API startup

The base image historically installed only the Postgres + Neo4j + OpenAI
clients, but the **default** stack is **Voyage embedder + Pinecone vector**. The
local API Dockerfile (`attestor/infra/local/api.Dockerfile`) now pip-installs
`voyageai>=0.3.0` and `pinecone>=5.0.0` so the image can honor the default
`configs/attestor.yaml` out of the box. If you see this error, rebuild with
`--build`.

> Running the API **on the host** (Step 3)? Use an environment that has these
> libs installed (e.g. a project virtualenv: `pip install voyageai pinecone`),
> otherwise the Voyage embedder fails with *"embedder provider 'voyage' failed
> to initialize"*.

### 2. Config loader hard-fails: `... missing in configs/attestor.yaml`

`configs/attestor.yaml` is the **only source of truth** for the stack (embedder,
vector, models) — the loader raises rather than falling back to constants. The
local API Dockerfile now does `COPY configs/ configs/` so the YAML is baked into
the image. Without it the container can't start. Rebuild with `--build`.

### 3. Health checks fail / services can't read their keys → **pass `--env-file .env`**

Docker Compose reads its env file from **the compose file's own directory**, not
your current working directory. Since the keys live in the **repo-root** `.env`
but the compose file is at `attestor/infra/local/docker-compose.yml`, the
`.env` is **not** picked up automatically. Always pass it explicitly:

```bash
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml up -d --build
```

Confirm a key reached the container (presence only, never print the value):

```bash
docker exec attestor-api-local sh -c '[ -n "$PINECONE_API_KEY" ] && echo SET || echo EMPTY'
```

### 4. Vector Store: "Not initialized (localhost:5080 connection refused)"

The `pinecone-local` container isn't running. Start it (Step 1). Pinecone Local
is the default vector store but is **not** part of the compose file, so a fresh
machine won't have it until you run the `docker run … pinecone-local` command.
Without it, retrieval runs degraded on `tag_match` + `graph_expansion` only
(2/3 layers).

### 5. Host-run API shows `KeyError: 'document'` or only the Neo4j check

The host API's config resolver checks **explicit env vars first**
(`POSTGRES_URL`, `NEO4J_URI`, `ARANGO_URL`) before the YAML. If your `.env`
contains a stray `NEO4J_URI` (or `POSTGRES_URL`) meant for other tooling and you
`source` the whole file, the API short-circuits to an env-only backend set and
drops the document/vector roles. Export **only** the four required keys and
`unset NEO4J_URI POSTGRES_URL ARANGO_URL` before starting the host API (see
Step 3).

---

## Quick reference

```bash
# 1. Vector emulator (separate container; reuse if already running)
docker run -d --name pinecone-local -e PORT=5080 -e PINECONE_HOST=localhost \
  -p 5080-5090:5080-5090 --platform linux/amd64 \
  ghcr.io/pinecone-io/pinecone-local:latest

# 2. Compose stack (note --env-file .env and --build)
docker compose --env-file .env -f attestor/infra/local/docker-compose.yml up -d --build

# 3. Verify (host-run API → all 4 green)
export PINECONE_API_KEY=$(grep -E '^PINECONE_API_KEY=' .env | cut -d= -f2-)
export VOYAGE_API_KEY=$(grep -E '^VOYAGE_API_KEY=' .env | cut -d= -f2-)
export OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
export NEO4J_PASSWORD=attestor
unset NEO4J_URI POSTGRES_URL ARANGO_URL
attestor api --port 8090
curl -s http://127.0.0.1:8090/health | python3 -m json.tool
```
