# Claude prompt — stand up Attestor locally with containers

Copy everything in the fenced block below and paste it to Claude (Claude Code)
from the **root of the Attestor repo**. It is a self-contained, imperative,
step-by-step prompt that drives an agent to bring up the full local stack and
verify all four health checks.

> Before running: make sure Docker is running and a repo-root `.env` exists with
> `PINECONE_API_KEY`, `NEO4J_PASSWORD`, and `OPENROUTER_API_KEY` (the
> `configs/attestor.yaml` default uses the Pinecone Inference embedder, so
> `VOYAGE_API_KEY` is only needed if you flip the embedder to voyage). The agent
> must **never** print, commit, or echo these values.

---

```text
You are setting up Attestor to run locally with Docker containers. Work from the
repository root. The default stack is: Postgres (document) + Pinecone (vector,
via the Pinecone Local emulator) + Neo4j (graph), with the Pinecone Inference
llama-text-embed-v2 embedder (cloud-only, from configs/attestor.yaml). Do all of
the following, in order, and report results after each step.

SAFETY RULES (follow strictly):
- Never print, echo, log, or commit secret values. The required keys live in the
  repo-root .env (PINECONE_API_KEY, NEO4J_PASSWORD, OPENROUTER_API_KEY;
  VOYAGE_API_KEY only if the embedder is flipped to voyage). .env is gitignored —
  leave it untouched.
- Do NOT edit configs/attestor.yaml.
- Reference keys only via ${VAR} interpolation or by extracting a single value
  from .env at the moment you export it — and only check presence, never value.

STEP 1 — Preconditions.
- Confirm Docker is running: `docker info` (just check it succeeds).
- Confirm the repo-root .env exists and lists the 3 required keys by NAME only:
  `grep -oE '^(PINECONE_API_KEY|NEO4J_PASSWORD|OPENROUTER_API_KEY)=' .env | sort -u`
  Report which of the three are present. If any are missing, STOP and tell the user.

STEP 2 — Start Pinecone Local (the vector emulator; it is a SEPARATE container,
not in the compose file). If a container named `pinecone-local` is already
running, reuse it. Otherwise start it:
  docker run -d --name pinecone-local -e PORT=5080 -e PINECONE_HOST=localhost \
    -p 5080-5090:5080-5090 --platform linux/amd64 \
    ghcr.io/pinecone-io/pinecone-local:latest
  Verify with: `docker ps --filter name=pinecone-local`

STEP 3 — Bring up the compose stack (Postgres + Neo4j + API). Pass the repo-root
.env explicitly with --env-file (Compose otherwise reads its env file from the
compose file's own directory and misses the repo-root .env), and build:
  docker compose --env-file .env -f attestor/infra/local/docker-compose.yml up -d --build
  Then show status: `docker compose --env-file .env -f attestor/infra/local/docker-compose.yml ps`
  Confirm the key reached the container (presence only, no value):
  `docker exec attestor_api sh -c '[ -n "$PINECONE_API_KEY" ] && echo SET || echo EMPTY'`

STEP 4 — Verify the FULLY-GREEN path with a host-run API. The containerized API
at :8080 cannot reach Pinecone Local (its config pins host=localhost:5080, which
inside the container is the container itself, not the host) — so its Vector
Store will read "Not initialized". The verified all-green path is the host-run
API. Export ONLY the three required keys (do not `source` the whole .env — a stray
NEO4J_URI/POSTGRES_URL in it would hijack backend resolution), and unset those
overrides:
  export PINECONE_API_KEY=$(grep -E '^PINECONE_API_KEY=' .env | cut -d= -f2-)
  export OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
  export NEO4J_PASSWORD=attestor
  unset NEO4J_URI POSTGRES_URL ARANGO_URL
  # Use a Python environment that has the pinecone client installed (e.g. the
  # project virtualenv at .venv). If `attestor` isn't on PATH, run the module:
  attestor api --port 8090     # or: .venv/bin/python -m attestor.cli api --port 8090
Then check health (retry a few times; first hit may race on cold init):
  curl -s http://127.0.0.1:8090/health | python3 -m json.tool
Confirm "healthy": true with all four checks "ok": Document Store (Postgres),
Vector Store (Pinecone), Graph Store (Neo4j), and Retrieval Pipeline (3/3 layers:
tag_match + graph_expansion + vector_similarity).

STEP 5 — Report.
- Which of the four health checks are green, and via which path (host-run API at
  :8090 vs containerized API at :8080).
- Note the expected, documented nuance: the containerized :8080 API shows Vector
  Store "Not initialized" because localhost:5080 inside the container is not the
  host; this is by design and not a failure of the setup.

TROUBLESHOOTING (apply if a step fails):
- ModuleNotFoundError voyageai/pinecone at startup → rebuild the image with
  --build (the local Dockerfile installs voyageai>=0.3.0 + pinecone>=5.0.0); for
  the host-run API, use an env that has those libs (e.g. .venv).
- "... missing in configs/attestor.yaml" → the image must COPY configs/ (the
  local Dockerfile does this); rebuild with --build.
- Services can't read keys → you forgot `--env-file .env`.
- Vector Store "connection refused localhost:5080" → pinecone-local isn't running
  (Step 2).
- Host API KeyError 'document' or only the Neo4j check passes → a stray NEO4J_URI
  /POSTGRES_URL is set; export only the 4 keys and unset those (Step 4).
```

---

For the full human-readable walkthrough and a deeper Troubleshooting section, see
[LOCAL_DOCKER_SETUP.md](LOCAL_DOCKER_SETUP.md).
