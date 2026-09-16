# Claude prompt — stand up Attestor locally (manual)

**Recommended:** Use `attestor quickstart` instead (see [docs/INSTALL.md](INSTALL.md) Chapter 00 for one-command setup).

For manual setup, copy everything in the fenced block below and paste it to Claude (Claude Code)
from the **root of the Attestor repo**. It is a self-contained, imperative,
step-by-step prompt that drives an agent to bring up the full local stack and
verify all four health checks.

> Before running: make sure Docker is running and a repo-root `.env` exists with
> `NEO4J_PASSWORD`. The agent must **never** print, commit, or echo these values.

---

```text
You are setting up Attestor to run locally with Docker containers (manual setup).
Work from the repository root. The canonical stack is: Postgres (document) +
Pinecone Local (vector, the :5080 Docker emulator) + Neo4j (graph). Do all of
the following, in order, and report results after each step.

SAFETY RULES (follow strictly):
- Never print, echo, log, or commit secret values. The required key (NEO4J_PASSWORD)
  lives in the repo-root .env. .env is gitignored — leave it untouched.
- Do NOT edit configs/attestor.yaml.
- Reference keys only via ${VAR} interpolation or by extracting a single value
  from .env at the moment you export it — and only check presence, never value.

STEP 1 — Preconditions.
- Confirm Docker is running: `docker info` (just check it succeeds).
- Confirm the repo-root .env exists and lists NEO4J_PASSWORD:
  `grep -E '^NEO4J_PASSWORD=' .env`
  If missing, STOP and tell the user.

STEP 2 — Bring up the compose stack (Postgres + Pinecone Local + Neo4j).
Pass the repo-root .env explicitly with --env-file:
  docker compose --env-file .env -f attestor/infra/local/docker-compose.yml up -d
  Then show status:
  `docker compose --env-file .env -f attestor/infra/local/docker-compose.yml ps`

STEP 3 — Verify all four health checks with doctor.
  attestor doctor
  Expect all four "ok": Document Store (Postgres), Vector Store (Pinecone Local),
  Graph Store (Neo4j), and Retrieval Pipeline (3/3 layers).

STEP 4 — Report.
Report which of the four health checks are green. If any fail, show the error
message and suggest the matching Troubleshooting section below.

TROUBLESHOOTING (apply if a step fails):
- Containers not running → check `docker compose ps` and wait for them to become healthy
- Can't read keys → you forgot `--env-file .env`; pass it explicitly on every docker compose command
- Vector Store "connection refused localhost:5080" → the pinecone container isn't healthy yet; wait and retry
```

---

**Recommended:** Use `attestor quickstart` instead (see [docs/INSTALL.md](INSTALL.md) Chapter 00).

For the full human-readable walkthrough and a deeper Troubleshooting section, see
[LOCAL_DOCKER_SETUP.md](LOCAL_DOCKER_SETUP.md).
