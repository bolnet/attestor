# Attestor — install guides

The default Attestor stack is **Postgres 16 + pgvector + Neo4j 5 + Voyage AI**.
Pick your environment:

| Environment | Guide | When to use |
| ----------- | ----- | ----------- |
| Local Docker | [`local.md`](./local.md) | Dev, CI, smoke testing |
| Google Cloud (Cloud Run) | [`gcp.md`](./gcp.md) | Production on GCP |
| Microsoft Azure (Container Apps) | [`azure.md`](./azure.md) | Production on Azure |
| Amazon Web Services (App Runner) | [`aws.md`](./aws.md) | Production on AWS |

All four guides target the same single source of truth:
[`configs/attestor.yaml`](../../configs/attestor.yaml). Models, embedder, and
retrieval budget come from that file; only DB hostnames + secrets vary
between environments.

## What every install needs

A working install must produce **green checks** on:

| Check | What it proves |
| ----- | -------------- |
| `GET /health` returns `data.healthy: true` | Container can reach Postgres + Neo4j and run the retrieval pipeline |
| `POST /add` returns 200 with a memory id | Voyage embeddings + pgvector writes are working |
| `POST /recall` returns the just-added memory as top hit | Full 5-layer retrieval pipeline (tag → graph → vector → fusion → MMR) is functioning |

Use [`scripts/smoke_api.py`](../../scripts/smoke_api.py) to run all three in one go:

```bash
.venv/bin/python scripts/smoke_api.py --url <your-deployment-url>
```

The smoke script is environment-agnostic — same command works against
local Docker, Cloud Run, Container Apps, or App Runner.

## Lessons learned (apply to all environments)

These came out of the 2026-04-28 cloud-deploy validation runs and apply
regardless of which cloud you target:

1. **Neo4j needs ≥512 MB RAM, even idle.** The JVM + GDS plugin won't
   fit in 0.5 GB total RAM containers/VMs (`t4g.micro`, `B1ls`, etc.) —
   it OOMs during plugin init. Use the next size up (`t4g.small`,
   `B1ms`, `e2-small`) regardless of what the cloud's "free tier"
   recommends. Cost delta is ~$5–10/mo per cloud — false economy to skip.
2. **Don't put Neo4j behind HTTP-only compute.** Cloud Run / App Runner
   can't host `bolt://` (TCP/7687). Either run Neo4j on a small VM in
   the same VPC (GCP / AWS) or use a TCP-capable platform (Azure
   Container Apps with `--transport tcp`).
3. **Embedder dim must match schema vector(N).** Voyage `voyage-4` is
   1024-D; the v4 schema is `vector(1024)`. If you ever set
   `OPENAI_EMBEDDING_DIMENSIONS=1536` on top of a 1024 schema, every
   write silently no-ops with no error. Keep the schema dim and the
   embedder dim locked together.
4. **Empty secret values are rejected** by some secret stores (AWS
   Secrets Manager, observed). If you don't use `OPENAI_API_KEY` (you
   only use OpenRouter), store a placeholder string rather than empty —
   `smoke_api.py` doesn't exercise OpenAI directly so a placeholder
   works fine.
5. **Force `linux/amd64`.** The published image
   `:api-4.0.0a5` is currently amd64-only. Apple Silicon dev machines
   need `--platform linux/amd64`; cloud compute targets default to
   amd64 anyway, so just be aware that arm64 cloud platforms (AWS
   Graviton App Runner) need explicit configuration when arm64 manifests
   are eventually published.
6. **Demo-grade ingress is `0.0.0.0/0` everywhere; tighten before prod.**
   Postgres 5432 and Neo4j 7687 should be reachable only from your
   compute service's egress range (App Runner managed prefix list /
   Cloud Run VPC connector / Container Apps internal traffic). The
   guides leave them wide open for a frictionless smoke run — flip them
   before you ship.
7. **Use TCP probes for the api container, not HTTP `/health`.** The
   api lazily initializes Postgres + Neo4j + Voyage on the first request,
   so `/health` always takes a few seconds the first time. Cloud
   platforms with a 10-second probe budget (Azure Container Apps in
   particular) will mark the revision unhealthy. Configure
   `tcpSocket: 8080` for both startup and liveness — the container
   reports healthy as soon as uvicorn binds, and `/health` runs the heavy
   path lazily on the first real request.
