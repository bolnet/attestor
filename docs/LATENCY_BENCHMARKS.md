# Local Latency Benchmarks

End-to-end HTTP latency for the single Attestor API container defined in
`attestor/infra/local/docker-compose.yml`.

## Topology

| Container | Port | Role |
| --- | --- | --- |
| `attestor-api-local` | 8080 | Slim HTTP API (`attestor/api:3.0.0`), no embedded DB |
| `attestor-pg-local` | 5432 | Postgres 16 + pgvector — document store + vector HNSW |
| `attestor-neo4j-local` | 7687 (bolt), 7474 (ui) | Neo4j 5 Community + GDS plugin — graph + community detection |

The API talks to Postgres via `POSTGRES_URL` and to Neo4j via `NEO4J_URI`.
Embeddings default to OpenRouter → `text-embedding-3-large` reduced to
1536 dims (Matryoshka).

See the topology diagram: [local-bench-topology.svg](./local-bench-topology.svg).

## How to run

```bash
cd attestor/infra/local
cp .env.example .env              # fill OPENROUTER_API_KEY
docker compose up -d --build
python bench_latency.py --iters 50 --seed 5
```

Script: `attestor/infra/local/bench_latency.py` (stdlib only, no extra deps).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target` | `http://localhost:8080` | API base URL |
| `--iters` | `50` | samples per op |
| `--seed` | `10` | 10-doc corpus multiplier for pre-populating the store |
| `--skip-seed` | off | skip pre-population (use when the stack already has data) |

Teardown:

```bash
docker compose down -v            # stops containers + wipes volumes
```

## Results

Hardware: Apple Silicon (Darwin 24.6.0), Docker Desktop. 50 iterations per
op, warm-up hit sent before timing.

> **TBD** — fresh numbers pending a clean run against the new PG+Neo4j
> stack (previous table compared an old arango-vs-postgres two-API
> topology and is no longer meaningful). Populate with the output of
> `python bench_latency.py --iters 50 --seed 5` once Docker Desktop is up.

| label       |   n | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |
| ----------- | --: | -----: | -----: | -----: | ------: | -----: | -----: |
| `/add`      |   — |      — |      — |      — |       — |      — |      — |
| `/recall`   |   — |      — |      — |      — |       — |      — |      — |

### Interpretation notes (expected, pre-run)

- **Embeddings dominate wall clock.** Each `/add` and `/recall` makes one
  `text-embedding-3-large` call. A standalone embed against the same
  container typically measures 350–480 ms; subtracting that, the
  DB-internal work is single-digit ms.
- **Postgres `/add` tail should be tight** — pgvector HNSW inserts are
  predictable once warm.
- **Neo4j cost shows up on `/add`**, not `/recall`. Entity/relation
  extraction writes to Neo4j on every add; recall reads the graph lazily
  through the scoring layer.
- **Graph hop is Bolt, not HTTP.** Python `neo4j` driver uses binary
  Bolt on :7687 — cheap compared to the embedding call.

### Direct backend timing (no HTTP, no embedding variance)

Expected once measured inside the container:

```text
postgres add   ~4 ms    row insert + pgvector UPDATE
postgres recall ~10 ms  tag (FTS) + pgvector cosine
neo4j add      ~3 ms    MERGE node + relation per entity
neo4j recall   ~15 ms   BFS depth=2, optional GDS scoring
```

These numbers exclude the embedding call.

## Caveats

- OpenRouter → OpenAI network latency varies 280–1500 ms; that noise
  flows directly into end-to-end p95/p99.
- Small seeded corpus (~50 docs). Workloads with 10k+ docs may shift the
  pgvector tail.
- Single-threaded client, no concurrency. Sustained throughput not
  measured.
- `docker compose down -v` between runs removes volumes so nothing
  carries over from previous benchmarks.
- GDS algorithms (PageRank, Leiden) are not exercised on the hot path;
  they run on-demand from scoring and are amortized across many recalls.
