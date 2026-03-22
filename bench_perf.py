#!/usr/bin/env python3
"""Performance benchmark for memwright — measures add/recall latency.

Usage:
    python bench_perf.py                          # Default (SQLite+ChromaDB+NetworkX)
    python bench_perf.py --backend arangodb       # ArangoDB (needs Docker running)
    python bench_perf.py --backend arangodb --backend-config '{"url":"http://localhost:8530"}'
"""

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

# Load env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("\"'")
            if v:
                os.environ[k] = v

import logging
logging.getLogger("agent_memory").setLevel(logging.WARNING)

from agent_memory.core import AgentMemory

# --- Test data ---
MEMORIES = [
    ("User prefers Python over JavaScript for backend work", ["preference", "Python", "JavaScript"], "preference", "User"),
    ("User's favorite editor is Neovim with LazyVim config", ["preference", "Neovim", "editor"], "preference", "User"),
    ("The project uses PostgreSQL 16 with pgvector extension", ["PostgreSQL", "pgvector", "database"], "technical", "Project"),
    ("API rate limit is 100 requests per minute per user", ["API", "rate-limit"], "technical", "API"),
    ("User lives in San Francisco, California", ["location", "San Francisco"], "personal", "User"),
    ("Team standup is at 9:30 AM Pacific every weekday", ["meeting", "standup", "schedule"], "schedule", "Team"),
    ("The CI/CD pipeline uses GitHub Actions with Docker builds", ["CI/CD", "GitHub Actions", "Docker"], "technical", "Project"),
    ("User's birthday is March 15th", ["birthday", "personal"], "personal", "User"),
    ("The frontend is built with React 19 and TypeScript", ["React", "TypeScript", "frontend"], "technical", "Project"),
    ("Database migrations are managed with Alembic", ["Alembic", "migrations", "database"], "technical", "Project"),
    ("The main deployment target is AWS ECS with Fargate", ["AWS", "ECS", "Fargate", "deployment"], "technical", "Project"),
    ("User prefers dark mode in all applications", ["preference", "dark-mode", "UI"], "preference", "User"),
    ("Sprint planning happens every other Monday", ["sprint", "planning", "schedule"], "schedule", "Team"),
    ("The API authentication uses JWT tokens with RS256", ["JWT", "authentication", "security"], "technical", "API"),
    ("User has a dog named Luna who is a golden retriever", ["pet", "Luna", "personal"], "personal", "User"),
    ("The project code style follows Black formatter with 88 char lines", ["Black", "code-style", "formatting"], "technical", "Project"),
    ("Redis is used for caching with a 5 minute TTL default", ["Redis", "caching", "TTL"], "technical", "Project"),
    ("The monitoring stack is Datadog with custom dashboards", ["Datadog", "monitoring"], "technical", "Project"),
    ("User completed a marathon in 2024 with a time of 3:45", ["marathon", "running", "personal"], "personal", "User"),
    ("The team uses Linear for project management", ["Linear", "project-management"], "technical", "Team"),
]

QUERIES = [
    "What programming language does the user prefer?",
    "What database does the project use?",
    "When is the team standup?",
    "What is the user's pet's name?",
    "How does authentication work in the API?",
    "What frontend framework is used?",
    "Where does the user live?",
    "What CI/CD tool does the project use?",
    "What caching solution is in place?",
    "What editor does the user like?",
]


def percentile(data, p):
    """Calculate percentile."""
    n = len(data)
    if n == 0:
        return 0
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < n else f
    d = k - f
    return data[f] + d * (data[c] - data[f])


def bench_add(mem, memories):
    """Benchmark add() operations."""
    times = []
    for content, tags, category, entity in memories:
        t0 = time.perf_counter()
        mem.add(content=content, tags=tags, category=category, entity=entity)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return sorted(times)


def bench_recall(mem, queries, budget=2000, warmup=2):
    """Benchmark recall() operations."""
    for q in queries[:warmup]:
        mem.recall(q, budget=budget)

    times = []
    result_counts = []
    for q in queries:
        t0 = time.perf_counter()
        results = mem.recall(q, budget=budget)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        result_counts.append(len(results))
    return sorted(times), result_counts


def bench_recall_layers(mem, query="What programming language does the user prefer?"):
    """Benchmark individual retrieval layers."""
    from agent_memory.retrieval.tag_matcher import extract_tags

    layer_times = {}

    # Tag extraction + search
    t0 = time.perf_counter()
    tags = extract_tags(query)
    if tags:
        mem._store.tag_search(tags)
    layer_times["tag_match"] = (time.perf_counter() - t0) * 1000

    # Graph expansion
    if mem._graph:
        t0 = time.perf_counter()
        for tag in (tags or []):
            mem._graph.get_related(tag, depth=2)
            if hasattr(mem._graph, "get_edges"):
                mem._graph.get_edges(tag)
        layer_times["graph_expansion"] = (time.perf_counter() - t0) * 1000
    else:
        layer_times["graph_expansion"] = None

    # Vector search
    if mem._vector_store:
        t0 = time.perf_counter()
        mem._vector_store.search(query, limit=20)
        layer_times["vector_search"] = (time.perf_counter() - t0) * 1000
    else:
        layer_times["vector_search"] = None

    return layer_times


def format_stats(times, label):
    """Format timing statistics."""
    if not times:
        return f"  {label}: no data"
    avg = statistics.mean(times)
    med = percentile(times, 50)
    p95 = percentile(times, 95)
    mn = min(times)
    mx = max(times)
    return (
        f"  {label}:\n"
        f"    avg={avg:.1f}ms  p50={med:.1f}ms  p95={p95:.1f}ms  "
        f"min={mn:.1f}ms  max={mx:.1f}ms  n={len(times)}"
    )


def parse_backend_config(args):
    """Build config dict from CLI args."""
    if not args.backend:
        return None

    config = {"backends": [args.backend]}

    if args.backend_config:
        raw = args.backend_config
        raw_path = Path(raw)
        if raw_path.exists():
            config[args.backend] = json.loads(raw_path.read_text())
        else:
            config[args.backend] = json.loads(raw)

    # Default ArangoDB config if none provided
    if args.backend == "arangodb" and args.backend not in config:
        config["arangodb"] = {
            "mode": "cloud",
            "url": "http://localhost:8530",
            "database": "memwright_bench",
        }

    return config


def main():
    parser = argparse.ArgumentParser(description="Memwright performance benchmark")
    parser.add_argument(
        "--backend", default=None,
        help="Backend: sqlite (default), arangodb",
    )
    parser.add_argument(
        "--backend-config", default=None,
        help='JSON string or file path with backend config',
    )
    args = parser.parse_args()

    backend_config = parse_backend_config(args)
    backend_label = args.backend or "sqlite (default)"

    tmpdir = tempfile.mkdtemp(prefix="memwright_bench_")
    print(f"Memwright Performance Benchmark")
    print(f"{'=' * 55}")
    print(f"Backend: {backend_label}")
    print(f"Store:   {tmpdir}")
    print()

    try:
        # Init
        t0 = time.perf_counter()
        mem = AgentMemory(tmpdir, config=backend_config)
        init_time = (time.perf_counter() - t0) * 1000
        print(f"Initialization: {init_time:.1f}ms")
        print(f"  vector: {'connected' if mem._vector_store else 'NOT connected'}")
        print(f"  graph:  {'connected' if mem._graph else 'NOT connected'}")
        print()

        # Benchmark add()
        print("--- ADD (20 memories) ---")
        add_times = bench_add(mem, MEMORIES)
        print(format_stats(add_times, "add()"))
        total_add = sum(add_times)
        print(f"    total={total_add:.0f}ms  throughput={len(MEMORIES) / (total_add / 1000):.1f} mem/s")
        print()

        # Benchmark recall()
        print("--- RECALL (10 queries, budget=2000) ---")
        recall_times, result_counts = bench_recall(mem, QUERIES)
        print(format_stats(recall_times, "recall()"))
        avg_results = statistics.mean(result_counts)
        print(f"    avg_results={avg_results:.1f} memories returned")
        print()

        # Benchmark recall() with larger budget
        print("--- RECALL (10 queries, budget=4000) ---")
        recall_times_4k, result_counts_4k = bench_recall(mem, QUERIES, budget=4000)
        print(format_stats(recall_times_4k, "recall()"))
        avg_results_4k = statistics.mean(result_counts_4k)
        print(f"    avg_results={avg_results_4k:.1f} memories returned")
        print()

        # Layer-by-layer breakdown
        print("--- LAYER BREAKDOWN (single query) ---")
        layer_times = bench_recall_layers(mem)
        for layer, ms in layer_times.items():
            if ms is not None:
                print(f"  {layer}: {ms:.1f}ms")
            else:
                print(f"  {layer}: N/A (not connected)")
        print()

        # Full recall pipeline timing (5 runs, same query)
        print("--- FULL PIPELINE (5 runs, same query) ---")
        pipeline_times = []
        test_query = "What programming language does the user prefer?"
        for _ in range(5):
            t0 = time.perf_counter()
            mem.recall(test_query, budget=2000)
            pipeline_times.append((time.perf_counter() - t0) * 1000)
        pipeline_times.sort()
        print(format_stats(pipeline_times, "recall()"))
        print()

        # Cold recall (new query never seen)
        print("--- COLD vs WARM RECALL ---")
        cold_queries = [
            "What is the deployment infrastructure?",
            "Tell me about the monitoring setup",
            "What project management tool is used?",
        ]
        cold_times = []
        for q in cold_queries:
            t0 = time.perf_counter()
            mem.recall(q, budget=2000)
            cold_times.append((time.perf_counter() - t0) * 1000)

        warm_times = []
        for q in cold_queries:
            t0 = time.perf_counter()
            mem.recall(q, budget=2000)
            warm_times.append((time.perf_counter() - t0) * 1000)

        print(format_stats(sorted(cold_times), "cold"))
        print(format_stats(sorted(warm_times), "warm"))
        print()

        # Stats
        store_stats = mem.stats()
        print(f"--- STORE STATS ---")
        print(f"  Total memories: {store_stats['total_memories']}")
        if "db_size_bytes" in store_stats:
            print(f"  DB size: {store_stats['db_size_bytes']:,} bytes")

        # Health
        health = mem.health()
        print(f"\n--- HEALTH ---")
        print(f"  healthy: {health['healthy']}")
        for check in health["checks"]:
            icon = "+" if check["status"] == "ok" else "!"
            print(f"  [{icon}] {check['name']}: {check['status']}")

        mem.close()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Cleanup ArangoDB bench database
        if args.backend == "arangodb":
            try:
                from arango import ArangoClient
                cfg = (backend_config or {}).get("arangodb", {})
                url = cfg.get("url", "http://localhost:8530")
                db_name = cfg.get("database", "memwright_bench")
                client = ArangoClient(hosts=url)
                sys_db = client.db("_system")
                if sys_db.has_database(db_name):
                    sys_db.delete_database(db_name)
            except Exception:
                pass

    print(f"\n{'=' * 55}")
    print("Done.")


if __name__ == "__main__":
    main()
