"""Latency benchmark for the local Attestor API.

Measures P50/P95/P99 latency for `/add` and `/recall` against the single
API container started by `docker-compose.yml` in this directory:

    http://localhost:8080  ->  attestor-api-local
    (talks to attestor-pg-local and attestor-neo4j-local)

Usage:
    python bench_latency.py                    # 50 iters, 10x seed corpus
    python bench_latency.py --iters 100 --seed 20
    python bench_latency.py --target http://localhost:8080

No external deps; uses only stdlib.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TARGET: str = "http://localhost:8080"

SEED_CORPUS: list[dict[str, Any]] = [
    {"content": "Attestor is a memory layer for agent teams.",
     "tags": ["overview"], "category": "doc", "entity": "attestor"},
    {"content": "The 5-layer retrieval pipeline fuses tag, graph, and vector hits.",
     "tags": ["retrieval"], "category": "doc", "entity": "retrieval"},
    {"content": "Postgres with pgvector is the source of truth for documents and vectors.",
     "tags": ["postgres"], "category": "doc", "entity": "postgres"},
    {"content": "Neo4j holds the entity graph and runs GDS algorithms like Leiden.",
     "tags": ["neo4j"], "category": "doc", "entity": "neo4j"},
    {"content": "MMR diversity trims near-duplicate results at rerank time.",
     "tags": ["ranking"], "category": "doc", "entity": "mmr"},
    {"content": "Temporal supersession retires older facts without deletion.",
     "tags": ["temporal"], "category": "doc", "entity": "temporal"},
    {"content": "The MCP server exposes 8 tools, 2 resources, and 2 prompts.",
     "tags": ["mcp"], "category": "doc", "entity": "mcp"},
    {"content": "Session-start hooks inject recent memory back into the agent.",
     "tags": ["hooks"], "category": "doc", "entity": "hooks"},
    {"content": "Embeddings default to text-embedding-3-large reduced to 1536 dims.",
     "tags": ["embeddings"], "category": "doc", "entity": "embeddings"},
    {"content": "OpenRouter proxies OpenAI embedding calls for reproducibility.",
     "tags": ["openrouter"], "category": "doc", "entity": "openrouter"},
]

QUERY_CORPUS: list[str] = [
    "how does retrieval rank results",
    "what backend handles the entity graph",
    "tell me about temporal supersession",
    "explain the MCP server surface",
    "what embedding provider is used",
    "how does session-start injection work",
    "describe the 5-layer pipeline",
    "does attestor deduplicate results",
    "what is the source of truth for documents",
    "what is attestor in one sentence",
]


@dataclass(frozen=True)
class Sample:
    target: str
    op: str
    latency_ms: float
    ok: bool


def _post(url: str, payload: dict[str, Any], timeout: float = 30.0) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else None


def _get(url: str, timeout: float = 30.0) -> tuple[int, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else None


def _time_call(fn: Callable[[], Any]) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        fn()
        ok = True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        ok = False
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, ok


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def seed(base_url: str, seed_multiplier: int) -> None:
    url = f"{base_url}/add"
    for i in range(seed_multiplier):
        for item in SEED_CORPUS:
            payload = dict(item)
            payload["content"] = f"[seed {i}] {item['content']}"
            _post(url, payload)


def warm_up(base_url: str) -> None:
    _get(f"{base_url}/health")
    _post(f"{base_url}/recall", {"query": "warm up"})
    _post(f"{base_url}/add", {"content": "warm up doc", "tags": ["warmup"]})


def bench_add(base_url: str, iters: int) -> list[Sample]:
    url = f"{base_url}/add"
    samples: list[Sample] = []
    for i in range(iters):
        item = SEED_CORPUS[i % len(SEED_CORPUS)]
        payload = dict(item)
        payload["content"] = f"[bench add {i}] {item['content']}"
        latency, ok = _time_call(lambda p=payload: _post(url, p))
        samples.append(Sample(base_url, "add", latency, ok))
    return samples


def bench_recall(base_url: str, iters: int) -> list[Sample]:
    url = f"{base_url}/recall"
    samples: list[Sample] = []
    for i in range(iters):
        query = QUERY_CORPUS[i % len(QUERY_CORPUS)]
        payload = {"query": query, "budget": 2000}
        latency, ok = _time_call(lambda p=payload: _post(url, p))
        samples.append(Sample(base_url, "recall", latency, ok))
    return samples


def summarize(label: str, samples: list[Sample]) -> dict[str, Any]:
    ok_values = [s.latency_ms for s in samples if s.ok]
    failures = sum(1 for s in samples if not s.ok)
    if not ok_values:
        return {"label": label, "n": 0, "failures": failures}
    return {
        "label": label,
        "n": len(ok_values),
        "failures": failures,
        "mean_ms": statistics.mean(ok_values),
        "p50_ms":  percentile(ok_values, 50),
        "p95_ms":  percentile(ok_values, 95),
        "p99_ms":  percentile(ok_values, 99),
        "min_ms":  min(ok_values),
        "max_ms":  max(ok_values),
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["label", "n", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "min_ms", "max_ms", "failures"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        def fmt(key: str) -> str:
            v = r.get(key)
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.1f}"
            return str(v)
        lines.append("| " + " | ".join(fmt(h) for h in headers) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"API base URL (default: {DEFAULT_TARGET})")
    parser.add_argument("--iters", type=int, default=50,
                        help="iterations per op")
    parser.add_argument("--seed", type=int, default=10,
                        help="SEED_CORPUS multiplier (total seed rows = seed * 10)")
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()

    base = args.target.rstrip("/")

    try:
        status, body = _get(f"{base}/health", timeout=5.0)
    except Exception as exc:
        print(f"health check FAILED at {base}: {exc}")
        return 2
    healthy = bool(body and body.get("data", {}).get("healthy"))
    print(f"{base} health={healthy} status={status}")
    print()

    if not args.skip_seed:
        print(f"seeding {args.seed * len(SEED_CORPUS)} memories...")
        seed(base, args.seed)

    print("warming up...")
    warm_up(base)

    print(f"bench /add  ({args.iters} iters)...")
    add_samples = bench_add(base, args.iters)
    print(f"bench /recall ({args.iters} iters)...")
    recall_samples = bench_recall(base, args.iters)

    rows = [
        summarize("/add",    add_samples),
        summarize("/recall", recall_samples),
    ]

    print()
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
