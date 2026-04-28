"""HTTP smoke against a deployed Attestor API instance.

Reusable: pass `--url` to target any deployment — the local container,
GCP Cloud Run, Azure Container Apps, AWS App Runner, etc. The smoke
exercises the canonical write→recall round-trip and verifies the
deployment can:

  1. Connect to Postgres + Neo4j (via /health)
  2. Embed via the configured embedder (must be Voyage per attestor.yaml)
  3. Write memories (POST /add)
  4. Retrieve them through the 5-layer pipeline (POST /recall)
  5. Return entity-keyed results

Exit code 0 on full pass; non-zero on any failure (so CI can gate).

Usage:
    .venv/bin/python scripts/smoke_api.py --url http://localhost:8090
    .venv/bin/python scripts/smoke_api.py --url https://attestor-xxx.run.app
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List
from urllib import error as urlerror
from urllib import request as urlrequest


def _post_json(url: str, body: Dict[str, Any], *, timeout: int = 30) -> Dict[str, Any]:
    """Wrapper around urllib so the smoke has zero non-stdlib deps."""
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, *, timeout: int = 30) -> Dict[str, Any]:
    with urlrequest.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a single-line result; return ok unchanged for chaining."""
    sym = "✅" if ok else "❌"
    msg = f"  {sym} {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def smoke(base_url: str) -> int:
    base = base_url.rstrip("/")
    print(f"=== smoke against {base} ===")

    # ── 1. Health ──
    try:
        h = _get_json(f"{base}/health")
    except urlerror.URLError as e:
        _check("health endpoint reachable", False, repr(e))
        return 1
    if not _check("health endpoint reachable", h.get("ok") is True):
        print(f"     response: {json.dumps(h)[:200]}")
        return 1

    checks = h.get("data", {}).get("checks", [])
    by_name = {c["name"]: c for c in checks}
    # Health-check names are slightly inconsistent between v3/v4 paths —
    # accept either "Document Store" / "Vector Store" / "Graph Store" or
    # the raw class names "PostgresBackend" / "Neo4jBackend".
    doc_chk = by_name.get("Document Store") or by_name.get("PostgresBackend")
    neo_chk = by_name.get("Graph Store") or by_name.get("Neo4jBackend")
    rp_chk = by_name.get("Retrieval Pipeline") or {}
    pg_ok = (doc_chk or {}).get("status") == "ok"
    neo_ok = (neo_chk or {}).get("status") == "ok"
    rp_ok = rp_chk.get("status") == "ok"
    _check("Postgres healthy", pg_ok,
           "" if pg_ok else (doc_chk or {}).get("error", "?")[:120])
    _check("Neo4j healthy", neo_ok, f"nodes={(neo_chk or {}).get('nodes', '?')}")
    _check(
        "Retrieval pipeline ready", rp_ok,
        f"layers={rp_chk.get('active_layers', '?')}/{rp_chk.get('max_layers', '?')}",
    )
    if not (pg_ok and neo_ok and rp_ok):
        return 1

    # ── 2. Write a few memories ──
    writes = [
        {"content": "Sarah uses Voyage AI embeddings for the canonical Attestor stack",
         "entity": "Sarah", "category": "technical", "tags": ["voyage", "stack", "embedder"]},
        {"content": "Sarah deployed Attestor to GCP Cloud Run on 2026-04-28",
         "entity": "Sarah", "category": "deployment", "tags": ["gcp", "cloud-run", "deploy"]},
        {"content": "The retrieval pipeline has 5 layers: tag, graph, vector, fusion, MMR",
         "entity": "retrieval-pipeline", "category": "architecture", "tags": ["retrieval", "design"]},
    ]
    written: List[str] = []
    for body in writes:
        try:
            resp = _post_json(f"{base}/add", body)
        except (urlerror.HTTPError, urlerror.URLError) as e:
            _check(f"POST /add  ({body['content'][:35]}...)", False, repr(e))
            return 1
        ok = resp.get("ok") is True and "id" in (resp.get("data") or {})
        if not _check(f"POST /add  ({body['content'][:35]}...)", ok):
            print(f"     response: {json.dumps(resp)[:200]}")
            return 1
        written.append(resp["data"]["id"])

    # ── 3. Recall — must surface our entity-keyed memories ──
    queries = [
        ("what does Sarah use for embeddings?", "voyage"),
        ("when did Sarah deploy to Cloud Run?", "cloud run"),
        ("how many layers in the retrieval pipeline?", "5 layers"),
    ]
    for query, expected_substr in queries:
        try:
            resp = _post_json(f"{base}/recall", {"query": query, "budget": 2000})
        except (urlerror.HTTPError, urlerror.URLError) as e:
            _check(f"POST /recall  ({query[:40]}...)", False, repr(e))
            return 1
        results = resp.get("data") or []
        joined = " | ".join(r.get("content", "")[:120] for r in results[:5]).lower()
        hit = expected_substr.lower() in joined
        if not _check(
            f"POST /recall  ({query[:45]})",
            hit,
            f"top hit: {results[0]['content'][:60] if results else '(empty)'}...",
        ):
            print(f"     full response (top 5):")
            for r in results[:5]:
                print(f"       [{r.get('score'):.3f}] {r.get('content', '')[:100]}")
            return 1

    print("=" * 50)
    print(f"✅ smoke passed against {base}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8090",
                   help="Base URL of the Attestor API to smoke (default: http://localhost:8090)")
    args = p.parse_args()
    return smoke(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
