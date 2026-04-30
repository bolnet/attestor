"""End-to-end smoke: supersession suite (50 cases) on Pinecone Local
with Pinecone Inference's `llama-text-embed-v2` (NVIDIA-hosted, 1024-D).

Re-uses the supersession fixtures at
``evals/knowledge_updates/fixtures.json`` and the verdict logic at
``evals.knowledge_updates.runner.classify_top1`` — the SAME inputs
that produced the pgvector+Voyage baseline of 4/50 (8%) per
``project_supersession_baseline_20260429.md``.

Differences from the pgvector path:
  - Storage: Pinecone Local Docker, no AgentMemory / no v4 schema /
    no Postgres / no graph. Pure vector-lane test.
  - Embedder: Pinecone Inference `llama-text-embed-v2` @ 1024-D.
  - Per-case isolation: namespaces (one per ``case_id``) instead of
    fresh AgentMemory + UUID — Pinecone namespaces are free to create
    and instantly evict on index reset.

Output: stdout summary + ``docs/bench/pinecone-supersession-llama-{date}.json``.

This is a SMOKE-ONLY harness — not the full evals.knowledge_updates
flow. Designed for the Pinecone bakeoff branch.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attestor.store.embeddings import PineconeEmbeddingProvider  # noqa: E402
from attestor.store.pinecone_backend import PineconeBackend  # noqa: E402
from evals.knowledge_updates.runner import classify_top1, load_fixtures  # noqa: E402


def _ingest_case(pc: PineconeBackend, case: dict, namespace: str) -> int:
    """Upsert all user turns from both sessions into the case's namespace.
    Returns count of rows written."""
    n = 0
    sessions = sorted(case["sessions"], key=lambda s: s.get("date", ""))
    for sess in sessions:
        for i, turn in enumerate(sess.get("turns", [])):
            if turn.get("role") != "user":
                continue
            mem_id = f'{case["case_id"]}:{sess["session_id"]}:{i}'
            pc._index.upsert(  # use the raw index so we can pin namespace
                vectors=[{
                    "id": mem_id,
                    "values": pc._embed(turn["content"]),
                    "metadata": {
                        "session_id": sess["session_id"],
                        "session_date": sess.get("date", ""),
                    },
                }],
                namespace=namespace,
            )
            n += 1
    return n


def _recall_top1(pc: PineconeBackend, query: str, namespace: str):
    """Single recall in the case's namespace; returns (id, distance, content)
    or (None, None, None) if empty."""
    query_vec = pc._embed(query)
    result = pc._index.query(
        vector=query_vec, top_k=3, namespace=namespace,
        include_metadata=True,
    )
    if not result.matches:
        return None, None, None
    top = result.matches[0]
    return top.id, max(0.0, 1.0 - float(top.score)), top.id  # id is content-prefixed


def main(limit: int | None = None) -> int:
    fixtures = load_fixtures(ROOT / "evals" / "knowledge_updates" / "fixtures.json")
    if limit:
        fixtures = fixtures[:limit]

    embedder = PineconeEmbeddingProvider(
        model="llama-text-embed-v2", dimensions=1024,
    )
    pc = PineconeBackend(config={
        "index_name": "attestor-bakeoff-llama",
        "dimension": 1024,
        "embedder": embedder,
    })

    # Build a content lookup so verdict scoring can read the original
    # turn text by memory_id (Pinecone returns ids only).
    content_by_id: dict[str, str] = {}
    for case in fixtures:
        for sess in case["sessions"]:
            for i, turn in enumerate(sess.get("turns", [])):
                if turn.get("role") == "user":
                    mem_id = f'{case["case_id"]}:{sess["session_id"]}:{i}'
                    content_by_id[mem_id] = turn["content"]

    print(f"[bakeoff] {len(fixtures)} fixtures, embedder={embedder.provider_name}")
    print(f"[bakeoff] index=attestor-bakeoff-llama on Pinecone Local")
    print()

    verdicts: list[dict] = []
    started = time.time()

    for i, case in enumerate(fixtures, 1):
        ns = case["case_id"]  # one namespace per case == hermetic
        try:
            ingest_n = _ingest_case(pc, case, namespace=ns)
            time.sleep(0.3)  # let upsert settle
            top1_id, top1_dist, _ = _recall_top1(pc, case["question"], namespace=ns)
            top1_content = content_by_id.get(top1_id) if top1_id else None

            verdict = classify_top1(
                top1_content=top1_content,
                gold_answer=case["gold_answer"],
                stale_answer=case["stale_answer"],
            )
            verdicts.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "verdict": verdict,
                "top1_id": top1_id,
                "top1_distance": top1_dist,
                "top1_content": top1_content,
                "ingest_n": ingest_n,
            })
            print(
                f"[bakeoff] {i:>2}/{len(fixtures)} {case['case_id']:<24s} "
                f"[{case['category']:<14s}] → {verdict}"
            )
        except Exception as e:
            print(f"[bakeoff] {i:>2}/{len(fixtures)} {case['case_id']} FAILED: {e}")
            verdicts.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "verdict": "miss",
                "error": str(e),
            })

    elapsed = time.time() - started

    # Aggregate
    by_verdict = Counter(v["verdict"] for v in verdicts)
    new_wins = by_verdict.get("new_wins", 0)
    total = len(verdicts)
    score_pct = (new_wins / total * 100.0) if total else 0.0

    by_category: dict[str, dict[str, int]] = {}
    for v in verdicts:
        bucket = by_category.setdefault(
            v["category"],
            {"new_wins": 0, "stale_wins": 0, "miss": 0, "ambiguous": 0},
        )
        bucket[v["verdict"]] = bucket.get(v["verdict"], 0) + 1

    print()
    print("=" * 72)
    print(
        f"[bakeoff] DONE — {new_wins}/{total} new_wins ({score_pct:.1f}%) "
        f"in {elapsed:.1f}s"
    )
    print(f"[bakeoff] verdict counts: {dict(by_verdict)}")
    print(f"[bakeoff] per-category:")
    for cat, vals in sorted(by_category.items()):
        print(
            f"  {cat:<16s} new_wins={vals.get('new_wins',0)} "
            f"stale_wins={vals.get('stale_wins',0)} "
            f"miss={vals.get('miss',0)} ambiguous={vals.get('ambiguous',0)}"
        )

    out_path = (
        ROOT / "docs" / "bench"
        / f"pinecone-supersession-llama-{datetime.now().strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "embedder": embedder.provider_name,
        "store": "pinecone-local:attestor-bakeoff-llama",
        "n_cases": total,
        "new_wins": new_wins,
        "stale_wins": by_verdict.get("stale_wins", 0),
        "miss": by_verdict.get("miss", 0),
        "ambiguous": by_verdict.get("ambiguous", 0),
        "score_pct": score_pct,
        "elapsed_sec": elapsed,
        "by_category": by_category,
        "verdicts": verdicts,
    }, indent=2))
    print(f"[bakeoff] persisted → {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    raise SystemExit(main(limit=args.limit))
