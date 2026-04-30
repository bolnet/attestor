"""Retrieval-only diagnostic on LME-S temporal-reasoning slice.

Identifies questions where vector retrieval FAILS to surface any of
the gold answer-bearing sessions in top-K. Pure embedding + cosine
search — no answerer, no judges. Tells you which questions a
downstream answerer wouldn't even have a chance on, regardless of how
strong the prompt is.

Stack:
  embedder:  Pinecone Inference `llama-text-embed-v2` @ 1024-D (NVIDIA)
  store:     Pinecone Local (Docker, in-memory, free)
  isolation: one Pinecone namespace per LME-S question

Outputs:
  stdout:                          per-question pass/fail
  docs/bench/pinecone-lme-temporal-diagnostic-<date>.json
                                   verdicts + top-5 detail per question

Cost: ~free under Pinecone's 5M-token starter (10 questions of LME-S
oracle/s ≈ 50-450k tokens depending on variant).

USAGE
-----
    set -a && source .env && set +a
    .venv/bin/python experiments/pinecone_lme_temporal_diagnostic.py --limit 10

Per the smoke-only rule (feedback_smoke_only_no_full_bench), Claude
caps at 10 samples. User runs the full 133-question slice on their
own cadence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attestor.longmemeval import load_or_download, parse_lme_date  # noqa: E402
from attestor.retrieval.hyde import hyde_search  # noqa: E402
from attestor.retrieval.multi_query import (  # noqa: E402
    multi_query_search, reciprocal_rank_fusion,
)
from attestor.retrieval.temporal_prefilter import detect_window  # noqa: E402
from attestor.store.embeddings import PineconeEmbeddingProvider  # noqa: E402
from attestor.store.pinecone_backend import PineconeBackend  # noqa: E402


def _ingest_haystack(pc, sample, namespace: str) -> int:
    """Upsert every turn of every haystack session into the per-question
    namespace. Per-turn (unbatched) so the wallclock spreads embedding
    tokens evenly under Pinecone Inference's 250k tokens/min free-tier
    cap. Batched ingest hits HTTP 429.

    Returns count of vectors written.
    """
    n = 0
    for sess_idx, session_turns in enumerate(sample.haystack_sessions):
        sess_id = sample.haystack_session_ids[sess_idx]
        sess_date = (
            sample.haystack_dates[sess_idx]
            if sess_idx < len(sample.haystack_dates) else ""
        )
        for turn_idx, turn in enumerate(session_turns):
            content = turn.content
            if not content.strip():
                continue
            mem_id = f"{sess_id}:t{turn_idx}"
            pc._index.upsert(
                vectors=[{
                    "id": mem_id,
                    "values": pc._embed(content),
                    "metadata": {
                        "session_id": sess_id,
                        "session_date": sess_date,
                        "role": turn.role,
                    },
                }],
                namespace=namespace,
            )
            n += 1
    return n


def _make_vector_search(pc, namespace: str, k: int):
    """Closure: q → list of hit dicts shaped for RRF merge.

    Used by ``multi_query_search`` / ``hyde_search``: each call embeds
    ``q`` and runs a top-K query in the per-question namespace, returns
    dicts keyed on ``memory_id`` (the RRF identity field). Wide top-K
    here gives RRF more material to work with.
    """
    def search(q: str):
        result = pc._index.query(
            vector=pc._embed(q), top_k=k, namespace=namespace,
            include_metadata=True,
        )
        return [
            {
                "memory_id": m.id,
                "session_id": m.metadata.get("session_id"),
                "session_date": m.metadata.get("session_date", ""),
                "distance": max(0.0, 1.0 - float(m.score)),
            }
            for m in (result.matches or [])
        ]
    return search


def _recall_via_multi_query(pc, query: str, namespace: str, k: int, n: int):
    vector_search = _make_vector_search(pc, namespace, k=k * 3)
    queries, merged = multi_query_search(
        query, vector_search=vector_search, n=n, merge="rrf",
    )
    top = merged[:k]
    return [(h["session_id"], h["memory_id"], h["distance"]) for h in top], queries


def _recall_via_hyde(pc, query: str, namespace: str, k: int):
    vector_search = _make_vector_search(pc, namespace, k=k * 3)
    queries, merged = hyde_search(
        query, vector_search=vector_search, merge="rrf",
    )
    top = merged[:k]
    return [(h["session_id"], h["memory_id"], h["distance"]) for h in top], queries


def _bm25_lane(sample, query: str, k: int) -> list[dict]:
    """In-memory BM25 over the sample's haystack turns.

    Returns hits shaped like the Pinecone vector lane so they can be
    RRF-merged with HyDE's merged_hits. BM25 catches rare-noun anchors
    (proper nouns, technical terms, product names) that dense embedders
    smear across visually-similar distractor sessions — the canonical
    cosine-vs-keyword complement.
    """
    from rank_bm25 import BM25Okapi
    docs = []
    meta = []
    for sess_idx, turns in enumerate(sample.haystack_sessions):
        sess_id = sample.haystack_session_ids[sess_idx]
        for turn_idx, turn in enumerate(turns):
            content = turn.content
            if not content.strip():
                continue
            docs.append(content.lower().split())
            meta.append({
                "memory_id": f"{sess_id}:t{turn_idx}",
                "session_id": sess_id,
            })
    if not docs:
        return []
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(query.lower().split())
    # Top-k by score, descending. Convert to vector-lane shape so RRF
    # merge keys match. Fake "distance" = 1 - normalized_score so low
    # is better (consistent with cosine).
    ranked = sorted(zip(scores, meta), key=lambda x: x[0], reverse=True)[:k]
    if not ranked:
        return []
    max_score = max(s for s, _ in ranked) or 1.0
    return [
        {
            "memory_id": m["memory_id"],
            "session_id": m["session_id"],
            "distance": 1.0 - (float(s) / max_score),
            "bm25_score": float(s),
        }
        for s, m in ranked
    ]


def _recall_via_hyde_bm25(pc, sample, namespace: str, k: int):
    """HyDE dense lane + local BM25 lane, RRF-merged.

    Two ranked lists: (1) HyDE-style dense retrieval (original question
    + hypothetical answer, RRF-merged via hyde_search), (2) BM25 over
    the haystack. Final merge via RRF k=60. Top-K of the final ranking
    is returned.
    """
    vector_search = _make_vector_search(pc, namespace, k=k * 3)
    hyde_queries, hyde_merged = hyde_search(
        sample.question, vector_search=vector_search, merge="rrf",
    )
    # Use the question (not the hypothetical) for BM25 — keyword match
    # on the user's actual nouns is what BM25 is best at.
    bm25_hits = _bm25_lane(sample, sample.question, k=k * 3)
    # Final RRF over the two lanes.
    merged = reciprocal_rank_fusion([hyde_merged, bm25_hits])
    top = merged[:k]
    return (
        [(h["session_id"], h["memory_id"], h["distance"]) for h in top],
        hyde_queries,
    )


def _recall_topk(
    pc, query: str, namespace: str, k: int = 10,
    *,
    temporal_prefilter: bool = False,
    question_date_str: str = "",
    tolerance_days: int = 3,
):
    """Return the top-K hits as (session_id, mem_id, distance) tuples.

    When ``temporal_prefilter=True`` and the question contains a
    relative time phrase ("how many weeks ago"), inflate top_k 5×
    to give the post-filter headroom, then drop hits whose
    ``session_date`` falls outside the implied window. This emulates
    what ``attestor/retrieval/orchestrator.py`` does at Step 0 when
    ``retrieval.temporal_prefilter.enabled=True``, except it filters
    AFTER the vector query rather than narrowing the candidate pool
    upstream — which is fine for a diagnostic but slightly weaker
    than the orchestrator's path.
    """
    query_vec = pc._embed(query)

    detected = None
    effective_k = k
    if temporal_prefilter:
        question_date = parse_lme_date(question_date_str) if question_date_str else None
        detected = detect_window(
            query, question_date=question_date, tolerance_days=tolerance_days,
        )
        if detected is not None:
            # Cast a wider net so we have something left after filtering.
            effective_k = k * 5

    result = pc._index.query(
        vector=query_vec, top_k=effective_k, namespace=namespace,
        include_metadata=True,
    )
    raw_hits = [
        (
            m.metadata.get("session_id"),
            m.id,
            max(0.0, 1.0 - float(m.score)),
            m.metadata.get("session_date", ""),
        )
        for m in (result.matches or [])
    ]

    if detected is None:
        # Strip session_date and return original shape.
        return [(s, mid, d) for s, mid, d, _ in raw_hits[:k]]

    # Post-filter by session_date in the detected window.
    lower = detected.window.start
    upper = detected.window.end
    kept: List[Tuple] = []
    for sid, mid, dist, sess_date_str in raw_hits:
        sess_dt = parse_lme_date(sess_date_str) if sess_date_str else None
        if sess_dt is None:
            # No date → keep, can't filter
            kept.append((sid, mid, dist))
            continue
        if (lower is None or sess_dt >= lower) and (upper is None or sess_dt <= upper):
            kept.append((sid, mid, dist))
        if len(kept) >= k:
            break

    # Defensive: if the filter dropped everything, fall back to the
    # un-filtered top-K so we never return empty when the question
    # had a temporal phrase but no haystack session is in-window.
    if not kept:
        return [(s, mid, d) for s, mid, d, _ in raw_hits[:k]]
    return kept


def _classify(top_hits, gold_session_ids: tuple, k: int) -> str:
    """Did ANY gold session appear in top-K?

    Returns:
        'recall@1'   gold session is the top-1 hit
        'recall@K'   gold session is in top-K (but not top-1)
        'miss'       no gold session in top-K (retrieval can't help here)
        'no_hits'    Pinecone returned nothing at all
    """
    if not top_hits:
        return "no_hits"
    if not gold_session_ids:
        return "no_gold"
    gold = set(gold_session_ids)
    if top_hits[0][0] in gold:
        return "recall@1"
    if any(s in gold for s, _, _ in top_hits):
        return "recall@K"
    return "miss"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="s",
                   help="LME variant: oracle | s | m. Default s.")
    p.add_argument("--category", default="temporal-reasoning",
                   help=("LME question_type filter. Default "
                         "'temporal-reasoning'. Other slices: "
                         "'multi-session', 'knowledge-update', "
                         "'single-session-user', 'single-session-assistant', "
                         "'single-session-preference'."))
    p.add_argument("--limit", type=int, default=10,
                   help="Max questions to score. Default 10 (smoke).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Top-K depth for the recall check. Default 10.")
    p.add_argument("--skip-ingest", action="store_true",
                   help=("Skip the haystack ingest step (assume the "
                         "namespaces are already populated from a prior "
                         "run). Cuts wallclock from ~17 min to ~30s for "
                         "10 questions when iterating retrieval flags."))
    p.add_argument("--temporal-prefilter", action="store_true",
                   help=("Apply RC4 temporal pre-filter (PR #98): detect "
                         "'N weeks ago' / 'last Monday' / etc. in the "
                         "question and drop top-K hits whose session_date "
                         "falls outside the implied event-time window."))
    p.add_argument("--tolerance-days", type=int, default=3,
                   help="Half-width of the temporal window in days.")
    p.add_argument("--multi-query", action="store_true",
                   help=("Apply PR #94 multi-query: rewrite question into "
                         "N paraphrases via OpenRouter, run each as a "
                         "vector lane, RRF-merge."))
    p.add_argument("--multi-query-n", type=int, default=3,
                   help="Number of paraphrases (default 3).")
    p.add_argument("--hyde", action="store_true",
                   help=("Apply PR-D HyDE: generate a hypothetical answer "
                         "via OpenRouter, run both question + hypothetical "
                         "as vector lanes, RRF-merge."))
    p.add_argument("--bm25-hybrid", action="store_true",
                   help=("Add a local BM25 lane (rank_bm25) over the "
                         "haystack and RRF-merge with HyDE. Catches "
                         "rare-noun anchors (proper nouns, products) "
                         "that dense cosine smears."))
    args = p.parse_args(argv)

    print(f"[diag] loading LME-{args.variant!r}…")
    samples = load_or_download(variant=args.variant)
    temporal = [s for s in samples if s.question_type == args.category]
    print(f"[diag] {len(temporal)} {args.category} questions in variant={args.variant!r}")
    if args.limit:
        temporal = temporal[: args.limit]
    print(f"[diag] running {len(temporal)} (smoke cap)")

    embedder = PineconeEmbeddingProvider(
        model="llama-text-embed-v2", dimensions=1024,
    )
    pc = PineconeBackend(config={
        "index_name": f"attestor-lme-{args.category[:30]}",
        "dimension": 1024,
        "embedder": embedder,
    })
    print(f"[diag] embedder={embedder.provider_name}  store=pinecone-local")
    print()

    started = time.time()
    verdicts = []
    for i, sample in enumerate(temporal, 1):
        ns = sample.question_id
        try:
            if args.skip_ingest:
                # Assume the namespace is already populated from a prior
                # run — saves the ~95% of wallclock that ingest costs.
                ingest_n = -1
            else:
                ingest_n = _ingest_haystack(pc, sample, namespace=ns)
                time.sleep(0.3)
            queries_used = None
            if args.bm25_hybrid:
                top_hits, queries_used = _recall_via_hyde_bm25(
                    pc, sample, namespace=ns, k=args.top_k,
                )
            elif args.multi_query:
                top_hits, queries_used = _recall_via_multi_query(
                    pc, sample.question, namespace=ns, k=args.top_k,
                    n=args.multi_query_n,
                )
            elif args.hyde:
                top_hits, queries_used = _recall_via_hyde(
                    pc, sample.question, namespace=ns, k=args.top_k,
                )
            else:
                top_hits = _recall_topk(
                    pc, sample.question, namespace=ns, k=args.top_k,
                    temporal_prefilter=args.temporal_prefilter,
                    question_date_str=sample.question_date,
                    tolerance_days=args.tolerance_days,
                )
            verdict = _classify(top_hits, sample.answer_session_ids, args.top_k)

            row = {
                "question_id": sample.question_id,
                "question": sample.question,
                "gold_session_ids": list(sample.answer_session_ids),
                "verdict": verdict,
                "ingest_n": ingest_n,
                "top_5": [
                    {"session_id": s, "mem_id": m, "distance": round(d, 4)}
                    for s, m, d in top_hits[:5]
                ],
            }
            if queries_used is not None:
                row["queries_used"] = queries_used
            verdicts.append(row)
            print(
                f"[diag] {i:>2}/{len(temporal)} {sample.question_id:<30s} "
                f"ingest={ingest_n:>3}  → {verdict}"
            )
        except Exception as e:
            verdicts.append({
                "question_id": sample.question_id,
                "verdict": "error",
                "error": str(e),
            })
            print(f"[diag] {i:>2}/{len(temporal)} {sample.question_id} ERROR: {e}")

    elapsed = time.time() - started
    counts = Counter(v["verdict"] for v in verdicts)
    total = len(verdicts)
    recall_1 = counts.get("recall@1", 0)
    recall_k = counts.get("recall@K", 0)
    miss = counts.get("miss", 0)
    found_pct = ((recall_1 + recall_k) / total * 100.0) if total else 0.0

    print()
    print("=" * 72)
    print(
        f"[diag] DONE — {recall_1 + recall_k}/{total} found ({found_pct:.1f}%) "
        f"in {elapsed:.1f}s  "
        f"[recall@1: {recall_1}, recall@K: {recall_k}, miss: {miss}]"
    )

    # FAILURES — what the user came for
    failures = [v for v in verdicts if v["verdict"] in ("miss", "no_hits", "error")]
    if failures:
        print()
        print(f"FAILURES ({len(failures)}/{total}) — questions where retrieval can't help any answerer:")
        print()
        for v in failures:
            print(f"  • {v['question_id']}: \"{v.get('question','')[:80]}\"")
            print(f"    gold sessions: {v.get('gold_session_ids')}")
            for hit in v.get("top_5", [])[:3]:
                print(f"    top: session={hit['session_id']}  dist={hit['distance']}")
            if v.get("error"):
                print(f"    error: {v['error']}")

    mode = "baseline"
    if args.bm25_hybrid:
        mode = "hyde-bm25"
    elif args.multi_query:
        mode = f"mq{args.multi_query_n}"
    elif args.hyde:
        mode = "hyde"
    elif args.temporal_prefilter:
        mode = "tprefilter"
    out_path = (
        ROOT / "docs" / "bench"
        / f"pinecone-lme-{args.category}-diagnostic-{mode}-{datetime.now().strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "embedder": embedder.provider_name,
        "store": "pinecone-local",
        "variant": args.variant,
        "category": args.category,
        "n_total": len(temporal),
        "n_scored": total,
        "top_k": args.top_k,
        "counts": dict(counts),
        "found_pct": found_pct,
        "failures": failures,
        "verdicts": verdicts,
        "elapsed_sec": elapsed,
    }, indent=2))
    print(f"\n[diag] persisted → {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
