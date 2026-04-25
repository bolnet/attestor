"""Matrix benchmark: every model × every binary classifier, in isolation.

For each model, runs all 6 binary classifiers (one per category) sequentially
without re-loading the model between targets. Each binary call returns YES/NO
(treated as 1/0) and is scored against the gold category as a binary problem.

Output: an F1 matrix (rows = models, cols = categories) plus per-cell P/R
and an aggregate macro-F1 per model.

Usage:
    python -m scripts.bench_binary_matrix \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --max-test 60 \\
        --shots-per-class 6 \\
        --output logs/bench_binary_matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_binary_models import (  # noqa: E402
    DEFAULT_ROSTER,
    Sample,
    classify_one,
    load_dataset,
    model_available,
    split_few_shot,
)
from classify_lme_v2 import load_env  # noqa: E402
from lme_prompts import CATEGORIES, build_binary_prompt  # noqa: E402


def evaluate_one_classifier(
    client: Any, entry: dict, target: str,
    pos_shots, neg_shots, test_pool, parallel: int, verbose: bool,
    progress_every: int = 25, model_label: str = "",
) -> dict:
    """Run one binary classifier (target=X) over the test pool. Returns
    metrics dict (no per-question results to keep memory in check).

    Prints a progress line every ``progress_every`` questions so long-running
    benches show signs of life."""
    system_prompt = build_binary_prompt(target, pos_shots, neg_shots)
    max_tokens = entry.get("max_tokens", 16)
    is_reasoning = entry.get("is_reasoning", False)
    total = len(test_pool)

    def _one(s: Sample) -> tuple[str, str, float]:
        pred, latency = classify_one(
            client, system_prompt, s.question, entry["model"],
            max_tokens=max_tokens, is_reasoning=is_reasoning,
        )
        gold = "YES" if s.is_target else "NO"
        return pred, gold, latency

    started = time.monotonic()
    tp = fp = tn = fn = unknown = errors = 0
    latencies: list[float] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_one, s): s for s in test_pool}
        for fut in as_completed(futures):
            pred, gold, latency = fut.result()
            latencies.append(latency)
            processed += 1
            if pred.startswith("ERROR"):
                errors += 1
                if gold == "YES":
                    fn += 1
                else:
                    fp += 1
            elif pred == "UNKNOWN":
                unknown += 1
                if gold == "YES":
                    fn += 1
                else:
                    fp += 1
            elif pred == "YES" and gold == "YES":
                tp += 1
            elif pred == "YES" and gold == "NO":
                fp += 1
            elif pred == "NO" and gold == "NO":
                tn += 1
            else:
                fn += 1
            if processed % progress_every == 0 or processed == total:
                avg_lat = sum(latencies) / len(latencies)
                est_remaining = (total - processed) * avg_lat / max(1, parallel)
                print(
                    f"      [{model_label} {target}] {processed}/{total} "
                    f"avg_lat={avg_lat:.2f}s eta_rem={est_remaining:.0f}s "
                    f"running: tp={tp} fp={fp} tn={tn} fn={fn} unk={unknown} err={errors}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "target": target,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "unknown": unknown, "errors": errors,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_s": round(avg_lat, 3),
        "elapsed_s": round(elapsed, 1),
    }


def run_model_across_all_targets(
    entry: dict, samples_by_target: dict[str, dict],
    parallel: int, verbose: bool,
    checkpoint_path: Path | None = None,
    all_rows: list[dict] | None = None,
) -> dict:
    """For one model, run all 6 binary classifiers and aggregate.

    If ``checkpoint_path`` and ``all_rows`` are provided, writes a checkpoint
    JSON after each target completes — preserves partial progress on interrupt.
    """
    from openai import OpenAI
    api_key = os.environ.get(entry["api_key_env"]) or "ollama"
    client = OpenAI(api_key=api_key, base_url=entry["base_url"])

    started = time.monotonic()
    per_target: list[dict] = []
    row = {
        "name": entry["name"],
        "model": entry["model"],
        "status": "in_progress",
        "per_target": per_target,
    }
    if all_rows is not None:
        all_rows.append(row)

    for target in CATEGORIES:
        bundle = samples_by_target[target]
        print(f"    [{entry['name']}] target={target} pool={len(bundle['test_pool'])}",
              flush=True)
        result = evaluate_one_classifier(
            client, entry, target,
            bundle["pos_shots"], bundle["neg_shots"], bundle["test_pool"],
            parallel, verbose,
            model_label=entry["name"],
        )
        per_target.append(result)
        print(f"      → done: acc={result['accuracy']:.3f} P={result['precision']:.3f} "
              f"R={result['recall']:.3f} F1={result['f1']:.3f} "
              f"unk={result['unknown']} err={result['errors']} "
              f"lat={result['avg_latency_s']:.2f}s/q total={result['elapsed_s']}s",
              flush=True)
        # Checkpoint after every target so a kill at 90% doesn't lose everything.
        if checkpoint_path is not None and all_rows is not None:
            row["macro_f1"] = round(
                sum(r["f1"] for r in per_target) / len(per_target), 4
            )
            row["total_elapsed_s"] = round(time.monotonic() - started, 1)
            checkpoint_path.write_text(json.dumps({"models": all_rows}, indent=2))

    row["status"] = "ok"
    row["macro_f1"] = round(
        sum(r["f1"] for r in per_target) / len(per_target) if per_target else 0.0, 4
    )
    row["total_elapsed_s"] = round(time.monotonic() - started, 1)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=60,
                   help="Per-target test pool size (balanced)")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of model names")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default="logs/bench_binary_matrix.json")
    args = p.parse_args()

    load_env(Path(args.env_file))

    roster = list(DEFAULT_ROSTER)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        roster = [r for r in roster if r["name"] in wanted]

    print("Model availability:")
    available: list[dict] = []
    for entry in roster:
        ok, msg = model_available(entry)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {entry['name']:<32}  {msg}", flush=True)
        if ok:
            available.append(entry)
    if not available:
        return 2

    # Build per-target shots + test pools ONCE so every model sees identical inputs.
    print(f"\nBuilding per-target sample bundles (shots+test) ...", flush=True)
    samples_by_target: dict[str, dict] = {}
    for target in CATEGORIES:
        all_samples = load_dataset(args.data, target)
        pos_shots, neg_shots, test_pool = split_few_shot(
            all_samples, args.shots_per_class, args.seed
        )
        # Balanced cap
        per_class = args.max_test // 2
        pos_test = [s for s in test_pool if s.is_target][:per_class]
        neg_test = [s for s in test_pool if not s.is_target][:per_class]
        rng = random.Random(args.seed + 1)
        balanced = pos_test + neg_test
        rng.shuffle(balanced)
        samples_by_target[target] = {
            "pos_shots": pos_shots,
            "neg_shots": neg_shots,
            "test_pool": balanced,
        }
        print(f"  {target:<28} pool={len(balanced)} ({len(pos_test)} pos / {len(neg_test)} neg)")

    print()
    rows: list[dict] = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for entry in available:
        print(f"\n▶  {entry['name']} ({entry['model']})", flush=True)
        try:
            row = run_model_across_all_targets(
                entry, samples_by_target, args.parallel, args.verbose,
                checkpoint_path=out_path, all_rows=rows,
            )
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            # If we already appended an in-progress row, mark it failed.
            if rows and rows[-1].get("name") == entry["name"]:
                rows[-1]["status"] = "error"
                rows[-1]["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            else:
                rows.append({"name": entry["name"], "model": entry["model"],
                             "status": "error",
                             "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        print(f"  → macro F1 = {row['macro_f1']:.4f}  total = {row['total_elapsed_s']}s",
              flush=True)

    # ── F1 matrix ──
    print()
    print("=" * 100)
    print("BINARY CLASSIFIER F1 MATRIX (rows = models, cols = target categories)")
    print("=" * 100)
    short = {
        "temporal-reasoning": "temp",
        "multi-session": "multi",
        "knowledge-update": "k-upd",
        "single-session-user": "ss-u",
        "single-session-assistant": "ss-a",
        "single-session-preference": "ss-p",
    }
    header = f"{'model':<28} " + " ".join(f"{short[c]:>6}" for c in CATEGORIES) + f"  {'macro':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "per_target" not in r:
            continue
        cells = []
        for c in CATEGORIES:
            t = next((x for x in r["per_target"] if x["target"] == c), None)
            cells.append(f"{t['f1']:>6.3f}" if t else f"{'-':>6}")
        print(f"{r['name']:<28} " + " ".join(cells) + f"  {r['macro_f1']:>7.4f}")

    # Errors at bottom.
    for r in rows:
        if "error" in r:
            print(f"  {r['name']}: ERROR — {r['error']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "shots_per_class": args.shots_per_class,
            "max_test_per_target": args.max_test,
            "seed": args.seed,
        },
        "models": rows,
    }, indent=2))
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
