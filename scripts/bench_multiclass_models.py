"""Benchmark the multi-class category classifier across multiple models.

For each model, runs the single-call multi-class classifier (one LLM call per
question, predicts one of the 6 categories) over a balanced test pool.
Reports overall accuracy, per-class precision/recall/F1, and a confusion matrix.

This is the multi-class counterpart to bench_binary_matrix.py — same models,
same data, but one LLM call per question instead of six.

Usage:
    python -m scripts.bench_multiclass_models \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --shots-per-class 6 \\
        --max-test 60 \\
        --balanced-test \\
        --output logs/bench_multiclass.json
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

from bench_binary_models import model_available  # noqa: E402
from classify_lme_v2 import OLLAMA_BASE_URL, OPENROUTER_BASE_URL, load_env  # noqa: E402
from lme_prompts import (  # noqa: E402
    CATEGORIES,
    CATEGORY_SET,
    Shot,
    build_multiclass_prompt,
    parse_category_response,
)


DEFAULT_ROSTER: list[dict] = [
    {"name": "ollama:qwen2.5-14b", "model": "qwen2.5:14b-instruct",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 24, "is_reasoning": False},
    {"name": "ollama:phi4-14b", "model": "phi4:14b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 24, "is_reasoning": False},
    {"name": "ollama:mistral-small-24b", "model": "mistral-small:24b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 24, "is_reasoning": False},
    {"name": "ollama:qwq-32b", "model": "qwq:32b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 4096, "is_reasoning": True},
    {"name": "ollama:llama3.2-3b", "model": "llama3.2:latest",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 24, "is_reasoning": False},
]


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    category: str


def load_dataset(path: Path) -> list[Sample]:
    raw = json.loads(path.read_text())
    return [
        Sample(str(r["question_id"]), r["question"], r["question_type"])
        for r in raw if r.get("question_type") in CATEGORY_SET
    ]


def split_few_shot(
    samples: list[Sample], shots_per_class: int, seed: int,
) -> tuple[dict[str, list[Shot]], list[Sample]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[Sample]] = {c: [] for c in CATEGORIES}
    for s in samples:
        by_cat[s.category].append(s)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])
    shots = {c: [Shot(s.question, s.category) for s in by_cat[c][:shots_per_class]]
             for c in CATEGORIES}
    test_pool: list[Sample] = []
    for c in CATEGORIES:
        test_pool.extend(by_cat[c][shots_per_class:])
    rng.shuffle(test_pool)
    return shots, test_pool


def _strip_reasoning(raw: str) -> str:
    """Strip <think>...</think> blocks (reasoning models like QwQ)."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1]
    return raw


def classify_one(
    client: Any, system_prompt: str, question: str, model: str,
    max_tokens: int = 24, is_reasoning: bool = False,
    max_retries: int = 2,
) -> tuple[str, float]:
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nCategory:"},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            if is_reasoning:
                raw = _strip_reasoning(raw)
            return parse_category_response(raw), time.monotonic() - t0
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR:{type(e).__name__}", time.monotonic() - t0
            time.sleep(1.5 * (attempt + 1))


def run_one_model(
    entry: dict, system_prompt: str, test_pool: list[Sample],
    parallel: int, progress_every: int, model_label: str,
) -> dict:
    from openai import OpenAI
    api_key = os.environ.get(entry["api_key_env"]) or "ollama"
    client = OpenAI(api_key=api_key, base_url=entry["base_url"])

    started = time.monotonic()
    results: list[dict] = []
    total = len(test_pool)

    def _one(s: Sample) -> dict:
        pred, latency = classify_one(
            client, system_prompt, s.question, entry["model"],
            max_tokens=entry.get("max_tokens", 24),
        )
        return {
            "question_id": s.question_id,
            "gold_category": s.category,
            "predicted": pred,
            "correct": pred == s.category,
            "latency_s": round(latency, 3),
        }

    processed = 0
    correct_so_far = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_one, s): s for s in test_pool}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            processed += 1
            if r["correct"]:
                correct_so_far += 1
            if processed % progress_every == 0 or processed == total:
                avg_lat = sum(x["latency_s"] for x in results) / len(results)
                print(
                    f"      [{model_label}] {processed}/{total} "
                    f"running_acc={correct_so_far/processed:.3f} "
                    f"avg_lat={avg_lat:.2f}s",
                    flush=True,
                )

    elapsed = time.monotonic() - started

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {
        c: {p: 0 for p in CATEGORIES} for c in CATEGORIES
    }
    unknown = errors = 0
    for r in results:
        gold, pred = r["gold_category"], r["predicted"]
        if pred.startswith("ERROR"):
            errors += 1
            continue
        if pred == "UNKNOWN":
            unknown += 1
            continue
        confusion[gold][pred] += 1

    accuracy = correct_so_far / total if total else 0.0
    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for c in CATEGORIES:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in CATEGORIES if g != c)
        fn = sum(confusion[c][p] for p in CATEGORIES if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_lat = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    return {
        "name": entry["name"],
        "model": entry["model"],
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "unknown": unknown,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
        "avg_latency_s": round(avg_lat, 3),
        "per_class": per_class,
        "confusion": confusion,
    }


def print_confusion(metrics: dict, model_name: str) -> None:
    short = {
        "temporal-reasoning": "temp", "multi-session": "multi",
        "knowledge-update": "k-upd", "single-session-user": "ss-u",
        "single-session-assistant": "ss-a", "single-session-preference": "ss-p",
    }
    print(f"\nConfusion matrix — {model_name}:")
    print(f"  {'gold ↓ / pred →':<28} " + " ".join(f"{short[c]:>5}" for c in CATEGORIES))
    for g in CATEGORIES:
        row = metrics["confusion"][g]
        print(f"  {g:<28} " + " ".join(f"{row[p]:>5}" for p in CATEGORIES))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=60)
    p.add_argument("--balanced-test", action="store_true", default=True)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--only", default=None)
    p.add_argument("--output", default="logs/bench_multiclass.json")
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

    samples = load_dataset(args.data)
    shots, test_pool = split_few_shot(samples, args.shots_per_class, args.seed)
    if args.balanced_test and args.max_test:
        per_class = max(1, args.max_test // len(CATEGORIES))
        by_cat: dict[str, list[Sample]] = {c: [] for c in CATEGORIES}
        for s in test_pool:
            by_cat[s.category].append(s)
        rng = random.Random(args.seed + 1)
        balanced: list[Sample] = []
        for c in CATEGORIES:
            balanced.extend(by_cat[c][:per_class])
        rng.shuffle(balanced)
        test_pool = balanced
    elif args.max_test:
        test_pool = test_pool[: args.max_test]

    print(f"\ntest pool: {len(test_pool)} samples")
    by_cat = {c: 0 for c in CATEGORIES}
    for s in test_pool:
        by_cat[s.category] += 1
    for c in CATEGORIES:
        print(f"  {c:<28} {by_cat[c]}")

    system_prompt = build_multiclass_prompt(shots)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for entry in available:
        print(f"\n▶  {entry['name']} ({entry['model']})", flush=True)
        try:
            row = run_one_model(
                entry, system_prompt, test_pool, args.parallel,
                args.progress_every, entry["name"],
            )
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            row = {"name": entry["name"], "model": entry["model"],
                   "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rows.append(row)
            out_path.write_text(json.dumps({"models": rows}, indent=2))
            continue
        rows.append(row)
        print(f"  → acc={row['accuracy']:.4f} macro_F1={row['macro_f1']:.4f} "
              f"unk={row['unknown']} err={row['errors']} "
              f"avg_lat={row['avg_latency_s']:.2f}s total={row['elapsed_s']}s",
              flush=True)
        out_path.write_text(json.dumps({"models": rows}, indent=2))

    # Final summary
    print()
    print("=" * 96)
    print("MULTI-CLASS CATEGORY CLASSIFIER — FINAL")
    print("=" * 96)
    short = {c: c[:8] for c in CATEGORIES}
    short["temporal-reasoning"] = "temp"
    short["multi-session"] = "multi"
    short["knowledge-update"] = "k-upd"
    short["single-session-user"] = "ss-u"
    short["single-session-assistant"] = "ss-a"
    short["single-session-preference"] = "ss-p"
    header = f"{'model':<28} {'acc':>6} " + " ".join(f"{short[c]:>6}" for c in CATEGORIES) + f"  {'macro':>7}  {'lat/q':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "per_class" not in r:
            print(f"{r['name']:<28} ERROR: {r.get('error', '?')}")
            continue
        cells = " ".join(f"{r['per_class'][c]['f1']:>6.3f}" for c in CATEGORIES)
        print(f"{r['name']:<28} {r['accuracy']:>6.3f} {cells}  {r['macro_f1']:>7.4f}  {r['avg_latency_s']:>6.2f}s")

    for r in rows:
        if "confusion" in r:
            print_confusion(r, r["name"])

    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
