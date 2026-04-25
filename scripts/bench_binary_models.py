"""Benchmark multiple models on the BINARY classifier task only.

Strips out v2's arbiter and multi-class fallback so the comparison is purely
"how good is each model at deciding YES/NO for one target category?"

For each model: same shots, same test pool, same target category. Reports
accuracy, precision, recall, F1, and per-question latency.

Usage:
    python -m scripts.bench_binary_models \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --target temporal-reasoning \\
        --max-test 60 \\
        --shots-per-class 6 \\
        --output logs/bench_binary.json
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

from classify_lme_v2 import OLLAMA_BASE_URL, OPENROUTER_BASE_URL, load_env  # noqa: E402
from lme_prompts import (  # noqa: E402
    CATEGORIES,
    CATEGORY_SET,
    Shot,
    build_binary_prompt,
    parse_binary_response,
)


DEFAULT_ROSTER: list[dict] = [
    {"name": "ollama:qwen2.5-14b", "model": "qwen2.5:14b-instruct",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 16, "is_reasoning": False},
    {"name": "ollama:phi4-14b", "model": "phi4:14b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 16, "is_reasoning": False},
    {"name": "ollama:mistral-small-24b", "model": "mistral-small:24b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 16, "is_reasoning": False},
    {"name": "ollama:qwq-32b", "model": "qwq:32b",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 4096, "is_reasoning": True},
    {"name": "ollama:llama3.2-3b", "model": "llama3.2:latest",
     "base_url": OLLAMA_BASE_URL, "api_key_env": "OLLAMA_KEY",
     "max_tokens": 16, "is_reasoning": False},
]


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    category: str
    is_target: bool


def load_dataset(path: Path, target: str) -> list[Sample]:
    raw = json.loads(path.read_text())
    out = []
    for r in raw:
        cat = r.get("question_type")
        if cat in CATEGORY_SET:
            out.append(Sample(
                question_id=str(r["question_id"]),
                question=r["question"],
                category=cat,
                is_target=(cat == target),
            ))
    return out


def split_few_shot(
    samples: list[Sample], shots_per_class: int, seed: int,
) -> tuple[list[Shot], list[Shot], list[Sample]]:
    rng = random.Random(seed)
    pos = [s for s in samples if s.is_target]
    neg = [s for s in samples if not s.is_target]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_shots = [Shot(s.question, s.category) for s in pos[:shots_per_class]]
    neg_shots = [Shot(s.question, s.category) for s in neg[:shots_per_class]]
    used = {s.question_id for s in pos[:shots_per_class] + neg[:shots_per_class]}
    test = [s for s in samples if s.question_id not in used]
    rng.shuffle(test)
    return pos_shots, neg_shots, test


def model_available(entry: dict) -> tuple[bool, str]:
    if "localhost" in entry["base_url"] or "127.0.0.1" in entry["base_url"]:
        try:
            import subprocess
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            tag = entry["model"]
            base = tag.split(":")[0]
            for line in res.stdout.splitlines()[1:]:
                if not line.strip():
                    continue
                local = line.split()[0]
                if local == tag or local.startswith(base + ":"):
                    return True, f"local: {local}"
            return False, f"{tag} not pulled"
        except Exception as e:
            return False, f"check failed: {e}"
    if not os.environ.get(entry["api_key_env"]):
        return False, f"{entry['api_key_env']} not set"
    return True, "remote"


def _strip_reasoning(raw: str) -> str:
    """Strip <think>...</think> blocks from reasoning-model output. Returns
    whatever follows the last </think> tag, or the original string."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1]
    return raw


def _parse_with_reasoning(raw: str) -> str:
    """Parse YES/NO from output that may contain a reasoning preamble.

    1. Strip any <think>...</think> block.
    2. Try the standard parser at the start.
    3. Fall back to scanning the LAST line for YES/NO."""
    after = _strip_reasoning(raw)
    primary = parse_binary_response(after)
    if primary != "UNKNOWN":
        return primary
    # Last-line scan — reasoning models often write "Final answer: YES" at end.
    for line in reversed(after.strip().splitlines()):
        line_clean = line.strip().strip("`*\"'.,").upper()
        if not line_clean:
            continue
        if "YES" in line_clean.split():
            return "YES"
        if "NO" in line_clean.split():
            return "NO"
    return "UNKNOWN"


def classify_one(
    client: Any, system_prompt: str, question: str, model: str,
    max_tokens: int = 16, is_reasoning: bool = False,
    max_retries: int = 2,
) -> tuple[str, float]:
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nAnswer:"},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            if is_reasoning:
                return _parse_with_reasoning(raw), time.monotonic() - t0
            return parse_binary_response(raw), time.monotonic() - t0
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR:{type(e).__name__}", time.monotonic() - t0
            time.sleep(1.5 * (attempt + 1))


def run_one_model(
    entry: dict, test_pool: list[Sample], system_prompt: str,
    parallel: int, verbose: bool,
) -> dict:
    from openai import OpenAI
    api_key = os.environ.get(entry["api_key_env"]) or "ollama"
    client = OpenAI(api_key=api_key, base_url=entry["base_url"])

    started = time.monotonic()
    results: list[dict] = []

    max_tokens = entry.get("max_tokens", 16)
    is_reasoning = entry.get("is_reasoning", False)

    def _one(s: Sample) -> dict:
        pred, latency = classify_one(
            client, system_prompt, s.question, entry["model"],
            max_tokens=max_tokens, is_reasoning=is_reasoning,
        )
        gold = "YES" if s.is_target else "NO"
        return {
            "question_id": s.question_id,
            "question": s.question,
            "gold_category": s.category,
            "gold_label": gold,
            "predicted": pred,
            "correct": pred == gold,
            "latency_s": round(latency, 3),
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_one, s): s for s in test_pool}
        completed = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            completed += 1
            if verbose and completed % 10 == 0:
                print(f"    [{completed}/{len(test_pool)}]", flush=True)

    elapsed = time.monotonic() - started

    tp = fp = tn = fn = unknown = errors = 0
    for r in results:
        gold, pred = r["gold_label"], r["predicted"]
        if pred.startswith("ERROR"):
            errors += 1
            (fn if gold == "YES" else fp).__iadd__  # type: ignore  # noqa
            if gold == "YES":
                fn += 1
            else:
                fp += 1
            continue
        if pred == "UNKNOWN":
            unknown += 1
            if gold == "YES":
                fn += 1
            else:
                fp += 1
            continue
        if pred == "YES" and gold == "YES":
            tp += 1
        elif pred == "YES" and gold == "NO":
            fp += 1
        elif pred == "NO" and gold == "NO":
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_latency = (sum(r["latency_s"] for r in results) / len(results)) if results else 0.0

    return {
        "name": entry["name"],
        "model": entry["model"],
        "elapsed_s": round(elapsed, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "unknown": unknown, "errors": errors,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_s": round(avg_latency, 3),
        "results": results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--target", required=True, choices=sorted(CATEGORY_SET))
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=60)
    p.add_argument("--balanced-test", action="store_true", default=True)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=2,
                   help="Concurrent calls per model (low for local models)")
    p.add_argument("--include-openrouter", action="store_true",
                   help="Add openai/gpt-4.1-mini as a baseline")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of model names")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default="logs/bench_binary.json")
    args = p.parse_args()

    load_env(Path(args.env_file))

    roster = list(DEFAULT_ROSTER)
    if args.include_openrouter:
        roster.append({
            "name": "openrouter:gpt-4.1-mini",
            "model": "openai/gpt-4.1-mini",
            "base_url": OPENROUTER_BASE_URL,
            "api_key_env": "OPENROUTER_API_KEY",
        })
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        roster = [r for r in roster if r["name"] in wanted]
        if not roster:
            print("ERROR: --only matched nothing", file=sys.stderr)
            return 2

    print(f"target: {args.target}")
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

    samples = load_dataset(args.data, args.target)
    pos_shots, neg_shots, test_pool = split_few_shot(
        samples, args.shots_per_class, args.seed
    )
    if args.balanced_test and args.max_test:
        per_class = args.max_test // 2
        pos_test = [s for s in test_pool if s.is_target][:per_class]
        neg_test = [s for s in test_pool if not s.is_target][:per_class]
        rng = random.Random(args.seed + 1)
        test_pool = pos_test + neg_test
        rng.shuffle(test_pool)
    elif args.max_test:
        test_pool = test_pool[: args.max_test]

    n_pos = sum(1 for s in test_pool if s.is_target)
    print(f"\ntest pool: {len(test_pool)} ({n_pos} positive, {len(test_pool) - n_pos} negative)")
    print(f"shots: {len(pos_shots)} positive + {len(neg_shots)} negative\n")

    system_prompt = build_binary_prompt(args.target, pos_shots, neg_shots)

    rows: list[dict] = []
    for entry in available:
        print(f"\n▶  {entry['name']} ({entry['model']})", flush=True)
        try:
            row = run_one_model(entry, test_pool, system_prompt, args.parallel, args.verbose)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            row = {"name": entry["name"], "model": entry["model"],
                   "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rows.append(row)
            continue
        rows.append(row)
        print(f"  acc={row['accuracy']:.3f}  P={row['precision']:.3f}  "
              f"R={row['recall']:.3f}  F1={row['f1']:.3f}  "
              f"lat={row['avg_latency_s']:.2f}s/q  total={row['elapsed_s']}s",
              flush=True)

    print()
    print("=" * 96)
    print(f"BINARY CLASSIFIER BENCHMARK — target: {args.target}")
    print("=" * 96)
    header = f"{'model':<32} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'lat/q':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    # Sort by F1 desc.
    ok_rows = [r for r in rows if "f1" in r]
    ok_rows.sort(key=lambda r: r["f1"], reverse=True)
    for r in ok_rows:
        print(f"{r['name']:<32} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['avg_latency_s']:>7.2f}s {r['elapsed_s']:>7.1f}s")
    err_rows = [r for r in rows if "error" in r]
    for r in err_rows:
        print(f"{r['name']:<32} ERROR: {r['error']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "target": args.target,
            "shots_per_class": args.shots_per_class,
            "test_pool_size": len(test_pool),
            "n_positive": n_pos,
            "seed": args.seed,
        },
        "models": rows,
    }, indent=2))
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
