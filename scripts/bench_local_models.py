"""Benchmark multiple LLM backends against the v2 classifier task.

For each model, run the parallel-binary + conflict-resolution pipeline on the
same test pool with the same shots/seed. Print a side-by-side comparison.

Designed for local Ollama models, but works with any OpenAI-compatible
endpoint (including OpenRouter for a hosted-vs-local comparison).

Usage:
    python -m scripts.bench_local_models \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --max-test 60 \\
        --shots-per-class 6 \\
        --output logs/bench_models.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add scripts/ to path so we can import siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_lme_v2 import (  # noqa: E402
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    Sample,
    build_binary_prompt,
    build_multiclass_prompt,
    evaluate,
    load_dataset,
    load_env,
    per_class_metrics,
    split_for_binaries,
)
from lme_prompts import CATEGORIES  # noqa: E402


# Model roster. Each entry is (display_name, model_id, base_url, api_key_env).
# Ollama-served models use the local endpoint; OpenRouter models use the
# remote endpoint and need an API key.
DEFAULT_ROSTER: list[dict] = [
    {
        "name": "ollama:qwen2.5-14b",
        "model": "qwen2.5:14b-instruct",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": "OLLAMA_KEY",
    },
    {
        "name": "ollama:qwq-32b",
        "model": "qwq:32b",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": "OLLAMA_KEY",
    },
    {
        "name": "ollama:phi4-14b",
        "model": "phi4:14b",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": "OLLAMA_KEY",
    },
    {
        "name": "ollama:mistral-small-24b",
        "model": "mistral-small:24b",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": "OLLAMA_KEY",
    },
    {
        "name": "ollama:llama3.2-3b",
        "model": "llama3.2:latest",
        "base_url": OLLAMA_BASE_URL,
        "api_key_env": "OLLAMA_KEY",
    },
]


def model_available(entry: dict) -> tuple[bool, str]:
    """Best-effort availability check. For Ollama, list local models. For
    remote endpoints, just check that the API key env is set."""
    if "localhost" in entry["base_url"] or "127.0.0.1" in entry["base_url"]:
        try:
            import subprocess
            res = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10
            )
            if res.returncode != 0:
                return False, f"ollama list failed: {res.stderr.strip()[:80]}"
            tag = entry["model"]
            # Tolerate "qwen2.5:14b-instruct" vs "qwen2.5:14b" suffix matches
            base = tag.split(":")[0]
            for line in res.stdout.splitlines()[1:]:
                if not line.strip():
                    continue
                local_tag = line.split()[0]
                if local_tag == tag or local_tag.startswith(base + ":"):
                    return True, f"available as {local_tag}"
            return False, f"{tag} not in `ollama list`"
        except FileNotFoundError:
            return False, "ollama binary not found"
        except Exception as e:
            return False, f"check failed: {type(e).__name__}: {e}"
    else:
        if not os.environ.get(entry["api_key_env"]):
            return False, f"{entry['api_key_env']} not set"
        return True, "remote — API key present"


def run_one_model(
    entry: dict,
    test_pool: list,
    binary_prompts: dict,
    multiclass_prompt: str,
    parallel_questions: int,
    binary_inner_parallel: int,
    verbose: bool,
) -> dict:
    """Run the v2 classifier for one model entry. Returns metrics dict + meta."""
    api_key = os.environ.get(entry["api_key_env"]) or "ollama"
    started = time.monotonic()
    try:
        results = evaluate(
            test_pool,
            binary_prompts,
            multiclass_prompt,
            entry["model"],          # binary model
            entry["model"],          # arbiter model (same)
            entry["model"],          # fallback model (same — keep test self-contained)
            api_key,
            entry["base_url"],
            parallel_questions=parallel_questions,
            binary_inner_parallel=binary_inner_parallel,
            verbose=verbose,
        )
    except Exception as e:
        return {
            "name": entry["name"],
            "model": entry["model"],
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": round(time.monotonic() - started, 1),
        }

    elapsed = time.monotonic() - started
    metrics = per_class_metrics(results)
    return {
        "name": entry["name"],
        "model": entry["model"],
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "decision_hist": metrics["decision_hist"],
        "yes_count_hist": metrics["yes_count_hist"],
        "per_class": metrics["per_class"],
        "confusion": metrics["confusion"],
        "results": results,
    }


def print_comparison(rows: list[dict]) -> None:
    print()
    print("=" * 90)
    print("BENCHMARK COMPARISON")
    print("=" * 90)
    header = f"{'model':<32} {'status':<8} {'acc':>7} {'macro F1':>9} {'elapsed':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['name']:<32} {'ok':<8} {r['accuracy']:>7.4f} {r['macro_f1']:>9.4f} {r['elapsed_s']:>9.1f}s")
        else:
            print(f"{r['name']:<32} {'ERROR':<8} {'-':>7} {'-':>9} {r['elapsed_s']:>9.1f}s")
            print(f"{'  ':<32} {r['error']}")

    print()
    print("Per-class F1 (rows = model, cols = category):")
    short = {
        "temporal-reasoning": "temp", "multi-session": "multi",
        "knowledge-update": "k-upd", "single-session-user": "ss-u",
        "single-session-assistant": "ss-a", "single-session-preference": "ss-p",
    }
    print(f"  {'model':<32} " + " ".join(f"{short[c]:>6}" for c in CATEGORIES))
    for r in rows:
        if r["status"] == "ok":
            row = " ".join(f"{r['per_class'][c]['f1']:>6.3f}" for c in CATEGORIES)
            print(f"  {r['name']:<32} {row}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=60)
    p.add_argument("--balanced-test", action="store_true", default=True)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel-questions", type=int, default=2,
                   help="Local models do best with low parallelism (default 2)")
    p.add_argument("--binary-inner-parallel", type=int, default=3,
                   help="Inner parallelism per question (default 3)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default="logs/bench_models.json")
    p.add_argument("--include-openrouter", action="store_true",
                   help="Include OpenRouter gpt-4.1-mini in the bench")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of model names to run")
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
            print(f"ERROR: --only filter matched no entries", file=sys.stderr)
            return 2

    # Availability gate.
    print("Model availability:")
    available: list[dict] = []
    for entry in roster:
        ok, msg = model_available(entry)
        marker = "✓" if ok else "✗"
        print(f"  {marker} {entry['name']:<32}  {msg}")
        if ok:
            available.append(entry)
    if not available:
        print("\nNo models available — pull at least one model and retry.", file=sys.stderr)
        return 2

    # Build the shared shots + test pool ONCE so every model sees identical inputs.
    samples = load_dataset(args.data)
    pos, neg, mc, test_pool = split_for_binaries(samples, args.shots_per_class, args.seed)
    if args.balanced_test and args.max_test:
        per_class = max(1, args.max_test // len(CATEGORIES))
        by_cat = {c: [] for c in CATEGORIES}
        for s in test_pool:
            by_cat[s.category].append(s)
        balanced = []
        for c in CATEGORIES:
            balanced.extend(by_cat[c][:per_class])
        test_pool = balanced
    elif args.max_test:
        test_pool = test_pool[: args.max_test]

    binary_prompts = {c: build_binary_prompt(c, pos[c], neg[c]) for c in CATEGORIES}
    multiclass_prompt = build_multiclass_prompt(mc)

    print(f"\nTest pool: {len(test_pool)} samples")
    by_cat_count = {c: 0 for c in CATEGORIES}
    for s in test_pool:
        by_cat_count[s.category] += 1
    for c in CATEGORIES:
        print(f"  {c:<28} {by_cat_count[c]}")
    print(f"Outer parallelism: {args.parallel_questions}, inner: {args.binary_inner_parallel}")
    print()

    rows: list[dict] = []
    for entry in available:
        print(f"\n{'─' * 60}")
        print(f"▶  Running {entry['name']} ({entry['model']})")
        print(f"{'─' * 60}")
        result = run_one_model(
            entry, test_pool, binary_prompts, multiclass_prompt,
            args.parallel_questions, args.binary_inner_parallel,
            args.verbose,
        )
        rows.append(result)
        if result["status"] == "ok":
            print(f"  ✓ accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  elapsed={result['elapsed_s']}s")
        else:
            print(f"  ✗ ERROR: {result['error']}  elapsed={result['elapsed_s']}s")

    print_comparison(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "shots_per_class": args.shots_per_class,
            "test_pool_size": len(test_pool),
            "balanced_test": args.balanced_test,
            "seed": args.seed,
        },
        "models": rows,
    }, indent=2))
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
