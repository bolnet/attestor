"""Parallel-binary + conflict-resolution LongMemEval category classifier.

For each question:
  1. Run all 6 binary classifiers in parallel (independent YES/NO).
  2. Collect YES verdicts.
  3. If exactly 1 YES → that's the prediction.
  4. If ≥2 YES   → arbiter LLM picks the best fit from those candidates.
  5. If 0 YES   → fallback to the multi-class classifier (one more call).

Designed to compare against ``classify_lme_categories.py`` (single binary)
and ``classify_lme_cascade.py`` (sequential binaries, first-YES wins). All
prompts come from ``scripts/lme_prompts.py`` and are unit-tested separately.

Usage:
    python -m scripts.classify_lme_v2 \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --shots-per-class 6 \\
        --max-test 240 \\
        --balanced-test \\
        --binary-model openai/gpt-5.1 \\
        --arbiter-model openai/gpt-5.1 \\
        --fallback-model openai/gpt-4.1-mini \\
        --env-file .env \\
        --output logs/classifier_v2.json
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

# Add scripts/ to path so we can import lme_prompts as a sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lme_prompts import (  # noqa: E402
    CATEGORIES,
    CATEGORY_SET,
    Shot,
    build_binary_prompt,
    build_conflict_resolution_prompt,
    build_multiclass_prompt,
    parse_binary_response,
    parse_category_response,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    category: str

    @classmethod
    def from_raw(cls, raw: dict) -> "Sample":
        return cls(
            question_id=str(raw["question_id"]),
            question=raw["question"],
            category=raw["question_type"],
        )


def load_dataset(path: Path) -> list[Sample]:
    raw = json.loads(path.read_text())
    return [Sample.from_raw(r) for r in raw if r.get("question_type") in CATEGORY_SET]


def split_for_binaries(
    samples: list[Sample],
    shots_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Shot]], dict[str, list[Shot]], dict[str, list[Shot]], list[Sample]]:
    """For each category C produce:
        - pos_shots[C]: shots_per_class samples with gold == C
        - neg_shots[C]: shots_per_class samples with gold != C, balanced
                        across the other 5 classes (round-robin)

    Also produce ``mc_shots`` for the multi-class fallback prompt
    (shots_per_class per category).

    Test pool excludes every sample used in any few-shot."""
    rng = random.Random(seed)

    by_cat: dict[str, list[Sample]] = {c: [] for c in CATEGORIES}
    for s in samples:
        by_cat[s.category].append(s)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])

    used_ids: set[str] = set()
    pos_shots: dict[str, list[Shot]] = {}
    for c in CATEGORIES:
        pick = by_cat[c][:shots_per_class]
        pos_shots[c] = [Shot(question=s.question, category=c) for s in pick]
        used_ids.update(s.question_id for s in pick)

    neg_shots: dict[str, list[Shot]] = {}
    for c in CATEGORIES:
        others = [oc for oc in CATEGORIES if oc != c]
        pools = {
            oc: [s for s in by_cat[oc] if s.question_id not in used_ids]
            for oc in others
        }
        picked: list[Shot] = []
        i = 0
        while len(picked) < shots_per_class:
            oc = others[i % len(others)]
            if pools[oc]:
                pick = pools[oc].pop(0)
                picked.append(Shot(question=pick.question, category=oc))
                used_ids.add(pick.question_id)
            i += 1
            if i > shots_per_class * len(others) * 2:
                break
        neg_shots[c] = picked

    # Multi-class fallback shots — reuse the per-category positives so we
    # don't burn more samples; this also keeps the fallback's prompt aligned
    # with what each binary saw.
    mc_shots = pos_shots

    test_pool = [s for s in samples if s.question_id not in used_ids]
    rng.shuffle(test_pool)
    return pos_shots, neg_shots, mc_shots, test_pool


def _llm_call(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    max_retries: int = 2,
) -> tuple[str, float]:
    """Single LLM call. Returns (raw_text or 'ERROR:Type', latency_s)."""
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or "", time.monotonic() - t0
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR:{type(e).__name__}", time.monotonic() - t0
            time.sleep(1.5 * (attempt + 1))


def classify_question(
    client: Any,
    sample: Sample,
    binary_prompts: dict[str, str],
    multiclass_prompt: str,
    binary_model: str,
    arbiter_model: str,
    fallback_model: str,
    inner_pool: ThreadPoolExecutor,
) -> dict:
    """Run the v2 pipeline on one sample. Returns full trace + prediction."""
    t0 = time.monotonic()
    # Phase 1: parallel binaries.
    binary_results: dict[str, dict] = {}

    def _run_binary(cat: str) -> tuple[str, str, float]:
        raw, latency = _llm_call(
            client, binary_model, binary_prompts[cat],
            f"Question: {sample.question}\nAnswer:", max_tokens=16,
        )
        return cat, raw, latency

    futures = {inner_pool.submit(_run_binary, c): c for c in CATEGORIES}
    for fut in as_completed(futures):
        cat, raw, latency = fut.result()
        verdict = parse_binary_response(raw) if not raw.startswith("ERROR") else "ERROR"
        binary_results[cat] = {
            "raw": raw,
            "verdict": verdict,
            "latency_s": round(latency, 3),
        }

    yes_categories = [c for c in CATEGORIES if binary_results[c]["verdict"] == "YES"]

    # Phase 2: arbitrate or fall back.
    arbiter_call = None
    fallback_call = None
    if len(yes_categories) == 1:
        predicted = yes_categories[0]
        decision_path = "single_yes"
    elif len(yes_categories) >= 2:
        arb_prompt = build_conflict_resolution_prompt(sample.question, yes_categories)
        raw, latency = _llm_call(
            client, arbiter_model, arb_prompt,
            "Pick ONE category from the candidates above.", max_tokens=24,
        )
        parsed = parse_category_response(raw, valid=yes_categories)
        if parsed in yes_categories:
            predicted = parsed
            decision_path = "arbiter_picked"
        else:
            # Arbiter failed — fall back to the first YES (deterministic).
            predicted = yes_categories[0]
            decision_path = "arbiter_failed_fallback_first_yes"
        arbiter_call = {
            "candidates": yes_categories,
            "raw": raw,
            "parsed": parsed,
            "latency_s": round(latency, 3),
        }
    else:
        # Zero binaries said YES → call multi-class as fallback.
        raw, latency = _llm_call(
            client, fallback_model, multiclass_prompt,
            f"Question: {sample.question}\nCategory:", max_tokens=16,
        )
        parsed = parse_category_response(raw)
        if parsed != "UNKNOWN":
            predicted = parsed
            decision_path = "fallback_multiclass"
        else:
            # Last resort — predict the most common single-session-* class.
            predicted = "single-session-user"
            decision_path = "fallback_default"
        fallback_call = {
            "raw": raw,
            "parsed": parsed,
            "latency_s": round(latency, 3),
        }

    return {
        "question_id": sample.question_id,
        "question": sample.question,
        "gold_category": sample.category,
        "predicted": predicted,
        "correct": predicted == sample.category,
        "yes_categories": yes_categories,
        "decision_path": decision_path,
        "binary_results": binary_results,
        "arbiter_call": arbiter_call,
        "fallback_call": fallback_call,
        "total_latency_s": round(time.monotonic() - t0, 3),
    }


def evaluate(
    test_pool: list[Sample],
    binary_prompts: dict[str, str],
    multiclass_prompt: str,
    binary_model: str,
    arbiter_model: str,
    fallback_model: str,
    api_key: str,
    base_url: str,
    parallel_questions: int,
    binary_inner_parallel: int,
    verbose: bool,
) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    # One inner pool for binary parallelism, shared across question workers.
    # Outer pool processes questions concurrently; each one submits 6 binaries.
    results: list[dict] = []
    completed = 0
    total = len(test_pool)

    inner_pool = ThreadPoolExecutor(max_workers=binary_inner_parallel)
    try:
        with ThreadPoolExecutor(max_workers=parallel_questions) as outer:
            futures = {
                outer.submit(
                    classify_question,
                    client, s, binary_prompts, multiclass_prompt,
                    binary_model, arbiter_model, fallback_model, inner_pool,
                ): s for s in test_pool
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1
                if verbose and completed % 20 == 0:
                    print(f"  [{completed}/{total}] processed", flush=True)
    finally:
        inner_pool.shutdown(wait=True)
    return results


def per_class_metrics(results: list[dict]) -> dict:
    confusion: dict[str, dict[str, int]] = {
        c: {p: 0 for p in CATEGORIES} for c in CATEGORIES
    }
    decision_hist = {"single_yes": 0, "arbiter_picked": 0,
                     "arbiter_failed_fallback_first_yes": 0,
                     "fallback_multiclass": 0, "fallback_default": 0}
    yes_count_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    correct = 0
    total = len(results)
    for r in results:
        gold = r["gold_category"]
        pred = r["predicted"]
        if pred in CATEGORY_SET:
            confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
        decision_hist[r["decision_path"]] = decision_hist.get(r["decision_path"], 0) + 1
        n_yes = len(r["yes_categories"])
        yes_count_hist[n_yes] = yes_count_hist.get(n_yes, 0) + 1

    accuracy = correct / total if total else 0.0
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
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total": total,
        "correct": correct,
        "decision_hist": decision_hist,
        "yes_count_hist": yes_count_hist,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_confusion(metrics: dict) -> None:
    short = {
        "temporal-reasoning": "temp", "multi-session": "multi",
        "knowledge-update": "k-upd", "single-session-user": "ss-u",
        "single-session-assistant": "ss-a", "single-session-preference": "ss-p",
    }
    print()
    print(f"{'gold ↓ / pred →':<28} " + " ".join(f"{short[c]:>5}" for c in CATEGORIES))
    for g in CATEGORIES:
        row = metrics["confusion"][g]
        print(f"{g:<28} " + " ".join(f"{row[p]:>5}" for p in CATEGORIES))


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--balanced-test", action="store_true")
    p.add_argument("--binary-model", default="openai/gpt-5.1")
    p.add_argument("--arbiter-model", default="openai/gpt-5.1")
    p.add_argument("--fallback-model", default="openai/gpt-4.1-mini")
    p.add_argument(
        "--base-url", default=OPENROUTER_BASE_URL,
        help=f"OpenAI-compatible endpoint (default: {OPENROUTER_BASE_URL}). "
             f"For local Ollama use {OLLAMA_BASE_URL}.",
    )
    p.add_argument(
        "--api-key-env", default="OPENROUTER_API_KEY",
        help="Env var holding the API key (default: OPENROUTER_API_KEY). "
             "Set to OLLAMA_KEY (any non-empty value) for local Ollama.",
    )
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel-questions", type=int, default=4)
    p.add_argument("--binary-inner-parallel", type=int, default=6)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    load_env(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env)
    # Ollama doesn't require a real key; allow anything non-empty.
    if not api_key:
        if "localhost" in args.base_url or "127.0.0.1" in args.base_url:
            api_key = "ollama"  # placeholder; openai client requires a value
        else:
            print(f"ERROR: env var {args.api_key_env} not set", file=sys.stderr)
            return 2

    samples = load_dataset(args.data)
    pos, neg, mc, test_pool = split_for_binaries(samples, args.shots_per_class, args.seed)

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

    print(f"shots: {args.shots_per_class}/class × 6 binaries + multiclass fallback")
    print(f"test pool: {len(test_pool)} samples")
    by_cat_count = {c: 0 for c in CATEGORIES}
    for s in test_pool:
        by_cat_count[s.category] += 1
    for c in CATEGORIES:
        print(f"  {c:<28} {by_cat_count[c]}")
    print(f"binary={args.binary_model}  arbiter={args.arbiter_model}  fallback={args.fallback_model}")
    print(f"parallel: outer={args.parallel_questions}  inner={args.binary_inner_parallel}")

    binary_prompts = {c: build_binary_prompt(c, pos[c], neg[c]) for c in CATEGORIES}
    multiclass_prompt = build_multiclass_prompt(mc)

    started = time.monotonic()
    results = evaluate(
        test_pool, binary_prompts, multiclass_prompt,
        args.binary_model, args.arbiter_model, args.fallback_model,
        api_key, args.base_url,
        args.parallel_questions, args.binary_inner_parallel,
        args.verbose,
    )
    elapsed = time.monotonic() - started

    metrics = per_class_metrics(results)
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["binary_model"] = args.binary_model
    metrics["arbiter_model"] = args.arbiter_model
    metrics["fallback_model"] = args.fallback_model
    metrics["shots_per_class"] = args.shots_per_class
    metrics["seed"] = args.seed

    print()
    print("=== overall ===")
    print(f"  accuracy:  {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  elapsed:   {metrics['elapsed_s']}s")
    print()
    print("  decision paths:")
    for k, v in metrics["decision_hist"].items():
        print(f"    {k:<40} {v}")
    print()
    print("  YES-count distribution (how many binaries said YES per question):")
    for k in sorted(metrics["yes_count_hist"]):
        print(f"    {k} YES votes: {metrics['yes_count_hist'][k]}")

    print()
    print("=== per-class ===")
    print(f"  {'category':<28} {'support':>8} {'P':>7} {'R':>7} {'F1':>7}")
    for c in CATEGORIES:
        m = metrics["per_class"][c]
        print(f"  {c:<28} {m['support']:>8} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")

    print_confusion(metrics)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": metrics,
            "shots_positive": {c: [{"question": s.question, "category": s.category} for s in pos[c]] for c in CATEGORIES},
            "shots_negative": {c: [{"question": s.question, "category": s.category} for s in neg[c]] for c in CATEGORIES},
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
