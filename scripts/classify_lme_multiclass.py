"""Few-shot multi-class classifier for LongMemEval question categories.

Given a question, predict ONE of the 6 LongMemEval categories using a
balanced few-shot prompt drawn from the dataset itself.

Categories (gold taxonomy):
    temporal-reasoning, multi-session, knowledge-update,
    single-session-user, single-session-assistant, single-session-preference

Reports overall accuracy, per-class precision/recall/F1, macro F1, and a
confusion matrix. Reproducible via --seed.

Usage:
    python -m scripts.classify_lme_multiclass \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --shots-per-class 4 \\
        --max-test 240 \\
        --balanced-test \\
        --model openai/gpt-5.1 \\
        --env-file .env \\
        --output logs/classifier_multiclass.json
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CATEGORIES = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]
CATEGORY_SET = set(CATEGORIES)

# Short, taxonomy-aligned descriptions used in the prompt header.
CATEGORY_DESCRIPTIONS = {
    "temporal-reasoning": "answer requires date arithmetic on event dates (durations, gaps, ordering by time).",
    "multi-session": "answer requires combining facts from multiple sessions (counts across time, totals, comparisons).",
    "knowledge-update": "asks about how a stated fact changed over time (current vs previous status, weight loss, switches).",
    "single-session-user": "answer is a fact the USER stated in one session (events, attendance, where/who/what they did).",
    "single-session-assistant": "answer is something the ASSISTANT recommended or stated in one session.",
    "single-session-preference": "asks about a user preference, like/dislike, or stated constraint.",
}


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
    out = [Sample.from_raw(r) for r in raw]
    bad = [s for s in out if s.category not in CATEGORY_SET]
    if bad:
        print(f"WARNING: {len(bad)} samples have unknown category, dropping", file=sys.stderr)
        out = [s for s in out if s.category in CATEGORY_SET]
    return out


def split_few_shot(
    samples: list[Sample],
    shots_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Sample]], list[Sample]]:
    """Stratified split: shots_per_class per category as few-shot, rest is test pool."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Sample]] = {c: [] for c in CATEGORIES}
    for s in samples:
        by_cat[s.category].append(s)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])

    shots = {c: by_cat[c][:shots_per_class] for c in CATEGORIES}
    test_pool: list[Sample] = []
    for c in CATEGORIES:
        test_pool.extend(by_cat[c][shots_per_class:])
    rng.shuffle(test_pool)
    return shots, test_pool


def build_prompt(shots: dict[str, list[Sample]]) -> str:
    """Build a system prompt with a category guide + interleaved few-shot."""
    lines = [
        "You classify questions into ONE of these LongMemEval categories. "
        "Reply with EXACTLY the category name (lowercase, with hyphens). "
        "No prose, no explanation, no quotes.",
        "",
        "Categories:",
    ]
    for c in CATEGORIES:
        lines.append(f"  - {c}: {CATEGORY_DESCRIPTIONS[c]}")
    lines.append("")
    lines.append("Examples:")
    lines.append("")

    # Round-robin so positions are interleaved across categories.
    n = max(len(v) for v in shots.values())
    for i in range(n):
        for c in CATEGORIES:
            if i < len(shots[c]):
                s = shots[c][i]
                lines.append(f"Question: {s.question}")
                lines.append(f"Category: {c}")
                lines.append("")

    lines.append("Now classify the next question. Reply with only the category name.")
    return "\n".join(lines)


def classify_one(
    client: Any,
    system_prompt: str,
    question: str,
    model: str,
    max_retries: int = 2,
) -> tuple[str, float]:
    """Returns (predicted_category, latency_seconds)."""
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nCategory:"},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            # Normalize: strip punctuation, take first token-like chunk.
            raw = raw.replace("`", "").replace('"', "").replace("'", "").strip()
            # Find a known category as a substring (handles minor wrappers).
            for c in CATEGORIES:
                if raw == c or raw.startswith(c) or c in raw:
                    return c, time.monotonic() - t0
            return f"UNKNOWN:{raw[:40]}", time.monotonic() - t0
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR:{type(e).__name__}", time.monotonic() - t0
            time.sleep(1.5 * (attempt + 1))


def evaluate(
    test_pool: list[Sample],
    system_prompt: str,
    model: str,
    api_key: str,
    parallel: int,
    verbose: bool,
) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    results: list[dict] = []
    completed = 0
    total = len(test_pool)

    def _run(sample: Sample) -> dict:
        pred, latency = classify_one(client, system_prompt, sample.question, model)
        return {
            "question_id": sample.question_id,
            "question": sample.question,
            "gold_category": sample.category,
            "predicted": pred,
            "correct": pred == sample.category,
            "latency_s": round(latency, 3),
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_run, s): s for s in test_pool}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            if verbose and completed % 25 == 0:
                print(f"  [{completed}/{total}] processed", flush=True)
    return results


def per_class_metrics(results: list[dict]) -> dict:
    """Compute per-class precision/recall/F1 + macro F1 + accuracy + confusion matrix.

    UNKNOWN/ERROR predictions count as wrong (and don't count toward any class TP/FP)."""
    # confusion[gold][predicted] = count
    confusion: dict[str, dict[str, int]] = {
        c: {p: 0 for p in CATEGORIES} for c in CATEGORIES
    }
    unknown = errors = 0
    correct = 0
    total = len(results)

    for r in results:
        gold = r["gold_category"]
        pred = r["predicted"]
        if pred.startswith("ERROR"):
            errors += 1
            continue
        if pred.startswith("UNKNOWN"):
            unknown += 1
            continue
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1

    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for c in CATEGORIES:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in CATEGORIES if g != c)
        fn = sum(confusion[c][p] for p in CATEGORIES if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
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
        "unknown": unknown,
        "errors": errors,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_confusion(metrics: dict) -> None:
    """Pretty-print the confusion matrix (rows = gold, cols = predicted)."""
    short = {
        "temporal-reasoning": "temp",
        "multi-session": "multi",
        "knowledge-update": "k-upd",
        "single-session-user": "ss-u",
        "single-session-assistant": "ss-a",
        "single-session-preference": "ss-p",
    }
    cols = [short[c] for c in CATEGORIES]
    print()
    print(f"{'gold ↓ / pred →':<24} " + " ".join(f"{c:>5}" for c in cols))
    for g in CATEGORIES:
        row = metrics["confusion"][g]
        print(f"{g:<24} " + " ".join(f"{row[p]:>5}" for p in CATEGORIES))


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
    p.add_argument("--shots-per-class", type=int, default=4)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--balanced-test", action="store_true",
                   help="Equal samples per gold class in test pool (cap ≈ --max-test)")
    p.add_argument("--model", default="openai/gpt-5.1")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    load_env(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
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

    print(f"shots: {args.shots_per_class}/class × {len(CATEGORIES)} = {sum(len(v) for v in shots.values())} total")
    print(f"test pool: {len(test_pool)} samples")
    by_cat_count = {c: 0 for c in CATEGORIES}
    for s in test_pool:
        by_cat_count[s.category] += 1
    for c in CATEGORIES:
        print(f"  {c:<28} {by_cat_count[c]}")
    print(f"model: {args.model}, parallel: {args.parallel}")

    system_prompt = build_prompt(shots)
    if args.verbose:
        print("--- system prompt ---")
        print(system_prompt)
        print("--- end prompt ---")

    started = time.monotonic()
    results = evaluate(
        test_pool, system_prompt, args.model, api_key,
        args.parallel, args.verbose,
    )
    elapsed = time.monotonic() - started

    metrics = per_class_metrics(results)
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["model"] = args.model
    metrics["shots_per_class"] = args.shots_per_class
    metrics["seed"] = args.seed

    print()
    print("=== overall ===")
    print(f"  accuracy:  {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  unknown:   {metrics['unknown']}")
    print(f"  errors:    {metrics['errors']}")
    print(f"  elapsed:   {metrics['elapsed_s']}s")

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
            "shots": {c: [{"question_id": s.question_id, "question": s.question} for s in shots[c]] for c in CATEGORIES},
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
