"""Cascade classifier: chain 6 binary YES/NO classifiers, first YES wins.

For each question, run binary classifiers in a fixed order. The first YES
short-circuits and emits that category. If all six say NO, fall back to the
default (last category in the order, by convention).

Each stage gets its own balanced few-shot prompt drawn from the dataset (same
seed as the binary script so shots are reproducible across runs).

Reports: overall accuracy, per-class P/R/F1, average stages per question,
confusion matrix, and a stage-exit histogram.

Usage:
    python -m scripts.classify_lme_cascade \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --order temporal-reasoning,multi-session,knowledge-update,single-session-preference,single-session-assistant,single-session-user \\
        --shots-per-class 6 \\
        --max-test 240 \\
        --balanced-test \\
        --model openai/gpt-5.1 \\
        --env-file .env \\
        --output logs/classifier_cascade.json
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

CATEGORY_DESCRIPTIONS = {
    "temporal-reasoning": "answer requires date arithmetic on event dates (durations, gaps, ordering by time).",
    "multi-session": "answer requires combining facts from multiple sessions (counts across time, totals, comparisons).",
    "knowledge-update": "asks how a stated fact changed over time (current vs previous status, weight loss, switches).",
    "single-session-user": "answer is a fact the USER stated in one session (events, attendance, where/who/what).",
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
    return [Sample.from_raw(r) for r in raw if r.get("question_type") in CATEGORY_SET]


def split_for_cascade(
    samples: list[Sample],
    shots_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Sample]], dict[str, list[Sample]], list[Sample]]:
    """For each category C, hold out:
        - shots_per_class POSITIVE shots (gold=C)
        - shots_per_class NEGATIVE shots (gold≠C, drawn from each other category)

    Test pool = samples NOT used as any few-shot.

    Returns (positive_shots, negative_shots, test_pool) keyed by category.
    """
    rng = random.Random(seed)

    by_cat: dict[str, list[Sample]] = {c: [] for c in CATEGORIES}
    for s in samples:
        by_cat[s.category].append(s)
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])

    used_ids: set[str] = set()
    pos_shots: dict[str, list[Sample]] = {}
    for c in CATEGORIES:
        pick = by_cat[c][:shots_per_class]
        pos_shots[c] = pick
        used_ids.update(s.question_id for s in pick)

    # For negative shots per stage C, draw shots_per_class samples balanced
    # across the OTHER 5 categories (round-robin).
    neg_shots: dict[str, list[Sample]] = {}
    for c in CATEGORIES:
        others = [oc for oc in CATEGORIES if oc != c]
        # pool of unused samples from each other category
        pools = {
            oc: [s for s in by_cat[oc] if s.question_id not in used_ids]
            for oc in others
        }
        picked: list[Sample] = []
        i = 0
        while len(picked) < shots_per_class:
            oc = others[i % len(others)]
            if pools[oc]:
                pick = pools[oc].pop(0)
                picked.append(pick)
                used_ids.add(pick.question_id)
            i += 1
            if i > shots_per_class * len(others) * 2:  # safety
                break
        neg_shots[c] = picked

    test_pool = [s for s in samples if s.question_id not in used_ids]
    rng.shuffle(test_pool)
    return pos_shots, neg_shots, test_pool


def build_binary_prompt(
    target: str, pos_shots: list[Sample], neg_shots: list[Sample]
) -> str:
    lines = [
        f'You are a question classifier. Decide whether the next question is '
        f'in category "{target}" — defined as: {CATEGORY_DESCRIPTIONS[target]}',
        "",
        "Reply with EXACTLY one word: YES or NO. No prose, no explanation.",
        "",
        "Examples:",
        "",
    ]
    for pos, neg in zip(pos_shots, neg_shots):
        lines.append(f"Question: {pos.question}")
        lines.append("Answer: YES")
        lines.append("")
        lines.append(f"Question: {neg.question}")
        lines.append("Answer: NO")
        lines.append("")
    lines.append("Now classify the next question. Reply with only YES or NO.")
    return "\n".join(lines)


def classify_binary(
    client: Any,
    system_prompt: str,
    question: str,
    model: str,
    max_retries: int = 2,
) -> tuple[str, float]:
    """Returns (YES/NO/UNKNOWN/ERROR:..., latency_seconds)."""
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nAnswer:"},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip().upper()
            if raw.startswith("YES"):
                return "YES", time.monotonic() - t0
            if raw.startswith("NO"):
                return "NO", time.monotonic() - t0
            return "UNKNOWN", time.monotonic() - t0
        except Exception as e:
            if attempt == max_retries:
                return f"ERROR:{type(e).__name__}", time.monotonic() - t0
            time.sleep(1.5 * (attempt + 1))


def classify_cascade(
    client: Any,
    prompts_in_order: list[tuple[str, str]],  # [(category, system_prompt)]
    question: str,
    model: str,
) -> dict:
    """Run the cascade for one question. Returns {predicted, stages_used, total_latency_s, trace}."""
    trace: list[dict] = []
    total_latency = 0.0
    for cat, prompt in prompts_in_order:
        verdict, latency = classify_binary(client, prompt, question, model)
        total_latency += latency
        trace.append({"stage": cat, "verdict": verdict, "latency_s": round(latency, 3)})
        if verdict == "YES":
            return {
                "predicted": cat,
                "stages_used": len(trace),
                "total_latency_s": round(total_latency, 3),
                "exit_reason": "first_yes",
                "trace": trace,
            }
        if verdict.startswith("ERROR"):
            return {
                "predicted": f"ERROR:{verdict[6:]}",
                "stages_used": len(trace),
                "total_latency_s": round(total_latency, 3),
                "exit_reason": "error",
                "trace": trace,
            }
    # All NO → default to the last category in the order.
    default_cat = prompts_in_order[-1][0]
    return {
        "predicted": default_cat,
        "stages_used": len(prompts_in_order),
        "total_latency_s": round(total_latency, 3),
        "exit_reason": "default_all_no",
        "trace": trace,
    }


def evaluate(
    test_pool: list[Sample],
    prompts_in_order: list[tuple[str, str]],
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
        out = classify_cascade(client, prompts_in_order, sample.question, model)
        out.update({
            "question_id": sample.question_id,
            "question": sample.question,
            "gold_category": sample.category,
            "correct": out["predicted"] == sample.category,
        })
        return out

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
    confusion: dict[str, dict[str, int]] = {
        c: {p: 0 for p in CATEGORIES} for c in CATEGORIES
    }
    errors = 0
    correct = 0
    total = len(results)
    stages_total = 0
    stage_exit_hist = {c: 0 for c in CATEGORIES}
    stage_exit_hist["default_all_no"] = 0

    for r in results:
        gold = r["gold_category"]
        pred = r["predicted"]
        stages_total += r["stages_used"]
        if pred.startswith("ERROR"):
            errors += 1
            continue
        if pred not in CATEGORY_SET:
            errors += 1
            continue
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
        if r["exit_reason"] == "default_all_no":
            stage_exit_hist["default_all_no"] += 1
        else:
            stage_exit_hist[pred] += 1

    accuracy = correct / total if total else 0.0
    avg_stages = stages_total / total if total else 0.0

    per_class = {}
    f1s = []
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

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "total": total,
        "correct": correct,
        "errors": errors,
        "avg_stages_per_question": round(avg_stages, 3),
        "stage_exit_hist": stage_exit_hist,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_confusion(metrics: dict) -> None:
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
    print(f"{'gold ↓ / pred →':<28} " + " ".join(f"{c:>5}" for c in cols))
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
    p.add_argument(
        "--order",
        default="temporal-reasoning,multi-session,knowledge-update,single-session-preference,single-session-assistant,single-session-user",
        help="Comma-separated cascade order. Last category is the default if all stages say NO.",
    )
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--balanced-test", action="store_true")
    p.add_argument("--model", default="openai/gpt-5.1")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    order = [c.strip() for c in args.order.split(",") if c.strip()]
    bad = [c for c in order if c not in CATEGORY_SET]
    if bad:
        print(f"ERROR: unknown categories in --order: {bad}", file=sys.stderr)
        return 2
    if set(order) != CATEGORY_SET:
        print(f"ERROR: --order must contain all 6 categories exactly once", file=sys.stderr)
        return 2

    load_env(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    samples = load_dataset(args.data)
    pos_shots, neg_shots, test_pool = split_for_cascade(
        samples, args.shots_per_class, args.seed
    )

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

    print(f"cascade order: {' → '.join(order)}")
    print(f"shots: {args.shots_per_class}/class × 6 stages = {args.shots_per_class * 12} shot-uses")
    print(f"test pool: {len(test_pool)} samples")
    by_cat_count = {c: 0 for c in CATEGORIES}
    for s in test_pool:
        by_cat_count[s.category] += 1
    for c in CATEGORIES:
        print(f"  {c:<28} {by_cat_count[c]}")
    print(f"model: {args.model}, parallel: {args.parallel}")

    prompts_in_order = [
        (cat, build_binary_prompt(cat, pos_shots[cat], neg_shots[cat]))
        for cat in order
    ]

    if args.verbose:
        print(f"--- stage 1 prompt ({order[0]}) ---")
        print(prompts_in_order[0][1])
        print("--- end ---")

    started = time.monotonic()
    results = evaluate(
        test_pool, prompts_in_order, args.model, api_key,
        args.parallel, args.verbose,
    )
    elapsed = time.monotonic() - started

    metrics = per_class_metrics(results)
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["model"] = args.model
    metrics["shots_per_class"] = args.shots_per_class
    metrics["seed"] = args.seed
    metrics["order"] = order

    print()
    print("=== overall ===")
    print(f"  accuracy:     {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  errors:       {metrics['errors']}")
    print(f"  avg stages:   {metrics['avg_stages_per_question']} (out of 6)")
    print(f"  elapsed:      {metrics['elapsed_s']}s")
    print()
    print("  exit by stage:")
    for c in order:
        print(f"    {c:<30} {metrics['stage_exit_hist'][c]}")
    print(f"    {'(default — all NO)':<30} {metrics['stage_exit_hist']['default_all_no']}")

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
            "shots_positive": {c: [{"question_id": s.question_id, "question": s.question} for s in pos_shots[c]] for c in CATEGORIES},
            "shots_negative": {c: [{"question_id": s.question_id, "question": s.question, "gold_category": s.category} for s in neg_shots[c]] for c in CATEGORIES},
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n  wrote: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
