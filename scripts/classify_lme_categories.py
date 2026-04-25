"""Few-shot binary classifier for LongMemEval question categories.

Given a target category (e.g. ``temporal-reasoning``), classify each held-out
question as YES (in-category) or NO (out-of-category) using an LLM with a
balanced few-shot prompt drawn from the dataset itself.

Reports accuracy, precision, recall, F1, and a confusion matrix. Reproducible:
the few-shot split is seeded.

Usage:
    python -m scripts.classify_lme_categories \\
        --data /path/to/longmemeval_s_cleaned.json \\
        --target temporal-reasoning \\
        --shots-per-class 6 \\
        --max-test 200 \\
        --model openai/gpt-4.1-mini \\
        --env-file .env \\
        --output logs/classifier_temporal.json
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

VALID_CATEGORIES = {
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}


@dataclass(frozen=True)
class Sample:
    question_id: str
    question: str
    category: str

    @property
    def is_target(self) -> bool:
        return self._is_target

    @classmethod
    def from_raw(cls, raw: dict, target: str) -> "Sample":
        s = cls(
            question_id=str(raw["question_id"]),
            question=raw["question"],
            category=raw["question_type"],
        )
        object.__setattr__(s, "_is_target", s.category == target)
        return s


def load_dataset(path: Path, target: str) -> list[Sample]:
    raw = json.loads(path.read_text())
    return [Sample.from_raw(r, target) for r in raw]


def split_few_shot(
    samples: list[Sample],
    shots_per_class: int,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified split: K positives + K negatives held out as few-shot examples,
    rest is the test pool. Returns (positive_shots, negative_shots, test_pool)."""
    rng = random.Random(seed)
    positives = [s for s in samples if s._is_target]
    negatives = [s for s in samples if not s._is_target]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_shots = positives[:shots_per_class]
    neg_shots = negatives[:shots_per_class]
    test = positives[shots_per_class:] + negatives[shots_per_class:]
    rng.shuffle(test)
    return pos_shots, neg_shots, test


def build_prompt(target: str, pos_shots: list[Sample], neg_shots: list[Sample]) -> str:
    """Build the system prompt with balanced few-shot examples.

    The model sees alternating pos/neg examples and must answer YES or NO."""
    lines = [
        f'You are a question classifier. Decide whether each question belongs '
        f'to the LongMemEval category "{target}".',
        "",
        f'A "{target}" question is one whose answer requires the kind of '
        f'reasoning typical of that category. Your output must be a single '
        f'word: YES or NO.',
        "",
        "Examples:",
        "",
    ]
    # Interleave to discourage position bias.
    for pos, neg in zip(pos_shots, neg_shots):
        lines.append(f'Question: {pos.question}')
        lines.append('Answer: YES')
        lines.append('')
        lines.append(f'Question: {neg.question}')
        lines.append('Answer: NO')
        lines.append('')
    lines.append('Now classify the next question. Reply with only YES or NO.')
    return "\n".join(lines)


def classify_one(
    client: Any,
    system_prompt: str,
    question: str,
    model: str,
    max_retries: int = 2,
) -> tuple[str, float]:
    """Call the LLM. Returns (prediction, latency_seconds). Prediction is YES/NO/UNKNOWN."""
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
            "gold_label": "YES" if sample._is_target else "NO",
            "predicted": pred,
            "correct": pred == ("YES" if sample._is_target else "NO"),
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


def confusion_matrix(results: list[dict]) -> dict:
    """Compute TP/FP/TN/FN + derived metrics. UNKNOWN/ERROR predictions count
    as wrong (false negatives if gold=YES, false positives if gold=NO)."""
    tp = fp = tn = fn = unknown = errors = 0
    for r in results:
        gold = r["gold_label"]
        pred = r["predicted"]
        if pred.startswith("ERROR"):
            errors += 1
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
        elif pred == "NO" and gold == "YES":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unknown": unknown,
        "errors": errors,
        "total": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def load_env(path: Path) -> None:
    """Minimal .env loader — sets OPENROUTER_API_KEY if not already set."""
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
    p.add_argument("--target", required=True, choices=sorted(VALID_CATEGORIES))
    p.add_argument("--shots-per-class", type=int, default=6)
    p.add_argument("--max-test", type=int, default=None,
                   help="Cap on test pool size (default: use all)")
    p.add_argument("--balanced-test", action="store_true",
                   help="Sample equal positives and negatives in test pool (cap = --max-test)")
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None,
                   help="Write full results JSON to this path")
    args = p.parse_args()

    load_env(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY (or OPENAI_API_KEY) not set", file=sys.stderr)
        return 2

    samples = load_dataset(args.data, args.target)
    pos_shots, neg_shots, test_pool = split_few_shot(
        samples, args.shots_per_class, args.seed
    )

    if args.balanced_test and args.max_test:
        per_class = args.max_test // 2
        pos_test = [s for s in test_pool if s._is_target][:per_class]
        neg_test = [s for s in test_pool if not s._is_target][:per_class]
        rng = random.Random(args.seed + 1)
        test_pool = pos_test + neg_test
        rng.shuffle(test_pool)
    elif args.max_test:
        test_pool = test_pool[: args.max_test]

    print(f"target category: {args.target}")
    print(f"shots: {len(pos_shots)} positive + {len(neg_shots)} negative = {len(pos_shots) + len(neg_shots)} total")
    print(f"test pool: {len(test_pool)} samples ({sum(1 for s in test_pool if s._is_target)} positive)")
    print(f"model: {args.model}, parallel: {args.parallel}")

    system_prompt = build_prompt(args.target, pos_shots, neg_shots)
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

    metrics = confusion_matrix(results)
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["target"] = args.target
    metrics["model"] = args.model
    metrics["shots_per_class"] = args.shots_per_class
    metrics["seed"] = args.seed

    print()
    print("=== metrics ===")
    for k in ("tp", "fp", "tn", "fn", "unknown", "errors", "total"):
        print(f"  {k}: {metrics[k]}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  elapsed:   {metrics['elapsed_s']}s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": metrics,
            "shots_positive": [{"question_id": s.question_id, "question": s.question, "category": s.category} for s in pos_shots],
            "shots_negative": [{"question_id": s.question_id, "question": s.question, "category": s.category} for s in neg_shots],
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"  wrote: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
