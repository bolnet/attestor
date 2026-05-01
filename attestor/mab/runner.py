"""MAB runner — dataset loading, run_mab() entry, and result printing."""

from __future__ import annotations

import logging
import statistics
import tempfile
import time
from collections import defaultdict
from typing import Any

from attestor.core import AgentMemory
from attestor.mab.retrieval import (
    DEFAULT_CHUNK_SIZE,
    _upgrade_embeddings_for_benchmark,
    answer_question,
    ingest_context,
)
from attestor.mab.scoring import score_question
from attestor.utils.tokens import estimate_tokens

logger = logging.getLogger("attestor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET = "ai-hyz/MemoryAgentBench"

SPLIT_MAP = {
    "Accurate_Retrieval": "AR",
    "Test_Time_Learning": "TTL",
    "Long_Range_Understanding": "LRU",
    "Conflict_Resolution": "CR",
}
CATEGORY_TO_SPLIT = {v: k for k, v in SPLIT_MAP.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mab(
    categories: list[str] | None = None,
    max_examples: int | None = None,
    cache_dir: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load MemoryAgentBench dataset from HuggingFace.

    Args:
        categories: Category codes to load (default: all).
        max_examples: Maximum examples per category.
        cache_dir: HuggingFace cache directory.

    Returns:
        Dict mapping category code to list of examples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: poetry add \"attestor[benchmark]\""
        )

    if categories is None:
        categories = list(CATEGORY_TO_SPLIT.keys())

    result: dict[str, list[dict[str, Any]]] = {}

    for cat in categories:
        split_name = CATEGORY_TO_SPLIT.get(cat)
        if not split_name:
            print(f"Warning: Unknown category '{cat}', skipping")
            continue

        ds = load_dataset(HF_DATASET, split=split_name, cache_dir=cache_dir)
        examples = []
        for idx, row in enumerate(ds):
            if max_examples and idx >= max_examples:
                break
            examples.append({
                "context": row["context"],
                "questions": row["questions"],
                "answers": row["answers"],
                "metadata": row.get("metadata", {}),
            })
        result[cat] = examples

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_mab(
    categories: list[str] | None = None,
    max_examples: int | None = None,
    max_questions: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_max_tokens: int | None = None,
    answer_model: str | None = None,
    recall_budget: int = 6000,
    verbose: bool = False,
    api_key: str | None = None,
    cache_dir: str | None = None,
    skip_examples: int = 0,
    backend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the MemoryAgentBench benchmark.

    Args:
        categories: Category codes to evaluate (default: ["AR", "CR"]).
        max_examples: Max examples per category.
        skip_examples: Skip first N examples per category.
        max_questions: Max questions per example.
        chunk_size: Tokens per chunk for ingestion.
        context_max_tokens: Truncate contexts to this many tokens.
        answer_model: LLM for answer synthesis.
        recall_budget: Token budget for recall queries.
        verbose: Print progress.
        api_key: Anthropic API key.
        cache_dir: HuggingFace cache directory.

    Returns:
        Dict with overall scores, per-category, per-subtask, timing, config.
    """
    if categories is None:
        categories = ["AR", "CR"]
    if answer_model is None:
        from attestor.config import get_stack
        answer_model = get_stack().models.benchmark_default

    # Load data
    data = load_mab(
        categories=categories,
        max_examples=max_examples,
        cache_dir=cache_dir,
    )

    # Results tracking
    category_scores: dict[str, list[float]] = defaultdict(list)
    subtask_scores: dict[str, list[float]] = defaultdict(list)
    all_details: list[dict[str, Any]] = []
    total_ingest_time = 0.0
    total_recall_time = 0.0
    total_chunks = 0
    total_questions = 0

    for cat, examples in data.items():
        if skip_examples:
            examples = examples[skip_examples:]
        if verbose:
            print(f"\n=== Category: {cat} ({len(examples)} examples) ===")

        for ex_idx, example in enumerate(examples):
            source = example["metadata"].get("source", "unknown")
            qa_pair_ids = example["metadata"].get("qa_pair_ids", [])

            if verbose:
                ctx_tokens = estimate_tokens(example["context"])
                print(
                    f"\n  Example {ex_idx + 1}/{len(examples)}: "
                    f"source={source}, ~{ctx_tokens} tokens, "
                    f"{len(example['questions'])} questions"
                )

            # Fresh store per example
            with tempfile.TemporaryDirectory() as tmpdir:
                mem = AgentMemory(tmpdir, config=backend_config)
                # Refresh the embedder so any provider key set after
                # store construction is picked up; provider selection
                # itself is owned by configs/attestor.yaml.
                _upgrade_embeddings_for_benchmark(mem)
                mem._retrieval.enable_temporal_boost = False

                # Disable vector+contradiction during bulk add, batch embed after
                vector_store = mem._vector_store
                mem._vector_store = None
                original_check = mem._temporal.check_contradictions
                mem._temporal.check_contradictions = lambda m: []

                # Ingest
                t0 = time.perf_counter()
                num_chunks, num_tokens = ingest_context(
                    mem, example["context"],
                    chunk_size=chunk_size,
                    context_max_tokens=context_max_tokens,
                    verbose=verbose,
                )
                total_chunks += num_chunks

                # Restore and batch embed
                mem._vector_store = vector_store
                mem._retrieval.vector_store = vector_store
                mem._temporal.check_contradictions = original_check
                embed_count = mem.batch_embed()
                ingest_time = time.perf_counter() - t0
                total_ingest_time += ingest_time

                if verbose:
                    print(
                        f"    Ingested {num_chunks} chunks "
                        f"({embed_count} embedded) in {ingest_time:.1f}s"
                    )

                # Answer and score each question
                questions = example["questions"]
                answers_list = example["answers"]
                if max_questions:
                    questions = questions[:max_questions]
                    answers_list = answers_list[:max_questions]

                for q_idx, (question, answers) in enumerate(
                    zip(questions, answers_list)
                ):
                    total_questions += 1

                    # Recall + answer
                    t0 = time.perf_counter()
                    prediction = answer_question(
                        mem, question,
                        budget=recall_budget,
                        model=answer_model,
                        api_key=api_key,
                        source=source,
                    )
                    recall_time = time.perf_counter() - t0
                    total_recall_time += recall_time

                    # Score
                    scores = score_question(prediction, answers, source)
                    primary_score = next(iter(scores.values()), 0.0)
                    category_scores[cat].append(primary_score)
                    subtask_scores[source].append(primary_score)

                    qa_id = (
                        qa_pair_ids[q_idx]
                        if qa_pair_ids and q_idx < len(qa_pair_ids)
                        else ""
                    )

                    detail = {
                        "category": cat,
                        "source": source,
                        "example_idx": ex_idx,
                        "question_idx": q_idx,
                        "qa_pair_id": qa_id,
                        "question": question[:200],
                        "prediction": prediction[:300],
                        "scores": scores,
                        "primary_score": primary_score,
                        "recall_ms": round(recall_time * 1000, 1),
                    }
                    all_details.append(detail)

                    if verbose:
                        score_str = " ".join(
                            f"{k}={v:.2f}" for k, v in scores.items()
                        )
                        status = "PASS" if primary_score >= 0.5 else "FAIL"
                        print(
                            f"    [{status}] Q{q_idx}: {score_str} "
                            f"({recall_time * 1000:.0f}ms)"
                        )

                mem.close()

    # Aggregate scores
    by_category: dict[str, dict[str, Any]] = {}
    for cat, scores_list in category_scores.items():
        if scores_list:
            by_category[cat] = {
                "accuracy": round(statistics.mean(scores_list) * 100, 1),
                "correct": sum(1 for s in scores_list if s >= 0.5),
                "total": len(scores_list),
            }

    by_subtask: dict[str, dict[str, Any]] = {}
    for subtask, scores_list in subtask_scores.items():
        if scores_list:
            by_subtask[subtask] = {
                "accuracy": round(statistics.mean(scores_list) * 100, 1),
                "correct": sum(1 for s in scores_list if s >= 0.5),
                "total": len(scores_list),
            }

    all_scores = [s for sl in category_scores.values() for s in sl]
    overall = round(statistics.mean(all_scores) * 100, 1) if all_scores else 0.0

    results = {
        "overall_accuracy": overall,
        "by_category": by_category,
        "by_subtask": by_subtask,
        "timing": {
            "total_ingest_s": round(total_ingest_time, 1),
            "total_recall_s": round(total_recall_time, 1),
            "avg_recall_ms": round(total_recall_time / max(total_questions, 1) * 1000, 1),
            "total_chunks": total_chunks,
            "total_questions": total_questions,
        },
        "config": {
            "categories": categories,
            "answer_model": answer_model,
            "chunk_size": chunk_size,
            "context_max_tokens": context_max_tokens,
            "recall_budget": recall_budget,
            "max_examples": max_examples,
            "max_questions": max_questions,
        },
        "details": all_details,
    }

    return results


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


def print_mab(results: dict[str, Any]) -> None:
    """Print formatted MAB benchmark results."""
    config = results["config"]
    timing = results["timing"]

    print("\nMemoryAgentBench Results — Attestor")
    print("=" * 55)
    print(f"Categories: {', '.join(config['categories'])}")
    print(f"Answer model: {config['answer_model']}")
    print(f"Chunk size: {config['chunk_size']} tokens")
    print(f"Questions evaluated: {timing['total_questions']}")

    print(f"\nOverall Accuracy: {results['overall_accuracy']}%")

    # By category
    print("\nBy Category:")
    print(f"  {'Category':>10s} {'Accuracy':>10s} {'Correct':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    for cat, info in results["by_category"].items():
        print(
            f"  {cat:>10s} {info['accuracy']:>9.1f}% "
            f"  {info['correct']}/{info['total']} "
        )

    # By subtask
    print("\nBy Sub-task:")
    print(f"  {'Sub-task':<25s} {'Accuracy':>10s} {'Questions':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    for subtask, info in results["by_subtask"].items():
        print(
            f"  {subtask:<25s} {info['accuracy']:>9.1f}% "
            f"  {info['total']:>8d}"
        )

    # Timing
    print("\nTiming:")
    print(f"  Ingest:  {timing['total_ingest_s']}s ({timing['total_chunks']} chunks)")
    print(f"  Recall:  {timing['total_recall_s']}s (avg {timing['avg_recall_ms']:.0f}ms/query)")

    # Reference scores from paper
    print("\nCompare with:")
    print("  GPT-4o (long-ctx):  AR ~83%, CR ~60%  (MAB paper)")
    print("  Mem0:               limited on dense tasks  (MAB paper)")
    print("  MemGPT:             AR ~50-70%  (MAB paper)")
