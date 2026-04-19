"""Braintrust eval: attestor recall pipeline on MAB, judged by Claude Opus 4.6.

Run:
    set -a && source .env && set +a
    .venv/bin/python evals/braintrust_mab.py --categories AR --max-examples 1 --max-questions 3

Or via braintrust CLI:
    .venv/bin/braintrust eval evals/braintrust_mab.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

from braintrust import Eval, init_logger
from anthropic import Anthropic

from attestor import AgentMemory
from attestor.mab import (
    DEFAULT_CHUNK_SIZE,
    _upgrade_embeddings_for_benchmark,
    answer_question,
    ingest_context,
    load_mab,
)

PROJECT = "attestor-mab"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-6")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-4.1-mini")
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
RECALL_BUDGET = 6000

JUDGE_PROMPT = """You are grading an answer produced by an AI that retrieves from \
a memory store. Decide whether the prediction is factually consistent with the \
reference answer(s).

Question: {question}

Reference answer(s):
{reference}

Predicted answer:
{prediction}

Reply with a single JSON object on one line:
{{"score": 0.0 to 1.0, "reason": "short rationale"}}
- 1.0 = fully correct and entailed by the reference
- 0.5 = partially correct / missing detail
- 0.0 = incorrect, contradictory, or "I don't know"
Do not output anything outside the JSON object."""


def build_dataset(
    categories: list[str],
    max_examples: int,
    max_questions: int,
) -> list[dict[str, Any]]:
    """Flatten MAB into per-question rows with precomputed predictions."""
    data = load_mab(categories=categories, max_examples=max_examples)
    rows: list[dict[str, Any]] = []

    for cat, examples in data.items():
        for ex_idx, example in enumerate(examples):
            source = example["metadata"].get("source", "unknown")
            with tempfile.TemporaryDirectory() as tmpdir:
                mem = AgentMemory(tmpdir)
                _upgrade_embeddings_for_benchmark(mem)
                mem._retrieval.enable_temporal_boost = False

                ingest_context(
                    mem,
                    example["context"],
                    chunk_size=CHUNK_SIZE,
                    context_max_tokens=None,
                    verbose=False,
                )

                questions = example["questions"][:max_questions]
                answers = example["answers"][:max_questions]
                for q, a in zip(questions, answers):
                    prediction = answer_question(
                        mem,
                        q,
                        budget=RECALL_BUDGET,
                        model=ANSWER_MODEL,
                        source=source,
                    )
                    rows.append(
                        {
                            "input": {
                                "question": q,
                                "category": cat,
                                "source": source,
                                "example_idx": ex_idx,
                            },
                            "expected": a if isinstance(a, list) else [a],
                            "metadata": {"category": cat, "source": source},
                            "prediction": prediction,
                        }
                    )
    return rows


def task(row_input: dict[str, Any], *, _rows_by_q: dict[str, str]) -> str:
    """Return the precomputed prediction for this question."""
    return _rows_by_q[row_input["question"]]


def llm_judge(
    client: Anthropic,
    question: str,
    reference: list[str],
    prediction: str,
) -> dict[str, Any]:
    ref_block = "\n".join(f"- {r}" for r in reference)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    reference=ref_block,
                    prediction=prediction,
                ),
            }
        ],
    )
    import json
    text = msg.content[0].text.strip()
    try:
        parsed = json.loads(text)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "metadata": {"judge_reason": parsed.get("reason", "")}}
    except Exception as exc:
        return {"score": 0.0, "metadata": {"judge_error": str(exc), "raw": text[:500]}}


def run(categories: list[str], max_examples: int, max_questions: int) -> None:
    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise SystemExit("BRAINTRUST_API_KEY not set. Run: set -a && source .env && set +a")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")

    init_logger(project=PROJECT)
    anthropic = Anthropic()

    rows = build_dataset(categories, max_examples, max_questions)
    rows_by_q = {r["input"]["question"]: r["prediction"] for r in rows}
    dataset = [
        {"input": r["input"], "expected": r["expected"], "metadata": r["metadata"]}
        for r in rows
    ]

    def factuality_scorer(input, output, expected, metadata):
        res = llm_judge(anthropic, input["question"], expected, output)
        return {"name": "factuality_opus46", "score": res["score"], "metadata": res["metadata"]}

    Eval(
        PROJECT,
        data=lambda: dataset,
        task=lambda row_input: rows_by_q[row_input["question"]],
        scores=[factuality_scorer],
        experiment_name=f"opus46-judge-{'-'.join(categories)}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["AR"])
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=3)
    args = parser.parse_args()
    run(args.categories, args.max_examples, args.max_questions)
