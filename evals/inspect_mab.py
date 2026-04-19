"""Debug inspector: run the same pipeline as the Braintrust eval, but print
per-row evidence locally so we can see where scores are coming from.

Usage:
    set -a && source .env && set +a
    .venv/bin/python evals/inspect_mab.py --categories AR --max-examples 1 --max-questions 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap

from anthropic import Anthropic

from attestor import AgentMemory
from attestor.mab import (
    DEFAULT_CHUNK_SIZE,
    _upgrade_embeddings_for_benchmark,
    answer_question,
    ingest_context,
    load_mab,
)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-6")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-4.1-mini")

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


def judge(client: Anthropic, question: str, reference: list[str], prediction: str) -> dict:
    ref_block = "\n".join(f"- {r}" for r in reference)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=ref_block, prediction=prediction
        )}],
    )
    text = msg.content[0].text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {"score": 0.0, "reason": f"parse_error raw={text[:200]}"}


def recalled_context(mem: AgentMemory, question: str, budget: int = 6000) -> list[str]:
    """Peek at what recall actually returned for this question."""
    results = mem.recall(question, budget=budget)
    return [r.memory.content for r in results[:5]]


def wrap(s: str, width: int = 100) -> str:
    if s is None:
        return "<None>"
    s = str(s)
    if len(s) > 400:
        s = s[:400] + " …"
    return textwrap.fill(s, width=width, replace_whitespace=False)


def run(categories: list[str], max_examples: int, max_questions: int) -> None:
    data = load_mab(categories=categories, max_examples=max_examples)
    client = Anthropic()

    total_score = 0.0
    n = 0

    for cat, examples in data.items():
        for ex_idx, example in enumerate(examples):
            source = example["metadata"].get("source", "unknown")
            print(f"\n{'='*100}")
            print(f"CATEGORY={cat}  EXAMPLE={ex_idx}  SOURCE={source}")
            print(f"CONTEXT LEN (chars) = {len(example['context'])}")
            print(f"{'='*100}")

            with tempfile.TemporaryDirectory() as tmpdir:
                mem = AgentMemory(tmpdir)
                _upgrade_embeddings_for_benchmark(mem)
                mem._retrieval.enable_temporal_boost = False

                n_chunks, n_tokens = ingest_context(
                    mem, example["context"],
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    context_max_tokens=None,
                    verbose=False,
                )
                print(f"Ingested: {n_chunks} chunks, ~{n_tokens} tokens\n")

                questions = example["questions"][:max_questions]
                answers = example["answers"][:max_questions]

                for i, (q, a) in enumerate(zip(questions, answers)):
                    ref = a if isinstance(a, list) else [a]
                    print(f"\n--- Q{i+1} ---")
                    print(f"QUESTION:   {wrap(q)}")
                    print(f"REFERENCE:  {ref}")

                    retrieved = recalled_context(mem, q)
                    print(f"RECALL (top {len(retrieved)} chunks):")
                    for j, chunk in enumerate(retrieved):
                        print(f"  [{j}] {wrap(chunk, 90)}")

                    prediction = answer_question(
                        mem, q, budget=6000, model=ANSWER_MODEL, source=source,
                    )
                    print(f"PREDICTION: {wrap(prediction)}")

                    verdict = judge(client, q, ref, prediction)
                    score = float(verdict.get("score", 0.0))
                    reason = verdict.get("reason", "")
                    print(f"JUDGE:      score={score}  reason={reason}")

                    total_score += score
                    n += 1

    print(f"\n{'='*100}")
    print(f"TOTAL: {n} questions, mean score = {total_score/n if n else 0:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["AR"])
    p.add_argument("--max-examples", type=int, default=1)
    p.add_argument("--max-questions", type=int, default=3)
    args = p.parse_args()
    run(args.categories, args.max_examples, args.max_questions)
