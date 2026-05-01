"""LoCoMo judge — LLM-as-judge correctness verdict for generated answers."""

from __future__ import annotations

import json
from typing import Any

from attestor.locomo.runner import DEFAULT_MODEL, _chat, _resolve_client


JUDGE_PROMPT = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:\n"
    "(1) a question (posed by one user to another user),\n"
    "(2) a 'gold' (ground truth) answer,\n"
    "(3) a generated answer\n\n"
    "The gold answer is usually concise. The generated answer might be longer.\n"
    "Be generous - as long as it touches on the same topic as the gold answer, count it as CORRECT.\n"
    "For time-related questions, accept different date formats if they refer to the same date/period.\n\n"
    "Question: {question}\n"
    "Gold answer: {expected_answer}\n"
    "Generated answer: {ai_response}\n\n"
    "First, provide a short (one sentence) explanation, then finish with CORRECT or WRONG.\n"
    'Return JSON with keys "reasoning" and "label".'
)


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Use LLM to judge if the generated answer is correct."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        ai_response=generated,
    )

    client, clean_model = _resolve_client(model)
    response_text = _chat(client, clean_model, prompt, max_tokens=300)

    try:
        result = json.loads(response_text)
        label = result.get("label", "WRONG").upper()
        reasoning = result.get("reasoning", "")
    except json.JSONDecodeError:
        if "CORRECT" in response_text.upper() and "WRONG" not in response_text.upper():
            label = "CORRECT"
        else:
            label = "WRONG"
        reasoning = response_text

    return {
        "label": label,
        "correct": label == "CORRECT",
        "reasoning": reasoning,
    }
