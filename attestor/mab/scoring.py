"""MAB scoring — metrics, answer extraction, and per-source score routing."""

from __future__ import annotations

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Scoring functions (pure, no external deps)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison: lowercase, remove articles/punct/whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def substring_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized ground truth is a substring of normalized prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction exactly matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1, precision, and recall between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return (0.0, 0.0, 0.0)

    # Special handling for yes/no
    if set(pred_tokens) <= {"yes", "no"} or set(gt_tokens) <= {"yes", "no"}:
        if pred_tokens == gt_tokens:
            return (1.0, 1.0, 1.0)
        return (0.0, 0.0, 0.0)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return (0.0, 0.0, 0.0)

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return (f1, precision, recall)


def binary_recall(prediction: str, answer_elements: list[str]) -> int:
    """Return 1 if ALL answer elements are present in prediction, 0 otherwise."""
    if not answer_elements:
        return 1

    pred_norm = normalize_answer(prediction)
    for element in answer_elements:
        if normalize_answer(element) not in pred_norm:
            return 0
    return 1


def ruler_recall(prediction: str, answer_elements: list[str]) -> float:
    """Return fraction of answer elements found in prediction."""
    if not answer_elements:
        return 0.0

    pred_norm = normalize_answer(prediction)
    found = sum(1 for el in answer_elements if normalize_answer(el) in pred_norm)
    return found / len(answer_elements)


def max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    """Compute max metric score over all valid ground truth answers.

    Handles: single string, list of strings, list of lists.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    if not ground_truths:
        return 0.0

    scores = []
    for gt in ground_truths:
        if isinstance(gt, list):
            for g in gt:
                result = metric_fn(prediction, g)
                if isinstance(result, tuple):
                    scores.append(result[0])
                elif isinstance(result, bool):
                    scores.append(float(result))
                else:
                    scores.append(float(result))
        else:
            result = metric_fn(prediction, gt)
            if isinstance(result, tuple):
                scores.append(result[0])
            elif isinstance(result, bool):
                scores.append(float(result))
            else:
                scores.append(float(result))

    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Source routing — exact-match sources & per-source budget hints
# ---------------------------------------------------------------------------

# Sources that use exact_match scoring
_EXACT_MATCH_SOURCES = frozenset([
    "factconsolidation", "memory_merging", "icl", "detective",
])


def _extract_answer(text: str) -> str:
    """Extract the core answer from LLM response, stripping reasoning."""
    text = text.strip()

    # Check for **bold** answer — take the last bolded phrase
    bold = re.findall(r'\*\*([^*]+)\*\*', text)
    if bold:
        return bold[-1].strip()

    # Check for "Answer: X" pattern
    ans_match = re.search(
        r'(?:^|\n)\s*(?:answer|ans)[:\s]+(.+?)(?:\.|$)',
        text, re.IGNORECASE,
    )
    if ans_match:
        return ans_match.group(1).strip()

    # If short enough, return as-is
    if len(text.split()) <= 10:
        return text.strip()

    # Last non-empty line might be the answer
    lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r'^(?:answer|ans)[:\s]*', '', last, flags=re.IGNORECASE)
        last = re.sub(r'^\*+|\*+$', '', last).strip()
        if len(last.split()) <= 10:
            return last

    return text


def _is_exact_source(source: str) -> bool:
    """Check if this source uses exact_match scoring."""
    return any(k in source for k in _EXACT_MATCH_SOURCES)


def _get_budget_for_source(source: str, default: int) -> int:
    """Adjust recall budget based on task type."""
    if "mh" in source or "memory_merging" in source:
        return max(default, 8000)  # Multi-hop needs more context
    if "eventqa" in source:
        return max(default, 6000)  # Event QA needs more recall
    return default


# ---------------------------------------------------------------------------
# Scoring router
# ---------------------------------------------------------------------------


def _flatten_answers(answers) -> list[str]:
    """Flatten answers to a single list of strings."""
    if not answers:
        return []
    if isinstance(answers[0], list):
        return [a for sublist in answers for a in sublist]
    return list(answers)


def score_question(
    prediction: str,
    answers: list,
    source: str,
) -> dict[str, float]:
    """Score a single prediction based on the sub-task's primary metric."""
    scores: dict[str, float] = {}

    if "ruler_qa" in source:
        scores["substring_exact_match"] = max_over_ground_truths(
            substring_exact_match, prediction, answers
        )
    elif "ruler_niah" in source:
        flat = _flatten_answers(answers)
        scores["ruler_recall"] = ruler_recall(prediction, flat)
    elif "eventqa" in source:
        flat = _flatten_answers(answers)
        scores["binary_recall"] = float(binary_recall(prediction, flat))
    elif "factconsolidation" in source or "memory_merging" in source:
        scores["exact_match"] = max_over_ground_truths(
            exact_match, prediction, answers
        )
    elif "icl" in source:
        scores["exact_match"] = max_over_ground_truths(
            exact_match, prediction, answers
        )
    elif "detective_qa" in source:
        scores["exact_match"] = max_over_ground_truths(
            exact_match, prediction, answers
        )
    elif "longmemeval" in source:
        # LLM-judge fallback: use substring_exact_match
        scores["substring_exact_match"] = max_over_ground_truths(
            substring_exact_match, prediction, answers
        )
    elif "infbench" in source:
        # LLM-judge fallback: use token F1
        scores["token_f1"] = max_over_ground_truths(
            lambda p, g: token_f1(p, g)[0], prediction, answers
        )
    elif "recsys" in source:
        # Recommendation recall — check if any answer appears in prediction
        scores["substring_exact_match"] = max_over_ground_truths(
            substring_exact_match, prediction, answers
        )
    else:
        scores["substring_exact_match"] = max_over_ground_truths(
            substring_exact_match, prediction, answers
        )

    return scores
