"""LongMemEval benchmark runner for Attestor.

LongMemEval (Wu et al., ICLR 2025) evaluates long-term memory of chat
assistants across five abilities: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and abstention. The
cleaned 500-question release splits those into six ``question_type``
categories in the dataset.

This package is the Attestor-native runner — split out of a single
2466-line ``attestor/longmemeval.py`` module while keeping every
public symbol importable from ``attestor.longmemeval`` for back-compat.

    Sample = {
        "question_id": str,
        "question_type": str,          # one of CATEGORY_NAMES keys
        "question": str,
        "question_date": str,          # "YYYY/MM/DD (DayOfWeek) HH:MM"
        "answer": str,                 # gold
        "answer_session_ids": list[str],
        "haystack_dates": list[str],   # per-session timestamps
        "haystack_session_ids": list[str],
        "haystack_sessions": list[list[{"role": str, "content": str}]],
    }

Dataset source (HuggingFace):
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

from __future__ import annotations

# Submodule import order matters: ``runner`` does ``import attestor.longmemeval
# as _pkg`` at call time and looks up ``_pkg.judge_answer`` etc. — so the
# judge / fixtures / reporter symbols must already be attached to the
# package by the time anyone executes a runner function.

from attestor.longmemeval.fixtures import (
    CATEGORY_NAMES,
    DATASET_VARIANTS,
    HF_BASE_URL,
    AnswerResult,
    DistilledFact,
    IngestStats,
    LMESample,
    LMETurn,
    TEMPORAL_CATEGORY,
    _MODE_RE,
    _REASONING_RE,
    _chat,
    _coerce_sample,
    _coerce_turn,
    _extract_json_array,
    _extract_retrieved_session_ids,
    _fact_from_record,
    _format_recall_context,
    _format_turn_content,
    _get_client,
    _get_client_for_model,
    _iso_date,
    _normalize_claim_type,
    _normalize_emphasis,
    _normalize_speaker,
    _normalize_str_list,
    _parse_distilled,
    _parse_predicted_mode,
    _resolve_variant,
    _short_date,
    _strip_reasoning,
    distill_turn,
    download_longmemeval,
    load_longmemeval,
    load_or_download,
    namespace_for,
    parse_lme_date,
)
from attestor.longmemeval.prompts import (
    ANSWER_PROMPT,
    DISTILL_PROMPT,
    JUDGE_PROMPT,
    PERSONALIZATION_JUDGE_PROMPT,
    PERSONALIZATION_PROMPT,
    VERIFY_PROMPT,
)
from attestor.longmemeval.judge import (
    DEFAULT_JUDGES,
    DEFAULT_MODEL,
    DEFAULT_PARALLEL,
    JudgeResult,
    _JUDGE_CONCURRENCY,
    _LABEL_FALLBACK_RE,
    _default_judges,
    _default_model,
    _judgement_to_dict,
    _parse_judge_response,
    _safe_judge_dict,
    judge_answer,
    judge_personalization,
)
from attestor.longmemeval.reporter import (
    LMERunReport,
    RunProvenance,
    SampleReport,
    _accuracy,
    _attestor_version,
    _blank_counter,
    _git_sha,
    _sha256_file,
    _sha256_str,
    _summarize,
    _summarize_dimensions,
)
from attestor.longmemeval.runner import (
    _answerer_call,
    _process_sample,
    answer_question,
    classify_question,
    ingest_history,
    is_recommendation_question,
    run,
    run_async,
)

__all__ = [
    # Constants
    "ANSWER_PROMPT",
    "CATEGORY_NAMES",
    "DATASET_VARIANTS",
    "DEFAULT_JUDGES",
    "DEFAULT_MODEL",
    "DEFAULT_PARALLEL",
    "DISTILL_PROMPT",
    "HF_BASE_URL",
    "JUDGE_PROMPT",
    "PERSONALIZATION_JUDGE_PROMPT",
    "PERSONALIZATION_PROMPT",
    "TEMPORAL_CATEGORY",
    "VERIFY_PROMPT",
    # Dataclasses
    "AnswerResult",
    "DistilledFact",
    "IngestStats",
    "JudgeResult",
    "LMERunReport",
    "LMESample",
    "LMETurn",
    "RunProvenance",
    "SampleReport",
    # Public functions
    "answer_question",
    "classify_question",
    "distill_turn",
    "download_longmemeval",
    "ingest_history",
    "is_recommendation_question",
    "judge_answer",
    "judge_personalization",
    "load_longmemeval",
    "load_or_download",
    "namespace_for",
    "parse_lme_date",
    "run",
    "run_async",
    # Internal symbols re-exported for tests/benchmarks
    "_JUDGE_CONCURRENCY",
    "_LABEL_FALLBACK_RE",
    "_MODE_RE",
    "_REASONING_RE",
    "_accuracy",
    "_answerer_call",
    "_attestor_version",
    "_blank_counter",
    "_chat",
    "_coerce_sample",
    "_coerce_turn",
    "_default_judges",
    "_default_model",
    "_extract_retrieved_session_ids",
    "_format_recall_context",
    "_format_turn_content",
    "_get_client",
    "_get_client_for_model",
    "_git_sha",
    "_iso_date",
    "_judgement_to_dict",
    "_parse_distilled",
    "_parse_judge_response",
    "_parse_predicted_mode",
    "_process_sample",
    "_resolve_variant",
    "_safe_judge_dict",
    "_sha256_file",
    "_sha256_str",
    "_short_date",
    "_strip_reasoning",
    "_summarize",
    "_summarize_dimensions",
]
