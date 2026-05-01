"""LoCoMo benchmark runner — split from a single 854-line module on 2026-05-01.

Re-exports the original public surface so callers using
``from attestor.locomo import X`` keep working unchanged.
"""

from __future__ import annotations

from attestor.locomo.judge import JUDGE_PROMPT, judge_answer
from attestor.locomo.reflection import (
    ANSWER_PROMPT,
    REFLECTION_PROMPT,
    RESOLVE_PROMPT,
    _build_context,
    _extract_names_from_question,
    _resolve_coreferences,
    answer_question,
)
from attestor.locomo.runner import (
    CATEGORY_NAMES,
    DEFAULT_MODEL,
    LOCOMO_URL,
    _chat,
    _default_model,
    _guess_entity_type,
    _resolve_client,
    download_locomo,
    ingest_conversation,
    load_locomo,
    print_locomo,
    run_locomo,
)

__all__ = [
    # Constants
    "ANSWER_PROMPT",
    "CATEGORY_NAMES",
    "DEFAULT_MODEL",
    "JUDGE_PROMPT",
    "LOCOMO_URL",
    "REFLECTION_PROMPT",
    "RESOLVE_PROMPT",
    # Public functions
    "answer_question",
    "download_locomo",
    "ingest_conversation",
    "judge_answer",
    "load_locomo",
    "print_locomo",
    "run_locomo",
    # Internal helpers (kept exported for parity with the old single-file
    # module; tests / external callers may have imported these by name).
    "_build_context",
    "_chat",
    "_default_model",
    "_extract_names_from_question",
    "_guess_entity_type",
    "_resolve_client",
    "_resolve_coreferences",
]
