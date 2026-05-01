"""MemoryAgentBench (MAB) benchmark runner for Attestor.

MemoryAgentBench (ICLR 2026) evaluates memory systems across 4 competencies:
Accurate Retrieval, Test-Time Learning, Long-Range Understanding,
and Conflict Resolution. 146 examples, 60-100 questions each.

Usage:
    attestor mab                               # Run AR + CR (default)
    attestor mab --categories AR CR TTL LRU    # Run specific categories
    attestor mab --max-examples 2              # Quick test
    attestor mab --chunk-size 2048             # Smaller chunks

Requires: poetry add datasets openai  (not included in attestor base deps)
"""

from __future__ import annotations

from attestor.mab.retrieval import (
    DEFAULT_CHUNK_SIZE,
    MAB_ANSWER_PROMPT,
    MAB_EXACT_PROMPT,
    _extract_entities_from_context,
    _merge_splits,
    _merge_splits_overlap,
    _rerank_results,
    _upgrade_embeddings_for_benchmark,
    answer_question,
    chunk_text,
    chunk_text_overlap,
    ingest_context,
    multi_hop_recall,
)
from attestor.mab.runner import (
    CATEGORY_TO_SPLIT,
    HF_DATASET,
    SPLIT_MAP,
    load_mab,
    print_mab,
    run_mab,
)
from attestor.mab.scoring import (
    _EXACT_MATCH_SOURCES,
    _extract_answer,
    _flatten_answers,
    _get_budget_for_source,
    _is_exact_source,
    binary_recall,
    exact_match,
    max_over_ground_truths,
    normalize_answer,
    ruler_recall,
    score_question,
    substring_exact_match,
    token_f1,
)

__all__ = [
    # Constants
    "DEFAULT_CHUNK_SIZE",
    "HF_DATASET",
    "SPLIT_MAP",
    "CATEGORY_TO_SPLIT",
    "MAB_ANSWER_PROMPT",
    "MAB_EXACT_PROMPT",
    # Scoring metrics
    "normalize_answer",
    "substring_exact_match",
    "exact_match",
    "token_f1",
    "binary_recall",
    "ruler_recall",
    "max_over_ground_truths",
    "score_question",
    # Chunking
    "chunk_text",
    "chunk_text_overlap",
    # Ingestion + retrieval + answer
    "ingest_context",
    "multi_hop_recall",
    "answer_question",
    # Dataset + runner + printer
    "load_mab",
    "run_mab",
    "print_mab",
    # Embedding upgrade helper
    "_upgrade_embeddings_for_benchmark",
    # Internal helpers (kept public for tests + downstream callers)
    "_EXACT_MATCH_SOURCES",
    "_extract_answer",
    "_flatten_answers",
    "_get_budget_for_source",
    "_is_exact_source",
    "_extract_entities_from_context",
    "_merge_splits",
    "_merge_splits_overlap",
    "_rerank_results",
]
