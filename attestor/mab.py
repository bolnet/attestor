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

import logging
import os
import re
import statistics
import string
import tempfile
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from attestor.core import AgentMemory

logger = logging.getLogger("attestor")
from attestor.utils.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Benchmark embedding upgrade
# ---------------------------------------------------------------------------


def _upgrade_embeddings_for_benchmark(mem: AgentMemory) -> None:
    """Ensure benchmarks use OpenAI text-embedding-3-small via OpenRouter.

    Benchmarks always use OpenRouter for embeddings — no fallback to local models.
    Works for all backends (Postgres/pgvector, ArangoDB, Azure, AWS, GCP).

    Raises RuntimeError if OPENROUTER_API_KEY is not set.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY required for benchmarks. "
            "Set it in .env or environment."
        )

    vector_store = mem._vector_store
    if vector_store and hasattr(vector_store, "_ensure_embedding_fn"):
        # Reset so it re-initializes with OpenRouter key
        vector_store._embedding_fn = None
        vector_store._ensure_embedding_fn()
        if hasattr(vector_store, "_openai_client") or hasattr(vector_store, "_embedder"):
            logger.info("Benchmark: using upgraded embeddings via OpenRouter")
        else:
            raise RuntimeError(
                "Backend failed to initialize OpenAI embeddings despite "
                "OPENROUTER_API_KEY being set. Check openai package."
            )
    elif vector_store:
        logger.warning("Benchmark: vector store has no known embedding upgrade path")


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

DEFAULT_CHUNK_SIZE = 1024

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


def token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
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


def binary_recall(prediction: str, answer_elements: List[str]) -> int:
    """Return 1 if ALL answer elements are present in prediction, 0 otherwise."""
    if not answer_elements:
        return 1

    pred_norm = normalize_answer(prediction)
    for element in answer_elements:
        if normalize_answer(element) not in pred_norm:
            return 0
    return 1


def ruler_recall(prediction: str, answer_elements: List[str]) -> float:
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
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into chunks bounded by token estimate.

    Strategy: split on sentence boundaries (regex), fall back to newlines,
    then word boundaries.
    """
    if not text:
        return [""]

    est = estimate_tokens(text)
    if est <= chunk_size:
        return [text]

    # Try sentence splitting first
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) > 1:
        return _merge_splits(sentences, chunk_size)

    # Fallback: split on newlines
    lines = text.split("\n")
    if len(lines) > 1:
        return _merge_splits(lines, chunk_size)

    # Last resort: split on word boundaries
    words = text.split()
    max_words = int(chunk_size / 1.3)
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def chunk_text_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: Optional[int] = None,
) -> List[str]:
    """Split text into overlapping chunks for better boundary coverage.

    Uses sentence-level splitting with overlap carried between chunks.
    Falls back to newlines, then word boundaries with stride.
    """
    if overlap is None:
        overlap = chunk_size // 4

    if not text:
        return [""]

    est = estimate_tokens(text)
    if est <= chunk_size:
        return [text]

    # Try sentence splitting first
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) > 1:
        return _merge_splits_overlap(sentences, chunk_size, overlap)

    # Fallback: split on newlines
    lines = text.split("\n")
    if len(lines) > 1:
        return _merge_splits_overlap(lines, chunk_size, overlap)

    # Last resort: word boundaries with stride
    words = text.split()
    max_words = int(chunk_size / 1.3)
    stride = max(1, int((chunk_size - overlap) / 1.3))
    chunks = []
    for i in range(0, len(words), stride):
        chunks.append(" ".join(words[i : i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks


def _merge_splits_overlap(
    splits: List[str], chunk_size: int, overlap: int
) -> List[str]:
    """Merge splits into chunks with overlap between consecutive chunks."""
    chunks = []
    current: List[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = estimate_tokens(split)
        if current_tokens + split_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            # Carry last N splits as overlap into next chunk
            overlap_splits: List[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                st = estimate_tokens(s)
                if overlap_tokens + st > overlap:
                    break
                overlap_splits.insert(0, s)
                overlap_tokens += st
            current = list(overlap_splits) + [split]
            current_tokens = overlap_tokens + split_tokens
        else:
            current.append(split)
            current_tokens += split_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _merge_splits(splits: List[str], chunk_size: int) -> List[str]:
    """Merge small splits into chunks that fit within the token budget."""
    chunks = []
    current: List[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = estimate_tokens(split)
        if current_tokens + split_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            current = [split]
            current_tokens = split_tokens
        else:
            current.append(split)
            current_tokens += split_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mab(
    categories: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
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

    result: Dict[str, List[Dict[str, Any]]] = {}

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
# Ingestion
# ---------------------------------------------------------------------------


def ingest_context(
    mem: AgentMemory,
    context: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_max_tokens: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Chunk and ingest a MAB context into memory.

    Returns (num_chunks, num_tokens_ingested).
    """
    # Optional context truncation for quick testing
    if context_max_tokens:
        words = context.split()
        max_words = int(context_max_tokens / 1.3)
        if len(words) > max_words:
            context = " ".join(words[:max_words])
            if verbose:
                print(f"    Truncated context to ~{context_max_tokens} tokens")

    chunks = chunk_text_overlap(context, chunk_size=chunk_size)
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk)
        total_tokens += tokens
        mem.add(
            content=chunk,
            tags=[f"chunk_{i}"],
            category="document",
            metadata={"chunk_index": i},
        )

    return len(chunks), total_tokens


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

MAB_ANSWER_PROMPT = """You are answering questions about a document. Below are relevant passages retrieved from memory. Use ONLY this information.

Passages:
{context}

Question: {question}

Instructions:
- Answer directly and concisely
- Include specific details like names, dates, numbers when available
- If the information isn't available in the passages, say "I don't know"
- Do NOT add information not found in the passages
- Keep your answer as brief as possible (ideally one sentence or a few words)"""

MAB_EXACT_PROMPT = """Answer the question using ONLY the passages below.

Passages (in document order — later passages OVERRIDE earlier ones when facts conflict):
{context}

Question: {question}

RULES:
1. CONFLICTS: If multiple passages state different facts about the same thing, ALWAYS use the LAST passage (the one appearing latest in the document). Earlier facts are outdated.
2. CHAIN: For multi-hop questions, follow the chain step by step through the passages.
3. FORMAT: Reply with ONLY the answer — a single entity name, number, or short phrase. No explanations.
4. If unknown, reply: I don't know

Good answers: "Belgium", "Charles Dickens", "42", "rugby", "United Kingdom"
Bad answers: "The answer is Belgium", "Based on passage 3, it is rugby"
4. If you cannot determine the answer, respond with exactly: I don't know

Answer:"""

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


def _rerank_results(results, question: str):
    """Re-rank retrieval results by keyword overlap with question."""
    # Extract keywords from question
    q_words = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
    # Also extract capitalized words (proper nouns)
    q_proper = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question))
    q_proper_lower = {w.lower() for w in q_proper}

    for r in results:
        content_lower = r.memory.content.lower()
        # Proper noun hits get extra weight
        if q_proper_lower:
            proper_hits = sum(1 for pn in q_proper_lower if pn in content_lower)
            proper_density = proper_hits / len(q_proper_lower)
            r.score += proper_density * 0.3

    return results


def _extract_entities_from_context(text: str) -> List[str]:
    """Extract capitalized proper nouns from context text."""
    patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+(?:of|the|de|von|van)\s+)?(?:[A-Z][a-z]+)*)\b', text)
    seen: set = set()
    entities: List[str] = []
    skip = {"The", "This", "That", "These", "Those", "There", "What",
            "When", "Where", "Which", "Who", "How", "Answer", "Question",
            "Based", "According", "Passages", "Yes", "No", "None",
            "However", "Therefore", "Furthermore", "Moreover", "Also",
            "Some", "Many", "Most", "All", "Each", "Every", "Any"}
    for ent in patterns:
        ent = ent.strip()
        if ent not in seen and ent not in skip and len(ent) > 2:
            seen.add(ent)
            entities.append(ent)
    return entities[:10]


def multi_hop_recall(
    mem: AgentMemory,
    question: str,
    budget: int = 8000,
) -> List:
    """Two-round retrieval for multi-hop questions.

    Round 1: Retrieve on original question.
    Round 2: Extract entities from round-1 results, retrieve on each.
    Merge and deduplicate both rounds.
    """
    from attestor.retrieval.scorer import fit_to_budget

    # Round 1: standard recall
    round1 = mem.recall(question, budget=budget)

    # Extract entities from round-1 context
    round1_text = " ".join(r.memory.content for r in round1)
    entities = _extract_entities_from_context(round1_text)

    # Round 2: recall on discovered entities NOT already in the question
    new_entities = [e for e in entities if e.lower() not in question.lower()]

    all_results = list(round1)
    seen_ids = {r.memory.id for r in round1}

    for entity in new_entities[:5]:
        round2 = mem.recall(entity, budget=2000)
        for r in round2:
            if r.memory.id not in seen_ids:
                r.score *= 0.8  # Slightly discount round-2 results
                all_results.append(r)
                seen_ids.add(r.memory.id)

    # Re-sort by score and fit to budget
    all_results.sort(key=lambda r: r.score, reverse=True)
    return fit_to_budget(all_results, budget)


def answer_question(
    mem: AgentMemory,
    question: str,
    budget: int = 6000,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    source: str = "",
) -> str:
    """Recall from memory + LLM synthesis to answer a MAB question.

    ``model`` defaults to ``stack.models.verifier`` from
    ``configs/attestor.yaml`` (the cross-family Anthropic pick that
    complements the gpt-4.1 judge in the canonical lineup).
    """
    if model is None:
        from attestor.config import get_stack
        model = get_stack().models.verifier
    use_exact = _is_exact_source(source)
    effective_budget = _get_budget_for_source(source, budget)

    # Multi-hop retrieval for multi-hop exact-match tasks
    if use_exact and ("mh" in source or "memory_merging" in source):
        results = multi_hop_recall(mem, question, budget=effective_budget)
    else:
        results = mem.recall(question, budget=effective_budget)

    # Re-rank by question keyword overlap
    if results:
        results = _rerank_results(results, question)

    if not results:
        return "I don't know"

    # For exact-match tasks, preserve document order (chunk_0, chunk_1, ...)
    # so the model sees later/overriding facts last.
    # For other tasks, sort by relevance score.
    if use_exact:
        def _chunk_order(r):
            for tag in (r.memory.tags or []):
                if tag.startswith("chunk_"):
                    try:
                        return int(tag.split("_")[1])
                    except (ValueError, IndexError):
                        pass
            return 999999
        results.sort(key=_chunk_order)
    else:
        results.sort(key=lambda r: r.score, reverse=True)

    context = "\n".join(r.memory.content for r in results)

    try:
        from openai import OpenAI
    except ImportError:
        return context  # Fallback: raw context

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return context  # No API key available

    prompt_template = MAB_EXACT_PROMPT if use_exact else MAB_ANSWER_PROMPT
    prompt = prompt_template.format(context=context, question=question)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    # Retry with exponential backoff on rate limits
    import time as _time
    from attestor.llm_trace import traced_create
    for attempt in range(5):
        try:
            response = traced_create(
                client,
                role="mab.chat",
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                _time.sleep(wait)
            else:
                raise
    else:
        return "I don't know"  # All retries exhausted

    raw = response.choices[0].message.content

    if use_exact:
        return _extract_answer(raw)
    return raw


# ---------------------------------------------------------------------------
# Scoring router
# ---------------------------------------------------------------------------


def _flatten_answers(answers) -> List[str]:
    """Flatten answers to a single list of strings."""
    if not answers:
        return []
    if isinstance(answers[0], list):
        return [a for sublist in answers for a in sublist]
    return list(answers)


def score_question(
    prediction: str,
    answers: List,
    source: str,
) -> Dict[str, float]:
    """Score a single prediction based on the sub-task's primary metric."""
    scores: Dict[str, float] = {}

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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_mab(
    categories: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    max_questions: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_max_tokens: Optional[int] = None,
    answer_model: Optional[str] = None,
    recall_budget: int = 6000,
    verbose: bool = False,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    skip_examples: int = 0,
    backend_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    category_scores: Dict[str, List[float]] = defaultdict(list)
    subtask_scores: Dict[str, List[float]] = defaultdict(list)
    all_details: List[Dict[str, Any]] = []
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
                # Use OpenAI embeddings for benchmarking if key available
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
    by_category: Dict[str, Dict[str, Any]] = {}
    for cat, scores_list in category_scores.items():
        if scores_list:
            by_category[cat] = {
                "accuracy": round(statistics.mean(scores_list) * 100, 1),
                "correct": sum(1 for s in scores_list if s >= 0.5),
                "total": len(scores_list),
            }

    by_subtask: Dict[str, Dict[str, Any]] = {}
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


def print_mab(results: Dict[str, Any]) -> None:
    """Print formatted MAB benchmark results."""
    config = results["config"]
    timing = results["timing"]

    print(f"\nMemoryAgentBench Results — Attestor")
    print("=" * 55)
    print(f"Categories: {', '.join(config['categories'])}")
    print(f"Answer model: {config['answer_model']}")
    print(f"Chunk size: {config['chunk_size']} tokens")
    print(f"Questions evaluated: {timing['total_questions']}")

    print(f"\nOverall Accuracy: {results['overall_accuracy']}%")

    # By category
    print(f"\nBy Category:")
    print(f"  {'Category':>10s} {'Accuracy':>10s} {'Correct':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    for cat, info in results["by_category"].items():
        print(
            f"  {cat:>10s} {info['accuracy']:>9.1f}% "
            f"  {info['correct']}/{info['total']} "
        )

    # By subtask
    print(f"\nBy Sub-task:")
    print(f"  {'Sub-task':<25s} {'Accuracy':>10s} {'Questions':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    for subtask, info in results["by_subtask"].items():
        print(
            f"  {subtask:<25s} {info['accuracy']:>9.1f}% "
            f"  {info['total']:>8d}"
        )

    # Timing
    print(f"\nTiming:")
    print(f"  Ingest:  {timing['total_ingest_s']}s ({timing['total_chunks']} chunks)")
    print(f"  Recall:  {timing['total_recall_s']}s (avg {timing['avg_recall_ms']:.0f}ms/query)")

    # Reference scores from paper
    print(f"\nCompare with:")
    print(f"  GPT-4o (long-ctx):  AR ~83%, CR ~60%  (MAB paper)")
    print(f"  Mem0:               limited on dense tasks  (MAB paper)")
    print(f"  MemGPT:             AR ~50-70%  (MAB paper)")
