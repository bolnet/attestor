"""MAB retrieval — chunking, ingestion, multi-hop recall, and answer synthesis."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from attestor.core import AgentMemory
from attestor.mab.scoring import (
    _extract_answer,
    _get_budget_for_source,
    _is_exact_source,
)
from attestor.utils.tokens import estimate_tokens

logger = logging.getLogger("attestor")

# ---------------------------------------------------------------------------
# Benchmark embedding upgrade
# ---------------------------------------------------------------------------


def _upgrade_embeddings_for_benchmark(mem: AgentMemory) -> None:
    """Refresh the vector store's embedding function for a benchmark run.

    Embedding-provider selection is owned by ``configs/attestor.yaml``
    (see ``attestor/store/embeddings.py`` for auto-detect rules:
    Pinecone Inference / Voyage / OpenAI / Azure / Bedrock / Vertex).
    This helper just nudges the active vector store to
    re-initialize its embedder so any provider key set after store
    construction (e.g. via ``--env-file``) is picked up. No provider key
    is read here — the underlying embedder factory handles that.

    Works for all backends (Postgres/pgvector, ArangoDB, Azure, AWS,
    GCP).
    """
    vector_store = mem._vector_store
    if vector_store and hasattr(vector_store, "_ensure_embedding_fn"):
        vector_store._embedding_fn = None
        vector_store._ensure_embedding_fn()
        logger.info("Benchmark: re-initialized vector store embedder")
    elif vector_store:
        logger.warning("Benchmark: vector store has no known embedding upgrade path")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1024


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
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
    overlap: int | None = None,
) -> list[str]:
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
    splits: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Merge splits into chunks with overlap between consecutive chunks."""
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = estimate_tokens(split)
        if current_tokens + split_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            # Carry last N splits as overlap into next chunk
            overlap_splits: list[str] = []
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


def _merge_splits(splits: list[str], chunk_size: int) -> list[str]:
    """Merge small splits into chunks that fit within the token budget."""
    chunks = []
    current: list[str] = []
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
# Ingestion
# ---------------------------------------------------------------------------


def ingest_context(
    mem: AgentMemory,
    context: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_max_tokens: int | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
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


def _rerank_results(results, question: str):
    """Re-rank retrieval results by keyword overlap with question."""
    # Extract keywords from question
    set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
    # Also extract capitalized words (proper nouns)
    q_proper = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question))
    q_proper_lower = {w.lower() for w in q_proper}

    for i, r in enumerate(results):
        content_lower = r.memory.content.lower()
        # Proper noun hits get extra weight
        if q_proper_lower:
            proper_hits = sum(1 for pn in q_proper_lower if pn in content_lower)
            proper_density = proper_hits / len(q_proper_lower)
            results[i] = replace(r, score=r.score + proper_density * 0.3)

    return results


def _extract_entities_from_context(text: str) -> list[str]:
    """Extract capitalized proper nouns from context text."""
    patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+(?:of|the|de|von|van)\s+)?(?:[A-Z][a-z]+)*)\b', text)
    seen: set = set()
    entities: list[str] = []
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
) -> list:
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
                # Slightly discount round-2 results
                all_results.append(replace(r, score=r.score * 0.8))
                seen_ids.add(r.memory.id)

    # Re-sort by score and fit to budget
    all_results.sort(key=lambda r: r.score, reverse=True)
    return fit_to_budget(all_results, budget)


def answer_question(
    mem: AgentMemory,
    question: str,
    budget: int = 6000,
    model: str | None = None,
    api_key: str | None = None,
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

    # Route through the LLM pool — provider/base_url/api_key_env all come
    # from ``configs/attestor.yaml`` via the ``provider/`` prefix on the
    # model id. Any missing provider key surfaces as a RuntimeError from
    # the pool; we let it propagate so misconfig fails loudly rather
    # than silently returning raw context.
    from attestor.llm_trace import get_client_for_model, traced_create

    try:
        client, clean_model = get_client_for_model(model)
    except ImportError:
        return context  # openai SDK not installed — degrade to raw context.

    prompt_template = MAB_EXACT_PROMPT if use_exact else MAB_ANSWER_PROMPT
    prompt = prompt_template.format(context=context, question=question)

    # Retry with exponential backoff on rate limits
    import time as _time
    for attempt in range(5):
        try:
            response = traced_create(
                client,
                role="mab.chat",
                model=clean_model,
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
