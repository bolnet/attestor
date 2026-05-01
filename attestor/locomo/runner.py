"""LOCOMO benchmark runner for Attestor.

LOCOMO (Long Conversation Memory) is the industry-standard benchmark
for evaluating AI agent memory systems. Used by Mem0, Zep, Letta, etc.

Usage:
    attestor locomo                     # Run with defaults — needs the API
                                        # key for the provider configured in
                                        # ``configs/attestor.yaml``
                                        # (``llm.api_key_env``).
    attestor locomo --judge-model claude-haiku  # Cheaper judge
    attestor locomo --data ./locomo10.json       # Use local data file

Requires: poetry add "attestor[extraction]"  (for the LLM judge)
"""

from __future__ import annotations

import json
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from attestor.core import AgentMemory
from attestor.mab import token_f1, _upgrade_embeddings_for_benchmark


def _default_model() -> str:
    """Resolve the LoCoMo benchmark default model from
    ``configs/attestor.yaml``."""
    from attestor.config import get_stack
    return get_stack().models.benchmark_default


DEFAULT_MODEL = _default_model()


def _resolve_client(model: str) -> tuple[Any, str]:
    """Look up the provider client for ``model`` via the LLM pool.

    YAML is the source of truth: ``configs/attestor.yaml`` declares the
    provider map (base_url + api_key_env) and the ``provider/`` prefix
    in ``model`` selects which one to use. Returns a ``(client,
    clean_model)`` pair where ``clean_model`` has the prefix stripped —
    pass that to the SDK.
    """
    from attestor.llm_trace import get_client_for_model
    return get_client_for_model(model)


def _chat(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int = 200,
    *,
    role: str = "locomo.chat",
) -> str:
    """Send a chat completion request and return the text."""
    from attestor.llm_trace import traced_create
    response = traced_create(
        client,
        role=role,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def _guess_entity_type(name: str) -> str:
    """Guess entity type from name for graph storage."""
    name_lower = name.lower()

    location_words = {"city", "country", "town", "state", "island", "lake",
                      "river", "mountain", "park", "beach", "street", "avenue"}
    if any(w in name_lower for w in location_words):
        return "location"

    org_words = {"company", "corp", "inc", "university", "school", "hospital",
                 "club", "team", "association", "foundation", "institute"}
    if any(w in name_lower for w in org_words):
        return "organization"

    if name and name[0].isupper():
        return "entity"

    return "concept"


def download_locomo(dest_path: str) -> str:
    """Download LOCOMO dataset if not already present."""
    if not Path(dest_path).exists():
        print(f"Downloading LOCOMO dataset to {dest_path}...")
        urllib.request.urlretrieve(LOCOMO_URL, dest_path)
        print("Downloaded.")
    return dest_path


def load_locomo(file_path: str) -> list[dict[str, Any]]:
    """Load and parse the LOCOMO dataset."""
    with open(file_path) as f:
        data = json.load(f)

    conversations = []
    for sample in data:
        conv = sample["conversation"]

        sessions = []
        session_num = 1
        while f"session_{session_num}" in conv:
            sessions.append({
                "session_id": session_num,
                "date_time": conv.get(f"session_{session_num}_date_time", ""),
                "turns": conv[f"session_{session_num}"],
            })
            session_num += 1

        qa_pairs = sample.get("qa", [])
        scoreable_qa = [qa for qa in qa_pairs if qa.get("category") != 5]

        conversations.append({
            "sample_id": sample.get("sample_id", ""),
            "speaker_a": conv.get("speaker_a", "A"),
            "speaker_b": conv.get("speaker_b", "B"),
            "sessions": sessions,
            "qa": scoreable_qa,
        })

    return conversations


def ingest_conversation(
    mem: AgentMemory,
    conversation: dict[str, Any],
    use_extraction: bool = False,
    extraction_model: str | None = None,
    api_key: str | None = None,
    verbose: bool = False,
    graph=None,
    resolve_pronouns: bool = False,
) -> int:
    """Ingest a LOCOMO conversation into an Attestor store.

    If use_extraction=True, uses LLM to extract atomic facts and relation
    triples from each session. Triples are stored in the entity graph.
    Otherwise, stores raw dialogue turns (legacy behavior).

    Args:
        graph: Optional entity graph for storing LLM-extracted triples.
               Separate from mem._graph to allow disabling rule-based graph
               extraction in core.add() while still storing LLM triples.

    Returns the number of memories added.
    """
    from attestor.locomo.reflection import _resolve_coreferences

    count = 0
    total_triples = 0
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    if extraction_model is None:
        from attestor.config import get_stack
        extraction_model = get_stack().models.extraction

    if use_extraction:
        from attestor.extraction.extractor import extract_from_session

        for session in conversation["sessions"]:
            session_id = f"session_{session['session_id']}"
            date_time = session.get("date_time", "")
            turns = session["turns"]

            if verbose:
                print(f"    Extracting facts from {session_id} ({len(turns)} turns)...")

            memories, triples = extract_from_session(
                turns=turns,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                session_date=date_time,
                model=extraction_model,
                api_key=api_key,
            )

            for mem_obj in memories:
                mem.add(
                    content=mem_obj.content,
                    tags=mem_obj.tags,
                    category=mem_obj.category,
                    entity=mem_obj.entity,
                    event_date=mem_obj.event_date or date_time,
                    confidence=mem_obj.confidence,
                    metadata={"session": session_id},
                )
                count += 1

            if triples and graph:
                for triple in triples:
                    subj = triple["subject"]
                    pred = triple["predicate"]
                    obj = triple["object"]

                    edge_meta = {
                        "event_date": triple.get("event_date") or date_time,
                        "session": session_id,
                    }

                    graph.add_entity(subj, "person")
                    graph.add_entity(obj, _guess_entity_type(obj))
                    graph.add_relation(subj, obj, pred, metadata=edge_meta)

                total_triples += len(triples)
                graph.save()

            if verbose:
                print(f"    → {len(memories)} facts, {len(triples)} relations extracted")
    else:
        for session in conversation["sessions"]:
            session_id = f"session_{session['session_id']}"
            date_time = session.get("date_time", "")
            turns = session["turns"]

            # Resolve pronouns and relative time references
            if resolve_pronouns:
                if verbose:
                    print(f"    Resolving coreferences in {session_id} ({len(turns)} turns)...")
                turns = _resolve_coreferences(
                    turns, date_time, speaker_a, speaker_b,
                    api_key=api_key, model=extraction_model,
                )

            for turn in turns:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                if not text:
                    continue

                # Map speaker labels to actual names
                display_speaker = speaker
                if speaker == "A":
                    display_speaker = speaker_a
                elif speaker == "B":
                    display_speaker = speaker_b

                tags = [display_speaker.lower(), session_id]

                mem.add(
                    content=f"{display_speaker}: {text}",
                    tags=tags,
                    category="conversation",
                    entity=display_speaker,
                    event_date=date_time,
                    metadata={"dia_id": turn.get("dia_id", ""), "session": session_id},
                )
                count += 1

    return count


def run_locomo(
    data_path: str | None = None,
    judge_model: str | None = None,
    answer_model: str | None = None,
    extraction_model: str | None = None,
    use_extraction: bool = False,
    resolve_pronouns: bool = True,
    max_conversations: int | None = None,
    max_questions_per_conv: int | None = None,
    recall_budget: int = 4000,
    verbose: bool = False,
    api_key: str | None = None,
    backend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full LOCOMO benchmark.

    Args:
        data_path: Path to locomo10.json. Downloads if None.
        judge_model: LLM model for judging answers.
        answer_model: LLM model for synthesizing answers from retrieved memories.
        extraction_model: LLM model for fact extraction during ingestion.
        use_extraction: If True, extract facts with LLM instead of storing raw turns.
        max_conversations: Limit number of conversations (for quick tests).
        max_questions_per_conv: Limit questions per conversation.
        recall_budget: Token budget for recall queries.
        verbose: Print progress details.

    Returns:
        Dict with per-category and overall scores.
    """
    from attestor.locomo.judge import judge_answer
    from attestor.locomo.reflection import answer_question

    if data_path is None:
        from attestor._paths import resolve_cache_dir

        cache_dir = resolve_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(cache_dir / "locomo10.json")
        download_locomo(data_path)

    # Resolve any unset model arg from the YAML SoT.
    if judge_model is None or answer_model is None or extraction_model is None:
        from attestor.config import get_stack
        s = get_stack()
        if judge_model is None:
            judge_model = s.models.judge
        if answer_model is None:
            answer_model = s.models.answerer
        if extraction_model is None:
            extraction_model = s.models.extraction

    conversations = load_locomo(data_path)
    if max_conversations:
        conversations = conversations[:max_conversations]

    category_results = {1: [], 2: [], 3: [], 4: []}
    all_results = []
    total_ingest_time = 0.0
    total_recall_time = 0.0
    total_judge_time = 0.0

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("sample_id", f"conv_{conv_idx}")
        if verbose:
            print(f"\n--- Conversation {conv_idx + 1}/{len(conversations)}: {conv_id} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = AgentMemory(tmpdir, config=backend_config)
            # Use OpenAI embeddings for benchmarking if key available
            _upgrade_embeddings_for_benchmark(mem)

            # Disable temporal boost and contradiction checking for benchmark
            if mem._retrieval:
                mem._retrieval.enable_temporal_boost = False
            if mem._temporal:
                mem._temporal.check_contradictions = lambda m: []

            original_graph = mem._graph

            t0 = time.perf_counter()
            mem_count = ingest_conversation(
                mem, conv,
                use_extraction=use_extraction,
                extraction_model=extraction_model,
                api_key=api_key,
                verbose=verbose,
                graph=original_graph,
                resolve_pronouns=resolve_pronouns,
            )

            # Batch embed all memories
            if mem._vector_store:
                embed_count = mem.batch_embed()
            else:
                embed_count = 0

            ingest_time = time.perf_counter() - t0
            total_ingest_time += ingest_time

            if verbose:
                graph_stats = ""
                if mem._graph:
                    g_info = mem._graph.graph_stats()
                    graph_stats = f", graph: {g_info.get('nodes', 0)} nodes, {g_info.get('edges', 0)} edges"
                print(f"  Ingested {mem_count} memories ({embed_count} embedded{graph_stats}) in {ingest_time:.2f}s")

            qa_pairs = conv.get("qa", [])
            if max_questions_per_conv:
                qa_pairs = qa_pairs[:max_questions_per_conv]

            for qa_idx, qa in enumerate(qa_pairs):
                question = qa["question"]
                expected = str(qa["answer"])
                category = qa["category"]

                t0 = time.perf_counter()
                generated = answer_question(
                    mem, question,
                    budget=recall_budget,
                    model=answer_model,
                    api_key=api_key,
                    speaker_a=conv.get("speaker_a", "A"),
                    speaker_b=conv.get("speaker_b", "B"),
                )
                recall_time = time.perf_counter() - t0
                total_recall_time += recall_time

                t0 = time.perf_counter()
                judgment = judge_answer(
                    question, expected, generated,
                    model=judge_model,
                    api_key=api_key,
                )
                judge_time = time.perf_counter() - t0
                total_judge_time += judge_time

                is_correct = judgment["correct"]
                f1, prec, rec = token_f1(str(generated), str(expected))
                result = {
                    "conversation": conv_id,
                    "question": question,
                    "expected": expected,
                    "generated": generated,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, "unknown"),
                    "correct": is_correct,
                    "f1": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "reasoning": judgment.get("reasoning", ""),
                    "recall_ms": round(recall_time * 1000),
                }

                all_results.append(result)
                category_results.setdefault(category, []).append(result)

                if verbose:
                    status = "CORRECT" if is_correct else "WRONG"
                    cat_name = CATEGORY_NAMES.get(category, "?")
                    print(f"  [{status}] F1={f1:.2f} ({cat_name}) Q: {question[:70]}...")

            mem.close()

    # Compute scores
    scores = {}
    total_correct = 0
    total_evaluated = 0

    all_f1_scores = []
    for cat_id, results_list in category_results.items():
        if not results_list:
            continue
        correct = sum(1 for r in results_list if r["correct"])
        total = len(results_list)
        accuracy = round(correct / max(total, 1) * 100, 1)
        cat_f1_scores = [r["f1"] for r in results_list]
        avg_f1 = round(sum(cat_f1_scores) / len(cat_f1_scores) * 100, 1)
        total_correct += correct
        total_evaluated += total
        all_f1_scores.extend(cat_f1_scores)
        scores[CATEGORY_NAMES.get(cat_id, "unknown")] = {
            "accuracy": accuracy,
            "f1": avg_f1,
            "correct": correct,
            "total": total,
        }

    overall_accuracy = round(total_correct / max(total_evaluated, 1) * 100, 1)
    overall_f1 = round(sum(all_f1_scores) / max(len(all_f1_scores), 1) * 100, 1)

    return {
        "overall_accuracy": overall_accuracy,
        "overall_f1": overall_f1,
        "total_correct": total_correct,
        "total_evaluated": total_evaluated,
        "by_category": scores,
        "timing": {
            "total_ingest_s": round(total_ingest_time, 2),
            "total_recall_s": round(total_recall_time, 2),
            "total_judge_s": round(total_judge_time, 2),
            "avg_recall_ms": round(total_recall_time / max(total_evaluated, 1) * 1000, 1),
        },
        "config": {
            "judge_model": judge_model,
            "answer_model": answer_model,
            "extraction_model": extraction_model or "none",
            "use_extraction": use_extraction,
            "resolve_pronouns": resolve_pronouns,
            "reflection": True,
            "recall_budget": recall_budget,
        },
        "conversations": len(conversations),
        "details": all_results,
    }


def print_locomo(results: dict[str, Any]) -> None:
    """Pretty-print LOCOMO benchmark results."""
    print("LOCOMO Benchmark Results — Attestor")
    print("=======================================================")
    print(f"Judge model: {results['config']['judge_model']}")
    print(f"Conversations: {results['conversations']}")
    print(f"Questions evaluated: {results['total_evaluated']}")
    print(f"Overall Accuracy: {results['overall_accuracy']}%"
          f"  ({results['total_correct']}/{results['total_evaluated']} correct)")
    print(f"Overall F1: {results.get('overall_f1', 0)}%")
    print("--------------------")
    print("By Category:")
    print(f"  {'Category':<20} {'Accuracy':>10} {'F1':>8} {'Correct':>10}")
    print(f"  {'----------':<20} {'----------':>10} {'------':>8} {'----------':>10}")
    for cat_name, scores in results["by_category"].items():
        f1_str = f"{scores.get('f1', 0):.1f}%"
        print(f"  {cat_name:<20} {scores['accuracy']:>9.1f}% {f1_str:>8} {scores['correct']:>4}/{scores['total']:<4}")
    print("--------------------")
    print("Timing:")
    timing = results["timing"]
    print(f"  Ingest:     {timing['total_ingest_s']:.2f}s")
    print(f"  Recall:     {timing['total_recall_s']:.2f}s  (avg {timing['avg_recall_ms']:.1f}ms/query)")
    print(f"  LLM Judge:  {timing['total_judge_s']:.2f}s")
    print("=======================================================")
    print("Compare with (LLM-judge accuracy / token F1):")
    print("  SimpleMem:      —    / 43.2% F1  (GPT-4.1-mini, arXiv:2601.02553)")
    print("  Mem0:           66.9% / 16.8% F1  (arXiv:2504.19413)")
    print("  Zep/Graphiti:   58.4% / —         (disputed)")
    print("  Letta/MemGPT:   74.0% / —         (GPT-4o mini, file-based)")
