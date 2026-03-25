"""LOCOMO benchmark runner for Memwright.

LOCOMO (Long Conversation Memory) is the industry-standard benchmark
for evaluating AI agent memory systems. Used by Mem0, Zep, Letta, etc.

Usage:
    agent-memory locomo                     # Run with defaults (needs OPENROUTER_API_KEY)
    agent-memory locomo --judge-model claude-haiku  # Cheaper judge
    agent-memory locomo --data ./locomo10.json       # Use local data file

Requires: poetry add "memwright[extraction]"  (for the LLM judge)
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.core import AgentMemory
from agent_memory.mab import token_f1, _upgrade_embeddings_for_benchmark

DEFAULT_MODEL = "openai/gpt-4.1-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_client(api_key: Optional[str] = None):
    """Get an OpenAI client configured for OpenRouter."""
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Pass api_key or set the env var."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def _chat(client, model: str, prompt: str, max_tokens: int = 200) -> str:
    """Send a chat completion request and return the text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

REFLECTION_PROMPT = (
    "You tried to answer a question but the retrieved context may be incomplete.\n\n"
    "Question: {question}\n"
    "Current answer: {answer}\n"
    "Retrieved context:\n{context}\n\n"
    "Analyze: Does the context contain enough information to answer the question?\n"
    "If YES, respond with just: SUFFICIENT\n"
    "If NO, respond with 1-3 alternative search queries (one per line) that might find the missing information.\n"
    "Each query should be a short phrase targeting specific missing facts.\n"
    "Return ONLY 'SUFFICIENT' or the queries, nothing else."
)

RESOLVE_PROMPT = (
    "Rewrite each dialogue turn below so that:\n"
    "1. ALL pronouns (he, she, they, his, her, it, etc.) are replaced with the actual name or noun they refer to\n"
    "2. ALL relative time references (yesterday, next week, last month, tomorrow, etc.) are resolved to absolute dates using the session date: {session_date}\n"
    "3. Keep the original meaning exactly — only change pronouns and time references\n"
    "4. Preserve the speaker labels exactly as given\n\n"
    "Conversation:\n{conversation}\n\n"
    "Return ONLY the rewritten conversation, one line per turn, in the format 'Speaker: text'. No other text."
)

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}

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

ANSWER_PROMPT = (
    "You are answering questions about conversations between {speaker_a} and {speaker_b}.\n"
    "Below are relevant facts and relationships from their conversations. Use ONLY this information.\n\n"
    "Facts:\n{context}\n{graph_context}\n"
    "Question: {question}\n\n"
    "Give a very CONCISE answer (short phrase about core information only).\n"
    "- Answer with just the key fact: a name, date, place, number, or short phrase\n"
    "- Do NOT write full sentences, explanations, or elaborations\n"
    "- Do NOT repeat the question or add context\n"
    "- If the question asks \"who\", answer with just the name\n"
    "- If the question asks \"when\", answer with just the date or time\n"
    "- If the question asks \"where\", answer with just the place\n"
    "- If multiple items, list them separated by commas\n"
    "- For \"Would...\" or \"likely\" questions, ALWAYS give your best inference from the facts.\n"
    "  Answer with \"Yes\" or \"No\" or \"Likely yes/no\" followed by a SHORT reason.\n"
    "  Example: \"Likely no; she prefers outdoor activities\"\n"
    "  Example: \"Yes, since she collects classic children's books\"\n"
    "- NEVER say \"I don't know\" if there are ANY relevant facts to reason from.\n"
    "  Only say \"I don't know\" if the facts contain absolutely nothing related to the question.\n\n"
    "Examples of good answers: \"Paris\", \"March 2024\", \"yoga and hiking\", "
    "\"Likely yes; she enjoys classical music\", \"No; she wants to be a counselor\""
)


def _resolve_coreferences(
    turns: List[Dict[str, Any]],
    session_date: str,
    speaker_a: str,
    speaker_b: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Resolve pronouns and relative time references in dialogue turns using LLM."""
    # Build conversation text
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        if speaker == "A":
            speaker = speaker_a
        elif speaker == "B":
            speaker = speaker_b
        text = turn.get("text", "")
        if text:
            lines.append(f"{speaker}: {text}")

    if not lines:
        return turns

    conversation_text = "\n".join(lines)
    prompt = RESOLVE_PROMPT.format(
        session_date=session_date,
        conversation=conversation_text,
    )

    client = _get_client(api_key)
    resolved_text = _chat(client, model, prompt, max_tokens=4096)
    resolved_lines = [l.strip() for l in resolved_text.split("\n") if l.strip()]

    resolved_turns = []
    for i, line in enumerate(resolved_lines):
        if ": " in line:
            speaker_name, text = line.split(": ", 1)
            # Map names back to A/B labels
            if speaker_name == speaker_a:
                speaker_label = "A"
            elif speaker_name == speaker_b:
                speaker_label = "B"
            else:
                speaker_label = speaker_name

            # Preserve original metadata if available
            orig = turns[i] if i < len(turns) else {}
            resolved_turns.append({
                **orig,
                "speaker": speaker_label,
                "text": text,
            })
        elif i < len(turns):
            resolved_turns.append(turns[i])

    return resolved_turns if resolved_turns else turns


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


def load_locomo(file_path: str) -> List[Dict[str, Any]]:
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
    conversation: Dict[str, Any],
    use_extraction: bool = False,
    extraction_model: str = "openai/gpt-4.1-mini",
    api_key: Optional[str] = None,
    verbose: bool = False,
    graph=None,
    resolve_pronouns: bool = False,
) -> int:
    """Ingest a LOCOMO conversation into a Memwright store.

    If use_extraction=True, uses LLM to extract atomic facts and relation
    triples from each session. Triples are stored in the entity graph.
    Otherwise, stores raw dialogue turns (legacy behavior).

    Args:
        graph: Optional entity graph for storing LLM-extracted triples.
               Separate from mem._graph to allow disabling rule-based graph
               extraction in core.add() while still storing LLM triples.

    Returns the number of memories added.
    """
    count = 0
    total_triples = 0
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    if use_extraction:
        from agent_memory.extraction.extractor import extract_from_session

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


def _extract_names_from_question(question: str) -> List[str]:
    """Extract proper nouns from a question for secondary retrieval."""
    import re

    words = question.split()
    names = []
    for i, word in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z']", "", word)
        if not cleaned or not cleaned[0].isupper() or i == 0 or len(cleaned) < 2:
            continue
        if cleaned.lower() in frozenset({
            "is", "are", "was", "were", "has", "have", "did", "does",
            "can", "could", "would", "should", "who", "what", "where",
            "when", "why", "how",
        }):
            continue
        names.append(cleaned)
    return names


def _build_context(
    results: List,
    mem: AgentMemory,
    names: List[str],
    speaker_a: str,
    speaker_b: str,
) -> tuple:
    """Build context string and graph context from retrieval results."""
    context = "\n".join(r.memory.content for r in results)

    graph_context = ""
    if mem._graph and hasattr(mem._graph, "get_edges"):
        graph_lines = []
        seen_triples = set()
        for name in names + [speaker_a, speaker_b]:
            edges = mem._graph.get_edges(name)
            for edge in edges[:10]:
                subj = edge.get("subject", "")
                pred = edge.get("predicate", "related_to")
                obj = edge.get("object", "")
                event_date = edge.get("event_date", "")
                triple_key = (subj.lower(), pred, obj.lower())
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    date_str = f" ({event_date})" if event_date else ""
                    graph_lines.append(f"- {subj} {pred} {obj}{date_str}")
        if graph_lines:
            graph_context = "\nRelationships:\n" + "\n".join(graph_lines[:30])

    return context, graph_context


def answer_question(
    mem: AgentMemory,
    question: str,
    budget: int = 4000,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    speaker_a: str = "A",
    speaker_b: str = "B",
    enable_reflection: bool = True,
) -> str:
    """Use Memwright recall + LLM synthesis to answer a LOCOMO question.

    With enable_reflection=True, checks if the initial retrieval is sufficient
    and does targeted follow-up queries if gaps are detected.
    """
    results = mem.recall(question, budget=budget)

    names = _extract_names_from_question(question)
    seen_ids = {r.memory.id for r in results}
    for name in names[:3]:
        extra = mem.recall(name, budget=1000)
        for r in extra:
            if r.memory.id not in seen_ids:
                results.append(r)
                seen_ids.add(r.memory.id)

    if not results:
        return "I don't have enough information to answer this question."

    results.sort(key=lambda r: r.score, reverse=True)

    client = _get_client(api_key)
    context, graph_context = _build_context(results, mem, names, speaker_a, speaker_b)

    # Reflection loop: check if context is sufficient, do follow-up queries if not
    if enable_reflection:
        reflection_prompt = REFLECTION_PROMPT.format(
            question=question,
            answer="(not yet generated)",
            context=context[:2000],
        )
        reflection_text = _chat(client, model, reflection_prompt, max_tokens=200)

        if "SUFFICIENT" not in reflection_text.upper():
            follow_up_queries = [q.strip() for q in reflection_text.split("\n") if q.strip()]
            for fq in follow_up_queries[:3]:
                extra = mem.recall(fq, budget=1000)
                for r in extra:
                    if r.memory.id not in seen_ids:
                        results.append(r)
                        seen_ids.add(r.memory.id)

            results.sort(key=lambda r: r.score, reverse=True)
            context, graph_context = _build_context(
                results, mem, names, speaker_a, speaker_b,
            )

    prompt = ANSWER_PROMPT.format(
        context=context,
        question=question,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        graph_context=graph_context,
    )

    return _chat(client, model, prompt, max_tokens=100)


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Use LLM to judge if the generated answer is correct."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        ai_response=generated,
    )

    client = _get_client(api_key)
    response_text = _chat(client, model, prompt, max_tokens=300)

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


def run_locomo(
    data_path: Optional[str] = None,
    judge_model: str = "openai/gpt-4.1-mini",
    answer_model: str = "openai/gpt-4.1-mini",
    extraction_model: str = "openai/gpt-4.1-mini",
    use_extraction: bool = False,
    resolve_pronouns: bool = True,
    max_conversations: Optional[int] = None,
    max_questions_per_conv: Optional[int] = None,
    recall_budget: int = 4000,
    verbose: bool = False,
    api_key: Optional[str] = None,
    backend_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    if data_path is None:
        cache_dir = Path.home() / ".cache" / "memwright"
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(cache_dir / "locomo10.json")
        download_locomo(data_path)

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


def print_locomo(results: Dict[str, Any]) -> None:
    """Pretty-print LOCOMO benchmark results."""
    print("LOCOMO Benchmark Results — Memwright")
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
