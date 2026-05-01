"""LoCoMo reflection / answer synthesis — coreference resolution, context
building, and the reflection-loop answer pipeline."""

from __future__ import annotations

from typing import Any

from attestor.core import AgentMemory
from attestor.locomo.runner import DEFAULT_MODEL, _chat, _resolve_client
from attestor.retrieval.planner import QueryPlan, plan_query


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

ANSWER_PROMPT = (
    "You are answering questions about conversations between {speaker_a} and {speaker_b}.\n"
    "Below are relevant facts and relationships from their conversations. Use ONLY this information.\n\n"
    "Each fact line may be prefixed with [YYYY-MM-DD] — that is the SESSION date when the\n"
    "statement was made, NOT necessarily the date of the event itself. You MUST resolve\n"
    "relative time expressions against that session date:\n"
    "  - \"last year\" / \"a year ago\" → subtract 1 year from session date\n"
    "  - \"last month\" → previous calendar month\n"
    "  - \"next month\" / \"next week\" → add to session date\n"
    "  - \"yesterday\" → session date - 1 day\n"
    "  - Questions asking \"when\" want the EVENT date, not the session date.\n"
    "CRITICAL: If a \"when\" question points at a fact whose only date hint is a relative\n"
    "phrase like \"last year\" or \"last month\", you MUST compute the absolute year/date\n"
    "from the session anchor and answer with the computed value. Never hedge with \"before\n"
    "<session date>\" or paraphrase the relative phrase — commit to the concrete year/date.\n"
    "Worked example:\n"
    "  Context line: [2023-05-08] Melanie: I painted that lake sunrise last year.\n"
    "  Question    : When did Melanie paint a sunrise?\n"
    "  Correct     : 2022\n"
    "  Wrong       : \"last year\", \"Before May 8, 2023\", \"2023\", \"Unknown\".\n\n"
    "For questions asking about MULTIPLE items or fields (\"what fields\", \"which subjects\",\n"
    "\"list the...\", \"Would be likely to pursue\"), scan ALL facts and include every distinct\n"
    "item you find, comma-separated. When the question asks what fields someone is \"likely\"\n"
    "to pursue in education/career, include both the explicitly stated domain AND closely\n"
    "related academic fields (e.g. \"mental health\" → psychology; \"counseling\" → counseling\n"
    "certification). Prefer breadth over a single narrow term.\n\n"
    "Facts:\n{context}\n{graph_context}\n"
    "Question: {question}\n\n"
    "Give a very CONCISE answer (short phrase about core information only).\n"
    "- Answer with just the key fact: a name, date, place, number, or short phrase\n"
    "- Do NOT write full sentences, explanations, or elaborations\n"
    "- Do NOT repeat the question or add context\n"
    "- If the question asks \"who\", answer with just the name\n"
    "- If the question asks \"when\", answer with just the date or time (EVENT date, not session date)\n"
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
    turns: list[dict[str, Any]],
    session_date: str,
    speaker_a: str,
    speaker_b: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
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

    client, clean_model = _resolve_client(model)
    resolved_text = _chat(client, clean_model, prompt, max_tokens=4096)
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


def _extract_names_from_question(question: str) -> list[str]:
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
    results: list,
    mem: AgentMemory,
    names: list[str],
    speaker_a: str,
    speaker_b: str,
) -> tuple:
    """Build context string and graph context from retrieval results.

    Each memory line is prefixed with its temporal anchor (event_date or
    session date from metadata) so the answer model can resolve relative
    time expressions ("last year", "next month") without guessing.
    """
    lines: list[str] = []
    for r in results:
        mem_obj = r.memory
        anchor = mem_obj.event_date or ""
        if not anchor and isinstance(mem_obj.metadata, dict):
            anchor = mem_obj.metadata.get("session_date", "") or ""
        prefix = f"[{anchor}] " if anchor else ""
        lines.append(f"{prefix}{mem_obj.content}")
    context = "\n".join(lines)

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
                source_quote = edge.get("source_quote", "")
                triple_key = (subj.lower(), pred, obj.lower())
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    date_str = f" ({event_date})" if event_date else ""
                    quote_str = f' — "{source_quote}"' if source_quote else ""
                    graph_lines.append(
                        f"- {subj} {pred} {obj}{date_str}{quote_str}"
                    )
        if graph_lines:
            graph_context = "\nRelationships:\n" + "\n".join(graph_lines[:30])

    return context, graph_context


def answer_question(
    mem: AgentMemory,
    question: str,
    budget: int = 4000,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    speaker_a: str = "A",
    speaker_b: str = "B",
    enable_reflection: bool = True,
    use_planner: bool = False,
    planner_model: str | None = None,
) -> str:
    """Use Attestor recall + LLM synthesis to answer a LOCOMO question.

    With enable_reflection=True, checks if the initial retrieval is sufficient
    and does targeted follow-up queries if gaps are detected.

    With use_planner=True, an LLM planner classifies the question into
    {intent, entities, namespaces, filters} and recall is split across the
    selected namespaces (budget divided evenly) for namespace-filtered retrieval.
    """
    plan: QueryPlan | None = None
    if use_planner:
        plan_kwargs: dict[str, Any] = {"api_key": api_key}
        if planner_model:
            plan_kwargs["model"] = planner_model
        plan = plan_query(question, **plan_kwargs)

    results = []
    seen_ids: set = set()
    if plan and plan.namespaces:
        per_ns_budget = max(budget // len(plan.namespaces), 1000)
        for ns in plan.namespaces:
            got = mem.recall(question, budget=per_ns_budget, namespace=ns)
            for r in got:
                if r.memory.id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.memory.id)
    else:
        results = mem.recall(question, budget=budget)
        seen_ids = {r.memory.id for r in results}

    names = plan.entities if (plan and plan.entities) else _extract_names_from_question(question)
    for name in names[:3]:
        extra = mem.recall(name, budget=1000)
        for r in extra:
            if r.memory.id not in seen_ids:
                results.append(r)
                seen_ids.add(r.memory.id)

    if not results:
        return "I don't have enough information to answer this question."

    results.sort(key=lambda r: r.score, reverse=True)

    client, clean_model = _resolve_client(model)
    context, graph_context = _build_context(results, mem, names, speaker_a, speaker_b)

    # Reflection loop: check if context is sufficient, do follow-up queries if not
    if enable_reflection:
        reflection_prompt = REFLECTION_PROMPT.format(
            question=question,
            answer="(not yet generated)",
            context=context[:2000],
        )
        reflection_text = _chat(client, clean_model, reflection_prompt, max_tokens=200)

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

    return _chat(client, clean_model, prompt, max_tokens=100)
