"""HyDE retrieval — Hypothetical Document Embedding (Phase 3 PR-D).

A small LLM generates a 1-2-sentence hypothetical answer to the user's
question; the orchestrator embeds and searches BOTH the original
question AND the hypothetical answer in parallel, then RRF-merges the
two ranked lists before the rest of the cascade.

Per the RCA roadmap, this is the +6-10% recall lever sibling to
multi-query (PR #94). Intuition: dense embedding similarity between
a question and an answer is asymmetric — questions don't structurally
resemble their answers. Running both as queries widens the search
pattern across the latent space, raising the chance the correct
memory is in *some* lane's top-K.

This module is pure — it produces ``(queries: List[str], merged_hits:
List[dict])`` but does not call the vector store. The orchestrator
wires the lane in by handing in a ``vector_search`` callable, mirror
of the multi_query integration pattern.

Configuration lives in ``configs/attestor.yaml`` under
``retrieval.hyde``:

    retrieval:
      hyde:
        enabled: false           # default off
        generator_model: null    # null → models.extraction
        generator_reasoning_effort: low
        merge: rrf               # rrf | union

Trace events emitted (when ATTESTOR_TRACE=1):

  - ``recall.hyde.generated`` — the original + the hypothetical preview
  - ``recall.hyde.merged``    — pre-merge per-lane sizes + final size

Mutually exclusive with multi_query in this PR. The orchestrator
prefers multi_query when both are enabled (it's been shipped longer
and proven; HyDE is new). The combination is left for a follow-up.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("attestor.retrieval.hyde")


# ──────────────────────────────────────────────────────────────────────
# Result shape
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HydeResult:
    """The generator's output. ``original_question`` is always
    present; ``hypothetical_answer`` is empty string on degraded paths
    so the caller can detect "generation failed → just use the
    question alone" without checking for sentinel exceptions.
    """

    original_question: str
    hypothetical_answer: str = ""

    @property
    def queries(self) -> List[str]:
        """Queries to fan out, in lane-priority order. The original
        is always rank-0; the hypothetical comes second when present.

        On degraded paths (empty ``hypothetical_answer``), returns just
        the original — the caller naturally falls back to single-query.
        """
        if not self.hypothetical_answer.strip():
            return [self.original_question]
        return [self.original_question, self.hypothetical_answer]


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────


_GENERATOR_PROMPT = """You are previewing a personal-memory search. The \
user's question refers to an event, fact, or experience from their past \
conversations. Write 1-2 sentences DESCRIBING that event in the form the \
user would have ORIGINALLY MENTIONED it in conversation — not in the \
form of an answer to the question.

The goal is to generate a snippet whose embedding is close to the \
original conversation turn containing the answer, NOT to the answer \
itself. Question-shape and answer-shape don't embed close to source-shape.

Examples:
- Question: "How many weeks ago did I meet my aunt?"
  Snippet: "Yesterday I drove out to visit my aunt Linda — she gave me \
a gorgeous antique chandelier from her attic."
- Question: "Which trip did I take first this year?"
  Snippet: "Just got back from Tokyo last month, my first international \
trip of the year. The cherry blossoms were already starting."
- Question: "How many days ago did I buy a smoker?"
  Snippet: "Picked up a Traeger pellet smoker over the weekend — finally \
ready to do real low-and-slow brisket at home."

Rules:
- 1-2 sentences, first-person, declarative, conversational.
- DESCRIBE the event directly; do NOT say "X weeks ago" or "first" or \
counts — narrate the event itself.
- Invent plausible specifics (names, places, objects, dates) — accuracy \
doesn't matter, surface-form match to a likely chat turn does.
- Do NOT echo the question, do NOT hedge, do NOT refuse.

Question: {question}

Snippet:"""


def _resolve_generator_model() -> str:
    """Generator model: env override > YAML > extraction default."""
    if env := os.environ.get("HYDE_GENERATOR_MODEL"):
        return env
    from attestor.config import get_stack
    stack = get_stack()
    hyde_cfg = getattr(stack.retrieval, "hyde", None)
    if hyde_cfg is not None and getattr(hyde_cfg, "generator_model", None):
        return hyde_cfg.generator_model
    return stack.models.extraction


def generate_hypothetical_answer(
    question: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> HydeResult:
    """Single LLM call returning a hypothetical answer for ``question``.

    On any error (missing key, malformed response, network timeout),
    returns ``HydeResult(original_question=question, hypothetical_answer="")``
    — the caller naturally degrades to a single-query lane.
    """
    if not question.strip():
        return HydeResult(original_question=question.strip())

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.debug("hyde.generate: no API key; returning degraded result")
        return HydeResult(original_question=question.strip())

    model = model or _resolve_generator_model()

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        from attestor.llm_trace import traced_create
        response = traced_create(
            client,
            role="hyde_generator",
            model=model,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "user", "content": _GENERATOR_PROMPT.format(
                    question=question,
                )},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("hyde.generate: LLM call failed: %s", e)
        return HydeResult(original_question=question.strip())

    # Strip any leading label the model may echo from the prompt frame.
    for label in ("snippet:", "hypothetical answer:"):
        if text.lower().startswith(label):
            text = text[len(label):].strip()
            break

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.hyde.generated",
            original=question[:200],
            hypothetical_length=len(text),
            hypothetical_preview=text[:200],
        )

    return HydeResult(
        original_question=question.strip(),
        hypothetical_answer=text,
    )


# ──────────────────────────────────────────────────────────────────────
# End-to-end lane
# ──────────────────────────────────────────────────────────────────────


def hyde_search(
    question: str,
    *,
    vector_search: Callable[[str], List[Dict[str, Any]]],
    model: Optional[str] = None,
    merge: str = "rrf",
    api_key: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Run the generator, fan out to ``vector_search`` for each query
    (original + hypothetical), merge the lanes, return
    ``(queries_used, merged_hits)``.

    Same pattern as ``attestor.retrieval.multi_query.multi_query_search``
    — RRF helpers are imported from there to avoid duplication. Falls
    back to single-query when generation fails (queries_used == [original]).
    """
    result = generate_hypothetical_answer(
        question, model=model, api_key=api_key,
    )
    queries = result.queries

    lanes: List[List[Dict[str, Any]]] = []
    for q in queries:
        try:
            hits = list(vector_search(q))
        except Exception as e:  # noqa: BLE001
            logger.debug("hyde: lane %r failed: %s", q[:60], e)
            hits = []
        lanes.append(hits)

    # Reuse multi_query's merge helpers — same RRF k=60, same union
    # behavior. Single source of truth for ranked-list fusion.
    from attestor.retrieval.multi_query import (
        reciprocal_rank_fusion, union_merge,
    )
    if merge == "union":
        merged = union_merge(lanes)
    else:
        merged = reciprocal_rank_fusion(lanes)

    from attestor import trace as _tr
    if _tr.is_enabled():
        _tr.event(
            "recall.hyde.merged",
            n_queries=len(queries),
            per_lane_sizes=[len(l) for l in lanes],
            merged_size=len(merged),
            merge_strategy=merge,
        )

    return queries, merged
