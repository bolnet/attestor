"""LongMemEval ingest + answer + run pipeline.

Distillation, raw-turn ingest, answer prompt + answerer, and the
async orchestrator (``run_async``/``run``) that drives ingest →
answer → judge for each sample under a per-run semaphore.

Internal calls to ``ingest_history``, ``answer_question``, and
``judge_answer`` go through the parent ``attestor.longmemeval``
package so that ``monkeypatch.setattr(lme, ...)`` in tests still
swings the call site (the original module-local lookup behavior).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable

from attestor.longmemeval.fixtures import (
    AnswerResult,
    IngestStats,
    LMESample,
    _chat,
    _extract_retrieved_session_ids,
    _format_recall_context,
    _format_turn_content,
    _get_client,
    _get_client_for_model,
    _iso_date,
    _parse_predicted_mode,
    _short_date,
    _strip_reasoning,
    distill_turn,
    namespace_for,
)
from attestor.longmemeval.judge import (
    DEFAULT_JUDGES,
    DEFAULT_MODEL,
    DEFAULT_PARALLEL,
    _JUDGE_CONCURRENCY,
    _judgement_to_dict,
    _safe_judge_dict,
    judge_personalization,
)
from attestor.longmemeval.prompts import (
    ANSWER_PROMPT,
    VERIFY_PROMPT,
)
from attestor.longmemeval.reporter import (
    LMERunReport,
    RunProvenance,
    SampleReport,
    _attestor_version,
    _git_sha,
    _sha256_file,
    _sha256_str,
    _summarize,
    _summarize_dimensions,
)

logger = logging.getLogger(__name__)



def ingest_history(
    mem: Any,
    sample: LMESample,
    *,
    use_extraction: bool = False,
    extraction_model: str | None = None,
    use_distillation: bool = False,
    distill_model: str | None = None,
    api_key: str | None = None,
    verbose: bool = False,
) -> IngestStats:
    """Ingest a LongMemEval haystack into an Attestor ``AgentMemory``.

    Two strategies:

    1. ``use_extraction=False`` (raw): store each turn as a memory, prefixed
       with ``[YYYY-MM-DD] Role:`` and tagged with the ISO ``event_date``.
       This is the belt-and-suspenders option — any backend that strips
       ``event_date`` still has the date inline.
    2. ``use_extraction=True``: run Attestor's LLM extractor per session to
       distill atomic facts + relation triples, then store those.

    Args:
        mem: Instantiated ``attestor.core.AgentMemory``.
        sample: Frozen ``LMESample``.
        use_extraction: Extract atomic facts instead of ingesting raw turns.
        extraction_model: OpenRouter model id (only used when extracting).
        api_key: Optional override for the extractor's API key.
        verbose: Print per-session progress to stdout.

    Returns:
        ``IngestStats`` with turn / memory / session counts.
    """
    if extraction_model is None or distill_model is None:
        from attestor.config import get_stack
        s = get_stack()
        if extraction_model is None:
            extraction_model = s.models.extraction
        if distill_model is None:
            distill_model = s.models.distill

    ns = namespace_for(sample)
    turns_seen = 0
    memories_added = 0
    skipped_empty = 0
    distilled_facts = 0
    skipped_by_distiller = 0

    sessions = list(
        zip(sample.haystack_session_ids, sample.haystack_dates, sample.haystack_sessions)
    )

    if use_distillation:
        # Per-turn LLM distillation — each turn → 0..N canonical facts.
        for session_id, session_date, turns in sessions:
            iso = _iso_date(session_date)
            short = _short_date(session_date)
            for turn_idx, turn in enumerate(turns):
                turns_seen += 1
                text = (turn.content or "").strip()
                if not text:
                    skipped_empty += 1
                    continue
                if verbose:
                    print(f"    distill {session_id}#{turn_idx} role={turn.role}")
                facts = distill_turn(
                    role=turn.role,
                    content=text,
                    session_date=short,
                    model=distill_model,
                    api_key=api_key,
                )
                if not facts:
                    skipped_by_distiller += 1
                    continue
                for fact_idx, fact in enumerate(facts):
                    distilled_facts += 1
                    # Still prefix with date for belt-and-suspenders — the
                    # distiller SHOULD have resolved dates, but if it drifted,
                    # the inline prefix is a safety net.
                    content_with_date = fact.content
                    if short and short not in fact.content:
                        content_with_date = f"[{short}] {fact.content}"
                    # v3-ablate-A: category/entity/tags identical to v2.
                    # Structured fields live ONLY in metadata.jsonb (inert
                    # to the retrieval pipeline). This isolates whether
                    # structured extraction by itself regresses anything.
                    mem.add(
                        content=content_with_date,
                        tags=[turn.role or "unknown", session_id, "lme", "distilled"],
                        category="fact",
                        entity=None,
                        namespace=ns,
                        event_date=iso,
                        metadata={
                            "session_id": session_id,
                            "role": turn.role,
                            "turn_idx": turn_idx,
                            "fact_idx": fact_idx,
                            "source": "lme_distilled",
                            "distill_model": distill_model,
                            # Structured fields — stored but not surfaced
                            # in retrieval or answer context for ablation-A.
                            "speaker": fact.speaker,
                            "claim_type": fact.claim_type,
                            "emphasis": fact.emphasis,
                            "entities": list(fact.entities),
                            "topics": list(fact.topics),
                        },
                    )
                    memories_added += 1
        return IngestStats(
            turns_seen=turns_seen,
            memories_added=memories_added,
            sessions=len(sessions),
            skipped_empty=skipped_empty,
            distilled_facts=distilled_facts,
            skipped_by_distiller=skipped_by_distiller,
        )

    if use_extraction:
        # Lazy import to keep the hot path (raw) free of extractor deps.
        from attestor.extraction.extractor import extract_from_session  # type: ignore

        for session_id, session_date, turns in sessions:
            # Map LongMemEval roles onto the speaker_a / speaker_b contract the
            # extractor expects.
            adapted_turns = [
                {
                    "speaker": "A" if t.role == "user" else "B",
                    "text": t.content,
                    "dia_id": f"{session_id}_t{idx}",
                }
                for idx, t in enumerate(turns)
                if t.content.strip()
            ]
            turns_seen += len(turns)
            skipped_empty += len(turns) - len(adapted_turns)
            if not adapted_turns:
                continue

            if verbose:
                print(f"    extracting {session_id} ({len(adapted_turns)} turns)")

            memories, _triples = extract_from_session(
                turns=adapted_turns,
                speaker_a="User",
                speaker_b="Assistant",
                session_date=_iso_date(session_date),
                model=extraction_model,
                api_key=api_key,
            )
            for m in memories:
                mem.add(
                    content=m.content,
                    tags=list(m.tags) + ["lme", session_id],
                    category=m.category,
                    entity=m.entity,
                    namespace=ns,
                    event_date=m.event_date or _iso_date(session_date),
                    confidence=m.confidence,
                    metadata={"session_id": session_id, "source": "lme_extracted"},
                )
                memories_added += 1
        return IngestStats(
            turns_seen=turns_seen,
            memories_added=memories_added,
            sessions=len(sessions),
            skipped_empty=skipped_empty,
        )

    # Raw path — option C: inline date tag in content AND event_date kwarg.
    for session_id, session_date, turns in sessions:
        iso = _iso_date(session_date)
        short = _short_date(session_date)
        if verbose:
            print(f"    raw ingest {session_id} ({len(turns)} turns) date={short}")
        for idx, turn in enumerate(turns):
            turns_seen += 1
            text = turn.content.strip()
            if not text:
                skipped_empty += 1
                continue
            mem.add(
                content=_format_turn_content(turn.role, text, short),
                tags=[turn.role or "unknown", session_id, "lme"],
                category="conversation",
                entity=None,
                namespace=ns,
                event_date=iso,
                metadata={
                    "session_id": session_id,
                    "role": turn.role,
                    "turn_idx": idx,
                    "source": "lme_raw",
                },
            )
            memories_added += 1

    return IngestStats(
        turns_seen=turns_seen,
        memories_added=memories_added,
        sessions=len(sessions),
        skipped_empty=skipped_empty,
    )


# ---------------------------------------------------------------------------
# Answer + judge
# ---------------------------------------------------------------------------

def is_recommendation_question(question: str, **_: Any) -> bool:
    """Deprecated: the unified ANSWER_PROMPT decides mode internally.

    Kept only so existing imports don't break. Always returns False now;
    the LLM handles mode selection inside a single prompt.
    """
    return False


def classify_question(question: str, **_: Any) -> int:
    """Deprecated: no separate classifier — see unified ANSWER_PROMPT."""
    return 0


def _answerer_call(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    question: str | None = None,
    context: str = "",
) -> str:
    """Produce the answerer's raw response, applying self-consistency
    or critique-and-revise when their YAML knobs are flipped on.

    Single-sample path (default): one greedy ``_chat`` call. Identical
    behavior to the legacy code site.

    Self-consistency path: K independent samples at non-zero
    temperature, reduced via majority vote (or judge_pick) to one
    final answer. Cost is K × answerer cost.

    Critique-revise path (Phase 3 PR-E): three-step pipeline — initial
    answer → critic LLM ground-checks against retrieved context → on
    VERDICT=revise, a second answerer call produces the corrected
    answer. ~3x answerer cost worst case (1x extra in the common
    pass case).

    Mutual exclusion: when both ``stack.self_consistency.enabled``
    and ``stack.critique_revise.enabled`` are True we prefer
    self-consistency (already shipped + proven) and log a warning;
    combining the two safely is a future PR.

    On any error in either layered path, falls back to the
    single-sample path so the LME run never breaks because of a
    config knob.

    Args:
        question: Optional question text — when omitted, the
            critique-revise path falls back to extracting the question
            from the user message in ``messages``. Pass it explicitly
            from the caller for cleaner critic prompts.
        context:  Optional retrieved-context block — embedded in the
            critique/revise prompts so the critic can ground-check
            the initial answer. Empty string degrades the critic but
            never breaks the call.
    """
    try:
        from attestor.config import get_stack
        stack = get_stack()
        sc = getattr(stack, "self_consistency", None)
        cr = getattr(stack, "critique_revise", None)
    except Exception:  # noqa: BLE001 — config errors must not break answerer
        sc = None
        cr = None

    sc_on = sc is not None and sc.enabled and sc.k > 1
    cr_on = cr is not None and cr.enabled

    if sc_on and cr_on:
        # Combining both layered strategies is a future PR; prefer the
        # one that's already shipped + proven (self-consistency).
        logger.warning(
            "stack.self_consistency and stack.critique_revise both "
            "enabled; preferring self_consistency for this run "
            "(critique_revise skipped).",
        )

    if sc_on:
        try:
            from attestor.longmemeval_consistency import (
                answer_with_self_consistency,
            )
            judge_model = sc.judge_model or stack.models.judge
            result = answer_with_self_consistency(
                client=client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                k=sc.k,
                temperature=sc.temperature,
                voter=sc.voter,
                judge_model=judge_model,
            )
            if result.chosen:
                return result.chosen
            # Empty consensus → fall through to single-sample retry.
        except Exception:  # noqa: BLE001
            # Self-consistency layer failed; degrade to single-sample.
            pass
    elif cr_on:
        try:
            from attestor.longmemeval_critique import (
                answer_with_critique_revise,
            )
            critic_model = cr.critic_model or stack.models.verifier
            revise_model = cr.revise_model or stack.models.answerer
            result = answer_with_critique_revise(
                client=client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                question=question,
                context=context,
                critic_model=critic_model,
                revise_model=revise_model,
                max_revisions=cr.max_revisions,
            )
            if result.final_answer:
                return result.final_answer
            # Empty final_answer → fall through to single-sample retry.
        except Exception:  # noqa: BLE001
            # Critique-revise layer failed; degrade to single-sample.
            pass

    return _chat(client, model, prompt, max_tokens=max_tokens, role="answerer")


def answer_question(
    mem: Any,
    sample: LMESample,
    *,
    budget: int = 4000,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_facts: int = 40,
    max_tokens: int = 1200,
    verify: bool = False,
    verify_model: str | None = None,
) -> AnswerResult:
    """Recall + synthesize an answer for a LongMemEval sample.

    Args:
        mem: ``AgentMemory`` instance, already populated via ``ingest_history``.
        sample: ``LMESample`` being answered.
        budget: Retrieval token budget for ``mem.recall``.
        model: OpenRouter model id for synthesis.
        api_key: Optional override for the configured provider's API
            key (see ``configs/attestor.yaml`` ``llm.providers.*.api_key_env``).
        max_facts: Cap on facts injected into the prompt (guards prompt size).
        max_tokens: Answerer ``max_tokens``. Bumped to 600 to accommodate the
            chain-of-thought <reasoning> block.
        verify: When True, run a second-pass verification that re-checks
            the first answer against the same facts. Catches arithmetic
            errors and over-abstention.
        verify_model: OpenRouter model id for the verifier. Defaults to
            ``model`` (same model self-verifying).

    Returns:
        ``AnswerResult`` with final answer + retrieval counts + latency +
        optional reasoning trace + verification flag.
    """
    import time

    ns = namespace_for(sample)
    t0 = time.monotonic()
    results = mem.recall(sample.question, budget=budget, namespace=ns) or []
    results = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)

    retrieved_session_ids = _extract_retrieved_session_ids(results[:max_facts])

    if not results:
        return AnswerResult(
            answer="I don't know.",
            retrieved_count=0,
            used_fact_count=0,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            retrieved_session_ids=retrieved_session_ids,
        )

    context = _format_recall_context(results, max_facts=max_facts)
    question_date = sample.question_date or "(unknown)"

    # Unified prompt: the model picks FACT vs RECOMMENDATION mode inside the
    # prompt (no separate classifier). Fact mode is strict; recommendation
    # mode weaves stored user preferences into a tailored proposal. See
    # ANSWER_PROMPT for the mode-decision rubric + worked examples.
    prompt = ANSWER_PROMPT.format(
        question=sample.question,
        question_date=question_date,
        context=context,
    )
    if api_key is not None:
        client = _get_client(api_key)
        clean_model = model
    else:
        client, clean_model = _get_client_for_model(model)
    raw = _answerer_call(
        client=client,
        model=clean_model,
        prompt=prompt,
        max_tokens=max_tokens,
        question=sample.question,
        context=context,
    ).strip()
    reasoning, first_answer = _strip_reasoning(raw)

    final_answer = first_answer
    verified = False
    raw_first = ""
    if verify:
        verify_prompt = VERIFY_PROMPT.format(
            question=sample.question,
            question_date=question_date,
            context=context,
            first_answer=first_answer,
        )
        # Verifier may target a different provider — re-route via the pool
        # so the provider-prefix is stripped before hitting the SDK.
        if api_key is not None:
            verify_client = client
            clean_verify_model = verify_model or model
        else:
            verify_client, clean_verify_model = _get_client_for_model(
                verify_model or model
            )
        verified_text = _chat(
            verify_client, clean_verify_model, verify_prompt,
            max_tokens=150, role="verifier",
        ).strip()
        # Only accept the verifier's output if it is a non-empty single line
        # that differs from the first answer. Preserve the verified=True flag
        # either way so telemetry records that the pass ran.
        verified = True
        cleaned = verified_text.splitlines()[0].strip() if verified_text else ""
        if cleaned:
            if cleaned != first_answer:
                raw_first = first_answer
                final_answer = cleaned
            else:
                final_answer = cleaned

    return AnswerResult(
        answer=final_answer,
        retrieved_count=len(results),
        used_fact_count=min(len(results), max_facts),
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        reasoning=reasoning,
        verified=verified,
        raw_first_answer=raw_first,
        retrieved_session_ids=retrieved_session_ids,
        predicted_mode=_parse_predicted_mode(reasoning),
        context=context,
    )


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------


async def _process_sample(
    sample: LMESample,
    *,
    mem_factory: Callable[[], Any],
    answer_model: str,
    judge_models: list[str],
    api_key: str | None,
    budget: int,
    use_extraction: bool,
    max_facts: int,
    use_distillation: bool = False,
    distill_model: str | None = None,
    verify: bool = False,
    verify_model: str | None = None,
) -> SampleReport:
    """Ingest → answer → judge one sample on its own AgentMemory instance.

    Isolation: each call creates its own ``AgentMemory`` via ``mem_factory``
    and closes it at the end. Per-sample namespace (``lme_<qid>``) keeps
    haystacks disjoint at the document layer; per-instance connections
    keep the Postgres driver's thread-unsafe connection objects disjoint
    at the transport layer. Both together give us correctness under
    per-sample concurrency.

    Judge calls inside a sample run in parallel via ``asyncio.gather`` —
    independent network round-trips, no shared state.

    A failing judge is recorded as WRONG but does NOT fail the sample.
    A failing ingest/answer re-raises so the top-level gather can count it.
    """
    # Resolve ingest/answer/judge through the parent package so that
    # ``monkeypatch.setattr(attestor.longmemeval, "ingest_history", ...)``
    # in tests still hijacks these call sites — matches the original
    # module-local lookup behavior before the package split.
    import attestor.longmemeval as _pkg

    mem = await asyncio.to_thread(mem_factory)
    try:
        stats: IngestStats = await asyncio.to_thread(
            _pkg.ingest_history,
            mem,
            sample,
            use_extraction=use_extraction,
            use_distillation=use_distillation,
            distill_model=distill_model,
            api_key=api_key,
            verbose=False,
        )
        ans: AnswerResult = await asyncio.to_thread(
            _pkg.answer_question,
            mem,
            sample,
            budget=budget,
            model=answer_model,
            api_key=api_key,
            max_facts=max_facts,
            verify=verify,
            verify_model=verify_model,
        )

        # Per-sample semaphore: caps in-flight judge LLM calls within
        # ONE sample. ``asyncio.to_thread`` is invoked lazily inside
        # ``_bounded_judge`` so the underlying judge function isn't
        # spawned until we own the semaphore slot — otherwise the
        # ``to_thread`` coros would queue with their own thread-pool
        # workers regardless of the gate.
        judge_sem = asyncio.Semaphore(_JUDGE_CONCURRENCY)

        async def _bounded_judge(jm: str) -> Any:
            async with judge_sem:
                return await asyncio.to_thread(
                    _pkg.judge_answer,
                    sample.question,
                    sample.answer,
                    ans.answer,
                    sample.question_type,
                    model=jm,
                    api_key=api_key,
                )

        judge_results = await asyncio.gather(
            *[_bounded_judge(jm) for jm in judge_models],
            return_exceptions=True,
        )
        judgments = {
            jm: _safe_judge_dict(jm, res)
            for jm, res in zip(judge_models, judge_results)
        }

        # Dimension B — multi-dimensional scoring computed inline so the
        # per-sample report is self-describing and post-hoc analysis doesn't
        # need to re-run anything.
        gold_sessions = tuple(sample.answer_session_ids)
        retrieved_sessions = ans.retrieved_session_ids
        gold_set = set(gold_sessions)
        overlap = sum(1 for s in retrieved_sessions if s in gold_set)
        retrieval_hit = overlap > 0
        predicted_mode = ans.predicted_mode

        # Personalization judge — only on samples the answerer claims are
        # RECOMMENDATION mode. One judge call per sample (not per
        # judge_model) — cheap; uses the first configured judge model.
        personalization_dict: dict | None = None
        if predicted_mode == "recommendation" and judge_models:
            judge_for_pers = judge_models[0]
            try:
                pj = await asyncio.to_thread(
                    judge_personalization,
                    sample.question,
                    sample.answer,
                    ans.answer,
                    ans.context,  # reuse the formatted context the answerer saw
                    model=judge_for_pers,
                    api_key=api_key,
                )
                personalization_dict = _judgement_to_dict(pj)
            except Exception as e:  # noqa: BLE001 — never sink the sample
                logger.warning(
                    "personalization judge failed for %s: %s",
                    sample.question_id, e,
                )
                personalization_dict = {
                    "label": "WRONG",
                    "correct": False,
                    "reasoning": f"personalization_judge_error: {type(e).__name__}: {e}",
                    "judge_model": f"{judge_for_pers}__personalization",
                }

        return SampleReport(
            question_id=sample.question_id,
            category=sample.question_type,
            question=sample.question,
            gold=sample.answer,
            answer=ans.answer,
            judgments=judgments,
            answer_latency_ms=ans.latency_ms,
            ingest_turns=stats.turns_seen,
            ingest_memories=stats.memories_added,
            retrieved_count=ans.retrieved_count,
            gold_session_ids=gold_sessions,
            retrieved_session_ids=retrieved_sessions,
            retrieval_hit=retrieval_hit,
            retrieval_overlap=overlap,
            predicted_mode=predicted_mode,
            personalization=personalization_dict,
        )
    finally:
        close = getattr(mem, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception as e:  # noqa: BLE001 — never mask the real error
                logger.warning("mem.close() failed: %s", e)


async def run_async(
    samples: list[LMESample],
    *,
    mem_factory: Callable[[], Any],
    answer_model: str = DEFAULT_MODEL,
    judge_models: list[str] | None = None,
    api_key: str | None = None,
    budget: int = 4000,
    use_extraction: bool = False,
    use_distillation: bool = False,
    distill_model: str | None = None,
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verify: bool = False,
    verify_model: str | None = None,
    verbose: bool = False,
    output_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    progress_callback: Callable[[int, int, SampleReport], None] | None = None,
) -> LMERunReport:
    """Parallel LongMemEval orchestrator — ingest → answer → judge per sample.

    Args:
        samples: ``LMESample`` list to score.
        mem_factory: Zero-arg callable that returns a fresh ``AgentMemory``.
            Called once PER SAMPLE so each task has isolated backend state.
        answer_model: OpenRouter model id for the answerer.
        judge_models: List of OpenRouter model ids. Multiple judges are
            called in parallel per sample and scored independently.
        api_key: Optional override for the configured provider's API
            key (see ``configs/attestor.yaml`` ``llm.providers.*.api_key_env``).
        budget: Recall token budget per question.
        use_extraction: Run the LLM extractor during ingest.
        max_facts: Cap on facts injected into the answerer prompt.
        parallel: Max concurrent samples. Default 4. Increase at your
            own rate-limit risk.
        verbose: Print per-sample verdicts to stdout as they arrive.
        output_path: Optional path to write the final JSON report.
        progress_callback: Optional ``(completed, total, sample_report)``
            hook fired as each sample finishes — useful for custom UIs.

    Returns:
        ``LMERunReport`` ordered the same way as ``samples`` (stable output
        despite concurrent execution).
    """
    from dataclasses import asdict

    judge_models = list(judge_models or list(DEFAULT_JUDGES))
    if distill_model is None or verify_model is None:
        from attestor.config import get_stack
        s = get_stack()
        if distill_model is None:
            distill_model = s.models.distill
        if verify_model is None and verify:
            verify_model = s.models.verifier
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    semaphore = asyncio.Semaphore(max(1, parallel))
    ordered_reports: list[SampleReport | None] = [None] * len(samples)

    async def _guarded(idx: int, sample: LMESample) -> None:
        async with semaphore:
            try:
                report = await _process_sample(
                    sample,
                    mem_factory=mem_factory,
                    answer_model=answer_model,
                    judge_models=judge_models,
                    api_key=api_key,
                    budget=budget,
                    use_extraction=use_extraction,
                    use_distillation=use_distillation,
                    distill_model=distill_model,
                    max_facts=max_facts,
                    verify=verify,
                    verify_model=verify_model,
                )
            except Exception as e:  # noqa: BLE001 — one bad sample must not sink the run
                logger.exception(
                    "sample %s failed; recording as all-WRONG",
                    sample.question_id,
                )
                report = SampleReport(
                    question_id=sample.question_id,
                    category=sample.question_type,
                    question=sample.question,
                    gold=sample.answer,
                    answer=f"pipeline_error: {type(e).__name__}: {e}",
                    judgments={
                        jm: {
                            "label": "WRONG",
                            "correct": False,
                            "reasoning": f"pipeline_error: {e}",
                            "judge_model": jm,
                        }
                        for jm in judge_models
                    },
                    answer_latency_ms=0.0,
                    ingest_turns=0,
                    ingest_memories=0,
                    retrieved_count=0,
                )
            ordered_reports[idx] = report
            if verbose:
                verdicts = ", ".join(
                    f"{jm.split('/')[-1]}={report.judgments[jm]['label']}"
                    for jm in judge_models
                )
                done = sum(1 for r in ordered_reports if r is not None)
                print(
                    f"[{done}/{len(samples)}] {sample.question_id} "
                    f"[{sample.question_type}] → {verdicts}",
                    flush=True,
                )
            if progress_callback is not None:
                done = sum(1 for r in ordered_reports if r is not None)
                progress_callback(done, len(samples), report)

    await asyncio.gather(
        *(_guarded(i, s) for i, s in enumerate(samples)), return_exceptions=False
    )

    sample_reports: list[SampleReport] = [r for r in ordered_reports if r is not None]
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_category, by_judge = _summarize(sample_reports, judge_models)

    # Inter-judge agreement — only meaningful with ≥2 judges.
    agreement: dict = {}
    if len(judge_models) >= 2:
        for a, b in itertools.combinations(judge_models, 2):
            both_correct = sum(
                1 for r in sample_reports
                if r.judgments.get(a, {}).get("correct")
                and r.judgments.get(b, {}).get("correct")
            )
            both_wrong = sum(
                1 for r in sample_reports
                if not r.judgments.get(a, {}).get("correct", True)
                and not r.judgments.get(b, {}).get("correct", True)
            )
            agree = both_correct + both_wrong
            total = len(sample_reports)
            agreement[f"{a}__vs__{b}"] = {
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "agreement_pct": round(100.0 * agree / total, 2) if total else 0.0,
            }

    by_judge_enriched = dict(by_judge)
    if agreement:
        by_judge_enriched["_inter_judge_agreement"] = agreement

    # Provenance — captured once per run for third-party verification.
    git_sha, git_dirty = _git_sha()
    ds_path_str = str(Path(dataset_path).expanduser().resolve()) if dataset_path else ""
    ds_sha = _sha256_file(ds_path_str) if ds_path_str and Path(ds_path_str).exists() else ""
    provenance = RunProvenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        attestor_version=_attestor_version(),
        python_version=sys.version.split()[0],
        platform=sys.platform,
        argv=tuple(sys.argv),
        dataset_path=ds_path_str,
        dataset_sha256=ds_sha,
        dataset_sample_count=len(samples),
        started_at_utc=started,
        completed_at_utc=completed,
    )

    # Echo the runtime config so the output is self-describing.
    run_config = {
        "answer_model": answer_model,
        "judge_models": list(judge_models),
        "budget": budget,
        "use_extraction": use_extraction,
        "use_distillation": use_distillation,
        "distill_model": distill_model if use_distillation else None,
        "max_facts": max_facts,
        "parallel": parallel,
        "verify": verify,
        "verify_model": verify_model if verify else None,
    }

    by_dimension = _summarize_dimensions(sample_reports)

    report = LMERunReport(
        total=len(sample_reports),
        answer_model=answer_model,
        judge_models=tuple(judge_models),
        by_category=by_category,
        by_judge=by_judge_enriched,
        started_at=started,
        completed_at=completed,
        samples=tuple(sample_reports),
        provenance=provenance,
        run_config=run_config,
        by_dimension=by_dimension,
    )

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(report), indent=2, sort_keys=False)
        out.write_text(payload)
        sidecar = out.with_suffix(out.suffix + ".sha256")
        sidecar.write_text(f"{_sha256_str(payload)}  {out.name}\n")
        logger.info("LongMemEval report written: %s (sha256 %s)", out, sidecar.name)

    return report


def run(
    samples: list[LMESample],
    *,
    mem_factory: Callable[[], Any] | None = None,
    mem: Any = None,
    answer_model: str = DEFAULT_MODEL,
    judge_models: list[str] | None = None,
    api_key: str | None = None,
    budget: int = 4000,
    use_extraction: bool = False,
    use_distillation: bool = False,
    distill_model: str | None = None,
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verify: bool = False,
    verify_model: str | None = None,
    verbose: bool = False,
    output_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    progress_callback: Callable[[int, int, SampleReport], None] | None = None,
) -> LMERunReport:
    """Synchronous entry point — thin wrapper over ``run_async``.

    Accepts either ``mem_factory`` (recommended — true per-sample isolation)
    or ``mem`` (single shared instance; parallel is forced to 1 for safety).
    """
    if mem_factory is None and mem is None:
        raise ValueError("run() requires mem_factory or mem")

    if mem_factory is None:
        # Legacy single-instance path: force serial to keep psycopg2 happy.
        shared_mem = mem
        def _single() -> Any:
            return shared_mem
        mem_factory = _single
        parallel = 1

    return asyncio.run(
        run_async(
            samples,
            mem_factory=mem_factory,
            answer_model=answer_model,
            judge_models=judge_models,
            api_key=api_key,
            budget=budget,
            use_extraction=use_extraction,
            use_distillation=use_distillation,
            distill_model=distill_model,
            max_facts=max_facts,
            parallel=parallel,
            verify=verify,
            verify_model=verify_model,
            verbose=verbose,
            output_path=output_path,
            dataset_path=dataset_path,
            progress_callback=progress_callback,
        )
    )
