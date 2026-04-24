"""Unit tests for attestor.longmemeval Phase 1 (schema, loader, date parser)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from attestor.longmemeval import (
    ANSWER_PROMPT,
    CATEGORY_NAMES,
    DATASET_VARIANTS,
    DEFAULT_JUDGES,
    DEFAULT_MODEL,
    IngestStats,
    JUDGE_PROMPT,
    JudgeResult,
    LMESample,
    LMETurn,
    TEMPORAL_CATEGORY,
    VERIFY_PROMPT,
    DISTILL_PROMPT,
    LMERunReport,
    PERSONALIZATION_PROMPT,
    PERSONALIZATION_JUDGE_PROMPT,
    RunProvenance,
    SampleReport,
    _extract_retrieved_session_ids,
    _parse_predicted_mode,
    _summarize_dimensions,
    is_recommendation_question,
    _coerce_sample,
    _format_recall_context,
    _format_turn_content,
    _iso_date,
    _parse_distilled,
    _parse_judge_response,
    _safe_judge_dict,
    _sha256_file,
    _sha256_str,
    _short_date,
    _strip_reasoning,
    ingest_history,
    load_longmemeval,
    namespace_for,
    parse_lme_date,
    run,
    run_async,
)

FIXTURE = Path(__file__).parent / "fixtures" / "lme_mini.json"


@pytest.mark.unit
def test_fixture_exists() -> None:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"


@pytest.mark.unit
def test_load_longmemeval_mini() -> None:
    samples = load_longmemeval(FIXTURE)
    assert len(samples) == 6
    # Covers every declared category exactly once.
    seen = {s.question_type for s in samples}
    assert seen == set(CATEGORY_NAMES), f"missing categories: {set(CATEGORY_NAMES) - seen}"


@pytest.mark.unit
def test_load_longmemeval_returns_frozen_samples() -> None:
    samples = load_longmemeval(FIXTURE)
    s = samples[0]
    assert isinstance(s, LMESample)
    with pytest.raises((AttributeError, TypeError)):
        # Frozen dataclass — assignment must raise.
        s.answer = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_load_longmemeval_respects_limit() -> None:
    assert len(load_longmemeval(FIXTURE, limit=2)) == 2
    assert len(load_longmemeval(FIXTURE, limit=0)) == 0
    assert len(load_longmemeval(FIXTURE, limit=None)) == 6


@pytest.mark.unit
def test_load_longmemeval_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_longmemeval("/nonexistent/lme.json")


@pytest.mark.unit
def test_load_longmemeval_rejects_non_list(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        load_longmemeval(bad)


@pytest.mark.unit
def test_haystack_turns_are_typed() -> None:
    samples = load_longmemeval(FIXTURE)
    for s in samples:
        assert s.haystack_sessions, f"empty haystack for {s.question_id}"
        for session in s.haystack_sessions:
            for turn in session:
                assert isinstance(turn, LMETurn)
                assert turn.role in {"user", "assistant", ""}, turn.role


@pytest.mark.unit
def test_is_temporal_flag() -> None:
    samples = load_longmemeval(FIXTURE)
    temporal = [s for s in samples if s.is_temporal]
    assert len(temporal) == 1
    assert temporal[0].question_type == TEMPORAL_CATEGORY


@pytest.mark.unit
def test_total_haystack_turns() -> None:
    samples = load_longmemeval(FIXTURE)
    for s in samples:
        assert s.total_haystack_turns > 0


@pytest.mark.unit
def test_parse_lme_date_ok() -> None:
    dt = parse_lme_date("2023/05/30 (Tue) 23:40")
    assert isinstance(dt, datetime)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2023, 5, 30, 23, 40)


@pytest.mark.unit
def test_parse_lme_date_empty() -> None:
    assert parse_lme_date("") is None
    assert parse_lme_date("   ") is None


@pytest.mark.unit
def test_parse_lme_date_bad_format() -> None:
    assert parse_lme_date("not a date") is None
    assert parse_lme_date("2023-05-30") is None  # LOCOMO style — must fail for LME parser


@pytest.mark.unit
def test_dataset_variants_known() -> None:
    assert set(DATASET_VARIANTS) == {"oracle", "s", "m"}
    for fn in DATASET_VARIANTS.values():
        assert fn.endswith(".json")


@pytest.mark.unit
def test_iso_and_short_date_roundtrip() -> None:
    raw = "2023/05/30 (Tue) 23:40"
    assert _iso_date(raw) == "2023-05-30T23:40"
    assert _short_date(raw) == "2023-05-30"
    # Unparsable input is returned verbatim — never raises.
    assert _iso_date("garbage") == "garbage"
    assert _short_date("") == ""


@pytest.mark.unit
def test_format_turn_content_belt_and_suspenders() -> None:
    out = _format_turn_content("user", "hello", "2023-05-30")
    assert out == "[2023-05-30] User: hello"
    assert _format_turn_content("assistant", "hi", "2023-05-30") == "[2023-05-30] Assistant: hi"
    # Unknown role passes through.
    assert "mystery" in _format_turn_content("mystery", "?", "2023-01-01")


@pytest.mark.unit
def test_namespace_isolates_samples() -> None:
    samples = load_longmemeval(FIXTURE)
    ns = {namespace_for(s) for s in samples}
    assert len(ns) == len(samples), "collision in per-sample namespaces"
    for s, n in zip(samples, ns):
        pass  # namespaces unique across fixture


@pytest.mark.integration
def test_ingest_history_raw_end_to_end(mem) -> None:
    sample = load_longmemeval(FIXTURE)[0]
    stats = ingest_history(mem, sample, use_extraction=False)

    assert isinstance(stats, IngestStats)
    assert stats.turns_seen > 0
    assert stats.memories_added > 0
    assert stats.sessions == len(sample.haystack_sessions)
    # Every non-empty turn becomes exactly one memory.
    assert stats.turns_seen == stats.memories_added + stats.skipped_empty


@pytest.mark.integration
def test_ingest_history_namespaces_and_event_date(mem) -> None:
    sample = load_longmemeval(FIXTURE)[0]
    ingest_history(mem, sample, use_extraction=False)

    ns = namespace_for(sample)
    # Hit the document store directly so this test is decoupled from retrieval.
    all_mems = mem._store.list_all(namespace=ns)
    assert all_mems, "no memories persisted under the sample namespace"
    # Every stored memory must carry a non-empty event_date (belt) and the
    # inline [YYYY-MM-DD] tag (suspenders).
    for m in all_mems:
        assert m.event_date, f"memory missing event_date: {m.content[:80]!r}"
        assert m.content.startswith("["), f"missing inline date tag: {m.content[:40]!r}"
    # Sanity: sample-0's haystack dates are in 2023, so every event_date should
    # begin with 2023.
    years = {m.event_date[:4] for m in all_mems}
    assert years == {"2023"}, f"unexpected years in event_date: {years}"


@pytest.mark.integration
def test_ingest_history_isolates_namespaces(mem) -> None:
    samples = load_longmemeval(FIXTURE)[:2]
    for s in samples:
        ingest_history(mem, s, use_extraction=False)

    ns_a = namespace_for(samples[0])
    ns_b = namespace_for(samples[1])
    a = mem._store.list_all(namespace=ns_a)
    b = mem._store.list_all(namespace=ns_b)
    assert a and b, "both samples should have ingested memories"
    # No memory ids leak across namespaces.
    assert {m.id for m in a}.isdisjoint({m.id for m in b})


@pytest.mark.unit
def test_parse_judge_response_clean_json() -> None:
    label, reasoning = _parse_judge_response(
        '{"reasoning": "matches", "label": "CORRECT"}'
    )
    assert label == "CORRECT"
    assert reasoning == "matches"


@pytest.mark.unit
def test_parse_judge_response_wrong() -> None:
    label, reasoning = _parse_judge_response(
        '{"reasoning": "date mismatch", "label": "WRONG"}'
    )
    assert label == "WRONG"
    assert reasoning == "date mismatch"


@pytest.mark.unit
def test_parse_judge_response_markdown_fence() -> None:
    raw = "```json\n{\"reasoning\": \"ok\", \"label\": \"CORRECT\"}\n```"
    assert _parse_judge_response(raw) == ("CORRECT", "ok")


@pytest.mark.unit
def test_parse_judge_response_trailing_prose() -> None:
    raw = 'Here is my verdict.\n{"reasoning": "ok", "label": "CORRECT"}\nThanks.'
    assert _parse_judge_response(raw)[0] == "CORRECT"


@pytest.mark.unit
def test_parse_judge_response_broken_json_regex_fallback() -> None:
    raw = "reasoning: date is right. verdict: CORRECT"
    assert _parse_judge_response(raw)[0] == "CORRECT"


@pytest.mark.unit
def test_parse_judge_response_prefers_last_label() -> None:
    # In-reasoning mention of WRONG should NOT override the final CORRECT.
    raw = "The AI could have said WRONG things but its answer is CORRECT"
    assert _parse_judge_response(raw)[0] == "CORRECT"


@pytest.mark.unit
def test_parse_judge_response_defaults_to_wrong() -> None:
    # Bad judge output must never inflate accuracy.
    assert _parse_judge_response("")[0] == "WRONG"
    assert _parse_judge_response("   ")[0] == "WRONG"
    assert _parse_judge_response("shrug")[0] == "WRONG"


@pytest.mark.unit
def test_parse_judge_response_invalid_label_falls_through_to_regex() -> None:
    # Invalid label in JSON -> regex should still find "WRONG" in text
    raw = '{"label": "MAYBE", "reasoning": "WRONG answer"}'
    # Should find WRONG via regex fallback
    assert _parse_judge_response(raw)[0] == "WRONG"


@pytest.mark.unit
def test_format_recall_context_handles_duck_typed_results() -> None:
    class FakeMem:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeResult:
        def __init__(self, content: str) -> None:
            self.memory = FakeMem(content)
            self.score = 0.5

    ctx = _format_recall_context(
        [FakeResult("[2023-05-30] User: hi"), FakeResult("[2023-05-31] Assistant: hello")]
    )
    assert "User: hi" in ctx
    assert "Assistant: hello" in ctx
    # Max-facts cap is respected.
    ctx2 = _format_recall_context(
        [FakeResult(f"fact {i}") for i in range(100)], max_facts=3
    )
    assert ctx2.count("\n") == 2  # 3 lines = 2 newlines


@pytest.mark.unit
def test_answer_and_judge_prompts_include_required_placeholders() -> None:
    # Sanity: if someone removes a placeholder, .format() will raise KeyError.
    assert "{context}" in ANSWER_PROMPT
    assert "{question}" in ANSWER_PROMPT
    assert "{question_date}" in ANSWER_PROMPT
    assert "{question}" in JUDGE_PROMPT
    assert "{expected}" in JUDGE_PROMPT
    assert "{generated}" in JUDGE_PROMPT
    assert "{category}" in JUDGE_PROMPT


@pytest.mark.unit
def test_judge_result_shape() -> None:
    r = JudgeResult(
        label="CORRECT", correct=True, reasoning="ok", raw="x", judge_model="m"
    )
    assert r.correct is True
    with pytest.raises((AttributeError, TypeError)):
        r.label = "WRONG"  # type: ignore[misc]  # frozen


@pytest.mark.unit
def test_safe_judge_dict_handles_exception() -> None:
    d = _safe_judge_dict("openai/gpt-4.1-mini", RuntimeError("rate limited"))
    assert d["label"] == "WRONG"
    assert d["correct"] is False
    assert "rate limited" in d["reasoning"]
    assert d["judge_model"] == "openai/gpt-4.1-mini"


@pytest.mark.unit
def test_safe_judge_dict_handles_judge_result() -> None:
    r = JudgeResult(
        label="CORRECT", correct=True, reasoning="ok", raw="x", judge_model="m"
    )
    d = _safe_judge_dict("m", r)
    assert d["label"] == "CORRECT"
    assert d["correct"] is True


@pytest.mark.unit
def test_default_judges_has_two_providers() -> None:
    # Dual-judge default avoids answerer-judge collusion.
    assert len(DEFAULT_JUDGES) == 2
    # Distinct providers.
    providers = {j.split("/")[0] for j in DEFAULT_JUDGES}
    assert len(providers) == 2


@pytest.mark.unit
def test_run_async_parallel_preserves_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Samples completed out-of-order must still report in input order.

    Patches ingest/answer/judge to avoid network and simulate completion skew.
    """
    import asyncio
    import attestor.longmemeval as lme

    samples = load_longmemeval(FIXTURE)
    # Give each sample a different synthetic "compute cost" to force out-of-order completion.
    delays = {s.question_id: (i % 3) * 0.01 for i, s in enumerate(samples)}

    def fake_mem_factory() -> object:
        class _M:
            def close(self) -> None: pass
        return _M()

    async def _sleepy_ingest(mem, sample, **kwargs):
        await asyncio.sleep(delays[sample.question_id])
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _fake_ingest(mem, sample, **kwargs):
        # Sync version called via to_thread — emulate compute with the sleep.
        import time
        time.sleep(delays[sample.question_id])
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _fake_answer(mem, sample, **kwargs):
        from attestor.longmemeval import AnswerResult
        return AnswerResult(
            answer=f"answer-for-{sample.question_id}",
            retrieved_count=1,
            used_fact_count=1,
            latency_ms=1.0,
        )

    def _fake_judge(question, expected, generated, category, **kwargs):
        model = kwargs.get("model", "m")
        # Deterministic: every answer is CORRECT — isolates ordering from scoring.
        return JudgeResult(
            label="CORRECT", correct=True, reasoning="fake",
            raw="{}", judge_model=model,
        )

    monkeypatch.setattr(lme, "ingest_history", _fake_ingest)
    monkeypatch.setattr(lme, "answer_question", _fake_answer)
    monkeypatch.setattr(lme, "judge_answer", _fake_judge)

    report = asyncio.run(
        run_async(
            samples,
            mem_factory=fake_mem_factory,
            judge_models=["openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5"],
            parallel=3,
            api_key="dummy",  # unused — _chat is never called
        )
    )
    # Output order matches input order despite concurrent completion.
    assert tuple(r.question_id for r in report.samples) == tuple(
        s.question_id for s in samples
    )
    assert report.total == len(samples)
    # Both judges scored every sample.
    for sr in report.samples:
        assert set(sr.judgments) == {"openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5"}


@pytest.mark.unit
def test_run_async_inter_judge_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two judges disagreeing → agreement pct reflects it; aggregation correct."""
    import asyncio
    import attestor.longmemeval as lme

    samples = load_longmemeval(FIXTURE)[:4]

    def fake_mem_factory() -> object:
        class _M:
            def close(self) -> None: pass
        return _M()

    def _fake_ingest(mem, sample, **kwargs):
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _fake_answer(mem, sample, **kwargs):
        from attestor.longmemeval import AnswerResult
        return AnswerResult(answer="ans", retrieved_count=1, used_fact_count=1, latency_ms=0)

    # Judge A says CORRECT on samples 0,1; WRONG on 2,3.
    # Judge B says CORRECT on samples 0,3; WRONG on 1,2.
    # They agree on samples 0 (both CORRECT) and 2 (both WRONG) → 2/4 = 50%.
    a_verdicts = {samples[0].question_id: True, samples[1].question_id: True,
                  samples[2].question_id: False, samples[3].question_id: False}
    b_verdicts = {samples[0].question_id: True, samples[1].question_id: False,
                  samples[2].question_id: False, samples[3].question_id: True}

    def _fake_judge(question, expected, generated, category, **kwargs):
        model = kwargs.get("model", "m")
        # Recover the sample's question_id from the gold answer: the fixture's
        # samples have distinct gold strings so we use `expected` as a key.
        qid_by_gold = {s.answer: s.question_id for s in samples}
        qid = qid_by_gold.get(expected)
        correct = (a_verdicts if model == "A" else b_verdicts).get(qid, False)
        return JudgeResult(
            label="CORRECT" if correct else "WRONG",
            correct=correct, reasoning="fake", raw="{}", judge_model=model,
        )

    monkeypatch.setattr(lme, "ingest_history", _fake_ingest)
    monkeypatch.setattr(lme, "answer_question", _fake_answer)
    monkeypatch.setattr(lme, "judge_answer", _fake_judge)

    report = asyncio.run(
        run_async(samples, mem_factory=fake_mem_factory, judge_models=["A", "B"], parallel=2)
    )
    agreement = report.by_judge["_inter_judge_agreement"]
    key = "A__vs__B"
    assert agreement[key]["both_correct"] == 1
    assert agreement[key]["both_wrong"] == 1
    assert agreement[key]["agreement_pct"] == 50.0
    # Independent judge buckets.
    assert report.by_judge["A"]["correct"] == 2
    assert report.by_judge["B"]["correct"] == 2


@pytest.mark.unit
def test_run_async_records_pipeline_error_as_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sample whose ingest blows up must not sink the whole run."""
    import asyncio
    import attestor.longmemeval as lme

    samples = load_longmemeval(FIXTURE)[:3]
    bad_id = samples[1].question_id

    def fake_mem_factory() -> object:
        class _M:
            def close(self) -> None: pass
        return _M()

    def _ingest(mem, sample, **kwargs):
        if sample.question_id == bad_id:
            raise RuntimeError("kaboom")
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _answer(mem, sample, **kwargs):
        from attestor.longmemeval import AnswerResult
        return AnswerResult(answer="ans", retrieved_count=1, used_fact_count=1, latency_ms=0)

    def _judge(question, expected, generated, category, **kwargs):
        return JudgeResult(label="CORRECT", correct=True, reasoning="", raw="{}", judge_model=kwargs.get("model", "m"))

    monkeypatch.setattr(lme, "ingest_history", _ingest)
    monkeypatch.setattr(lme, "answer_question", _answer)
    monkeypatch.setattr(lme, "judge_answer", _judge)

    report = asyncio.run(
        run_async(samples, mem_factory=fake_mem_factory, judge_models=["m"], parallel=2)
    )
    assert report.total == 3
    bad = next(s for s in report.samples if s.question_id == bad_id)
    assert not any(j["correct"] for j in bad.judgments.values())
    assert "pipeline_error" in bad.answer
    # The other two samples scored normally.
    good = [s for s in report.samples if s.question_id != bad_id]
    assert all(j["correct"] for s in good for j in s.judgments.values())


@pytest.mark.unit
def test_strip_reasoning_extracts_final_answer() -> None:
    raw = "<reasoning>step 1, step 2</reasoning>\n7 days"
    reasoning, final = _strip_reasoning(raw)
    assert reasoning == "step 1, step 2"
    assert final == "7 days"


@pytest.mark.unit
def test_strip_reasoning_no_tags_treats_as_final() -> None:
    assert _strip_reasoning("just an answer") == ("", "just an answer")
    assert _strip_reasoning("") == ("", "")


@pytest.mark.unit
def test_strip_reasoning_handles_multiline_final_answer() -> None:
    raw = "<reasoning>work</reasoning>\n\n  `2 months`\n\nextra prose"
    reasoning, final = _strip_reasoning(raw)
    assert reasoning == "work"
    # Backticks and leading/trailing whitespace trimmed; first non-empty line wins.
    assert final == "2 months"


@pytest.mark.unit
def test_strip_reasoning_multiline_reasoning() -> None:
    raw = "<reasoning>line1\nline2\nline3</reasoning>\nfinal"
    reasoning, final = _strip_reasoning(raw)
    assert "line1" in reasoning and "line3" in reasoning
    assert final == "final"


@pytest.mark.unit
def test_answer_prompt_includes_arithmetic_guidance() -> None:
    # Guard against a future edit that silently removes the CoT / date-math section.
    assert "DATE ARITHMETIC" in ANSWER_PROMPT
    assert "<reasoning>" in ANSWER_PROMPT
    # Abstention guidance appears in the FACT-mode branch as "respond exactly: I don't know"
    # and in the RECOMMENDATION-mode branch as a prohibition; both must stay.
    assert "respond exactly: I don't know" in ANSWER_PROMPT
    assert "MUST NOT respond \"I don't know\"" in ANSWER_PROMPT


@pytest.mark.unit
def test_verify_prompt_includes_all_placeholders() -> None:
    for ph in ["{question}", "{question_date}", "{context}", "{first_answer}"]:
        assert ph in VERIFY_PROMPT, f"missing placeholder {ph} in VERIFY_PROMPT"


@pytest.mark.unit
def test_parse_distilled_bullet_lines() -> None:
    # Legacy prose fallback path — bullet lines still parse into structured
    # records with sensible defaults (speaker inferred from fallback,
    # claim_type='fact', emphasis='mentioned').
    raw = (
        "- The user works as a software engineer at Acme Corp as of 2023-05-30.\n"
        "- The user prefers Python over JavaScript.\n"
    )
    facts = _parse_distilled(raw, fallback_speaker="user")
    assert len(facts) == 2
    assert "Acme Corp" in facts[0].content
    assert "Python" in facts[1].content
    assert all(f.speaker == "user" for f in facts)
    assert all(f.claim_type == "fact" for f in facts)
    assert all(f.emphasis == "mentioned" for f in facts)


@pytest.mark.unit
def test_parse_distilled_accepts_alt_bullet_chars() -> None:
    raw = "* fact one\n• fact two\n- fact three"
    facts = _parse_distilled(raw)
    assert [f.content for f in facts] == ["fact one", "fact two", "fact three"]


@pytest.mark.unit
def test_parse_distilled_skip_returns_empty() -> None:
    assert _parse_distilled("SKIP") == []
    assert _parse_distilled("  skip  ") == []
    assert _parse_distilled("") == []


@pytest.mark.unit
def test_parse_distilled_ignores_non_bullet_lines() -> None:
    # Preamble prose that sometimes slips in must not be treated as a fact
    # when the model emitted bullets instead of JSON.
    raw = (
        "Here are the facts I found:\n"
        "- The user visited Paris on 2023-06-10.\n"
        "That's all I could extract."
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 1
    assert facts[0].content == "The user visited Paris on 2023-06-10."


@pytest.mark.unit
def test_parse_distilled_strips_markdown_fences() -> None:
    raw = "```\n- fact one\n- fact two\n```"
    facts = _parse_distilled(raw)
    assert [f.content for f in facts] == ["fact one", "fact two"]


@pytest.mark.unit
def test_parse_distilled_structured_json_array() -> None:
    # Preferred path: structured JSON record per fact. Populates all
    # downstream fields (speaker, claim_type, emphasis, entities, topics).
    raw = (
        "[\n"
        '  {"content": "The user prefers dark chocolate.", '
        '"speaker": "user", "claim_type": "preference", '
        '"emphasis": "explicit", "entities": ["dark chocolate"], '
        '"topics": ["food"]},\n'
        '  {"content": "The assistant recommended Roscioli.", '
        '"speaker": "assistant", "claim_type": "recommendation", '
        '"emphasis": "explicit", "entities": ["Roscioli"], '
        '"topics": ["restaurant", "italian"]}\n'
        "]"
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 2
    f0, f1 = facts
    assert f0.claim_type == "preference"
    assert f0.speaker == "user"
    assert f0.emphasis == "explicit"
    assert f0.entities == ("dark chocolate",)
    assert f0.topics == ("food",)
    assert f1.claim_type == "recommendation"
    assert f1.speaker == "assistant"
    assert f1.entities == ("Roscioli",)


@pytest.mark.unit
def test_parse_distilled_structured_normalizes_bad_enums() -> None:
    # LLMs occasionally hallucinate values outside the allowed vocab; the
    # parser must coerce silently rather than drop the record.
    raw = (
        "[\n"
        '  {"content": "x", "speaker": "SYSTEM", '
        '"claim_type": "weird_kind", "emphasis": "unknown"}\n'
        "]"
    )
    facts = _parse_distilled(raw, fallback_speaker="user")
    assert len(facts) == 1
    f = facts[0]
    assert f.speaker == "user"        # fallback used
    assert f.claim_type == "fact"      # coerced from unknown
    assert f.emphasis == "mentioned"   # coerced from unknown


@pytest.mark.unit
def test_parse_distilled_structured_drops_empty_content() -> None:
    raw = (
        "[\n"
        '  {"content": "", "speaker": "user"},\n'
        '  {"content": "real fact", "speaker": "user"}\n'
        "]"
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 1
    assert facts[0].content == "real fact"


@pytest.mark.unit
def test_parse_distilled_structured_fenced_array() -> None:
    raw = (
        "```json\n"
        "[\n"
        '  {"content": "The user visited Paris.", "speaker": "user", '
        '"claim_type": "event", "emphasis": "explicit"}\n'
        "]\n"
        "```"
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 1
    assert facts[0].claim_type == "event"
    assert facts[0].content == "The user visited Paris."


@pytest.mark.unit
def test_parse_distilled_structured_tolerates_preamble() -> None:
    raw = (
        "Sure! Here is the output:\n"
        "[\n"
        '  {"content": "fact", "speaker": "assistant", '
        '"claim_type": "fact", "emphasis": "explicit"}\n'
        "]\n"
        "Let me know if anything else is needed."
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 1
    assert facts[0].content == "fact"
    assert facts[0].speaker == "assistant"


@pytest.mark.unit
def test_parse_distilled_structured_entities_string_shape() -> None:
    # Models sometimes emit entities as a comma-delimited string instead of
    # a JSON array; normalize to tuple.
    raw = (
        "[\n"
        '  {"content": "fact", "speaker": "user", '
        '"claim_type": "fact", "emphasis": "explicit", '
        '"entities": "Rome, Italy", "topics": "travel;italy"}\n'
        "]"
    )
    facts = _parse_distilled(raw)
    assert len(facts) == 1
    assert facts[0].entities == ("Rome", "Italy")
    assert facts[0].topics == ("travel", "italy")


@pytest.mark.unit
def test_distill_prompt_includes_all_placeholders_and_rules() -> None:
    assert "{role}" in DISTILL_PROMPT
    assert "{content}" in DISTILL_PROMPT
    assert "{session_date}" in DISTILL_PROMPT
    # Non-negotiable rules that must stay in the prompt.
    assert "PRESERVE" in DISTILL_PROMPT or "preserve" in DISTILL_PROMPT
    assert "third person" in DISTILL_PROMPT
    assert "SKIP" in DISTILL_PROMPT
    assert "NEVER FABRICATE" in DISTILL_PROMPT


@pytest.mark.unit
def test_distill_prompt_preserves_assistant_facts() -> None:
    """Post-bugfix: the distiller MUST retain assistant-provided facts even
    when the user didn't explicitly commit to acting on them. Surfaced by
    single-session-assistant falling to 20-30% before the fix."""
    # The prompt must explicitly tell the model to keep assistant utterances.
    assert "The assistant told the user" in DISTILL_PROMPT
    # Explicit anti-regression on the "skip assistant echoes" rule.
    assert "The assistant recommended" in DISTILL_PROMPT
    # Guide against over-skipping.
    assert "When" in DISTILL_PROMPT and "doubt" in DISTILL_PROMPT and "KEEP" in DISTILL_PROMPT
    # Worked examples are in the prompt — model-as-few-shot.
    assert "Plesiosaur" in DISTILL_PROMPT
    assert "dark chocolate" in DISTILL_PROMPT


@pytest.mark.unit
def test_sha256_str_deterministic() -> None:
    assert _sha256_str("hello") == _sha256_str("hello")
    assert _sha256_str("hello") != _sha256_str("hellO")
    assert len(_sha256_str("x")) == 64


@pytest.mark.unit
def test_sha256_file_matches_str_hash(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"attestor audit")
    assert _sha256_file(p) == _sha256_str("attestor audit")


@pytest.mark.unit
def test_run_async_emits_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run must embed git SHA, dataset hash, argv, timestamps — the six
    verification artifacts. Also: the output JSON must have a sidecar
    .sha256 next to it matching the JSON's content hash.
    """
    import asyncio
    from dataclasses import asdict
    import attestor.longmemeval as lme

    samples = load_longmemeval(FIXTURE)[:2]
    ds_path = FIXTURE  # real file → real SHA

    def fake_mem_factory() -> object:
        class _M:
            def close(self) -> None: pass
        return _M()

    def _ingest(mem, sample, **kwargs):
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _answer(mem, sample, **kwargs):
        from attestor.longmemeval import AnswerResult
        return AnswerResult(answer="ok", retrieved_count=1, used_fact_count=1, latency_ms=0)

    def _judge(question, expected, generated, category, **kwargs):
        return JudgeResult(label="CORRECT", correct=True, reasoning="", raw="{}", judge_model=kwargs.get("model", "m"))

    monkeypatch.setattr(lme, "ingest_history", _ingest)
    monkeypatch.setattr(lme, "answer_question", _answer)
    monkeypatch.setattr(lme, "judge_answer", _judge)

    out = tmp_path / "report.json"
    report = asyncio.run(
        run_async(
            samples,
            mem_factory=fake_mem_factory,
            judge_models=["m"],
            parallel=2,
            output_path=out,
            dataset_path=ds_path,
        )
    )

    # Provenance block present with the six required fields.
    assert report.provenance is not None
    p = report.provenance
    assert isinstance(p, RunProvenance)
    assert p.git_sha  # may be "unknown" outside a git repo, never empty
    assert p.attestor_version
    assert p.python_version
    assert p.platform
    assert isinstance(p.argv, tuple) and len(p.argv) >= 1
    assert p.dataset_path == str(Path(ds_path).resolve())
    assert len(p.dataset_sha256) == 64  # real SHA256 since fixture exists
    assert p.dataset_sample_count == 2
    # Timestamps must be ISO-8601 UTC (end with +00:00 in from datetime.now(UTC))
    assert "T" in p.started_at_utc
    assert "T" in p.completed_at_utc

    # run_config echoes the knobs that shape the number.
    assert report.run_config["answer_model"]
    assert "use_distillation" in report.run_config
    assert "parallel" in report.run_config
    assert report.schema_version == "1.1"

    # Sidecar .sha256 file exists and matches the JSON's content.
    sidecar = out.with_suffix(out.suffix + ".sha256")
    assert sidecar.exists()
    line = sidecar.read_text().strip()
    parts = line.split("  ")
    assert len(parts) == 2
    sha, name = parts
    assert name == out.name
    # Recompute and compare — proves the sidecar is valid.
    assert sha == _sha256_str(out.read_text())


@pytest.mark.unit
def test_run_async_handles_missing_dataset_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run without dataset_path still produces provenance (empty SHA, empty path)."""
    import asyncio
    import attestor.longmemeval as lme

    samples = load_longmemeval(FIXTURE)[:1]

    def fake_mem_factory() -> object:
        class _M:
            def close(self) -> None: pass
        return _M()

    def _ingest(mem, sample, **kwargs):
        return IngestStats(turns_seen=1, memories_added=1, sessions=1, skipped_empty=0)

    def _answer(mem, sample, **kwargs):
        from attestor.longmemeval import AnswerResult
        return AnswerResult(answer="ok", retrieved_count=1, used_fact_count=1, latency_ms=0)

    def _judge(question, expected, generated, category, **kwargs):
        return JudgeResult(label="CORRECT", correct=True, reasoning="", raw="{}", judge_model=kwargs.get("model", "m"))

    monkeypatch.setattr(lme, "ingest_history", _ingest)
    monkeypatch.setattr(lme, "answer_question", _answer)
    monkeypatch.setattr(lme, "judge_answer", _judge)

    report = asyncio.run(
        run_async(samples, mem_factory=fake_mem_factory, judge_models=["m"], parallel=1)
    )
    assert report.provenance is not None
    assert report.provenance.dataset_path == ""
    assert report.provenance.dataset_sha256 == ""


@pytest.mark.unit
def test_answer_prompt_has_both_modes() -> None:
    # Anti-regression: the unified ANSWER_PROMPT must teach both modes.
    assert "FACT mode" in ANSWER_PROMPT
    assert "RECOMMENDATION mode" in ANSWER_PROMPT
    assert "DECIDE THE QUESTION MODE FIRST" in ANSWER_PROMPT
    # Must keep date-arithmetic rubric for fact temporal questions.
    assert "DATE ARITHMETIC" in ANSWER_PROMPT
    # Must keep CoT contract.
    assert "<reasoning>" in ANSWER_PROMPT
    # Must instruct not to abstain on recommendation questions.
    assert "MUST NOT respond \"I don't know\"" in ANSWER_PROMPT
    # Placeholders.
    assert "{context}" in ANSWER_PROMPT
    assert "{question}" in ANSWER_PROMPT
    assert "{question_date}" in ANSWER_PROMPT


@pytest.mark.unit
def test_answer_prompt_has_worked_examples_for_both_modes() -> None:
    # Fact example (temporal): museum / concert between-days arithmetic.
    assert "Rijksmuseum" in ANSWER_PROMPT
    # Fact example (recall + disambiguation): Orlando Sugar Factory rec.
    assert "Sugar Factory" in ANSWER_PROMPT
    # Recommendation example: Lisbon hotels.
    assert "Lisbon" in ANSWER_PROMPT and "boutique" in ANSWER_PROMPT
    # Recommendation example: kitchen tips.
    assert "magnetic knife strip" in ANSWER_PROMPT


@pytest.mark.unit
def test_answer_prompt_teaches_disambiguation_on_tagged_facts() -> None:
    # The disambiguation section must exist and name the structured tags.
    assert "FACT TAGS" in ANSWER_PROMPT
    assert "speaker=" in ANSWER_PROMPT
    assert "type=" in ANSWER_PROMPT
    assert "emphasis=" in ANSWER_PROMPT
    # Core disambiguation rule: explicit beats mentioned.
    assert "explicit" in ANSWER_PROMPT and "mentioned" in ANSWER_PROMPT
    # Personalization signal named.
    assert "type=preference" in ANSWER_PROMPT


@pytest.mark.unit
def test_distill_prompt_emits_json_schema() -> None:
    # Regression guard: the distiller must teach the model to emit
    # structured JSON with every required field.
    required_fields = [
        "\"content\"",
        "\"speaker\"",
        "\"claim_type\"",
        "\"emphasis\"",
        "\"entities\"",
        "\"topics\"",
    ]
    for f in required_fields:
        assert f in DISTILL_PROMPT, f"DISTILL_PROMPT missing field: {f}"
    # Vocabulary keywords the parser recognizes.
    for claim in ("preference", "recommendation", "event", "mentioned"):
        assert claim in DISTILL_PROMPT
    # SKIP sentinel still taught.
    assert "SKIP" in DISTILL_PROMPT


@pytest.mark.unit
def test_parse_predicted_mode_recognizes_both_modes() -> None:
    # Forms the unified ANSWER_PROMPT teaches the model to use.
    assert _parse_predicted_mode("Mode: FACT.\nDates ...") == "fact"
    assert _parse_predicted_mode("Mode: RECOMMENDATION.\nUser preferences...") == "recommendation"
    assert _parse_predicted_mode("This is a FACT mode question; ...") == "fact"
    assert _parse_predicted_mode("Treat this as RECOMMENDATION mode") == "recommendation"
    # No mode tokens → unknown (empty string).
    assert _parse_predicted_mode("Just some text without the keywords") == ""
    assert _parse_predicted_mode("") == ""


@pytest.mark.unit
def test_extract_retrieved_session_ids_pulls_from_metadata() -> None:
    # Duck-typed result objects mirroring what mem.recall returns.
    class _M:
        def __init__(self, sid: str | None) -> None:
            self.metadata = {"session_id": sid} if sid else {}
    class _R:
        def __init__(self, sid: str | None) -> None:
            self.memory = _M(sid)

    sids = _extract_retrieved_session_ids([
        _R("answer_280352e9"),
        _R("sharegpt_yywfIrx_0"),
        _R(None),                 # missing metadata → dropped, not substituted
        _R("answer_280352e9"),    # duplicates kept (precision math is set-based)
    ])
    assert sids == ("answer_280352e9", "sharegpt_yywfIrx_0", "answer_280352e9")


@pytest.mark.unit
def test_summarize_dimensions_basic_aggregation() -> None:
    # Three reports: one fact-mode CORRECT with retrieval hit; one rec-mode CORRECT
    # with personalization CORRECT; one fact-mode WRONG with no retrieval hit.
    samples = [
        SampleReport(
            question_id="q1", category="temporal-reasoning",
            question="?", gold="g1", answer="a1",
            judgments={"j": {"label": "CORRECT", "correct": True, "reasoning": "", "judge_model": "j"}},
            answer_latency_ms=0, ingest_turns=1, ingest_memories=1, retrieved_count=1,
            gold_session_ids=("answer_a",),
            retrieved_session_ids=("answer_a", "noise"),
            retrieval_hit=True, retrieval_overlap=1,
            predicted_mode="fact",
        ),
        SampleReport(
            question_id="q2", category="single-session-preference",
            question="?", gold="g2", answer="a2",
            judgments={"j": {"label": "CORRECT", "correct": True, "reasoning": "", "judge_model": "j"}},
            answer_latency_ms=0, ingest_turns=1, ingest_memories=1, retrieved_count=1,
            gold_session_ids=("answer_b",),
            retrieved_session_ids=("answer_b",),
            retrieval_hit=True, retrieval_overlap=1,
            predicted_mode="recommendation",
            personalization={"label": "CORRECT", "correct": True, "reasoning": "ok", "judge_model": "j__personalization"},
        ),
        SampleReport(
            question_id="q3", category="temporal-reasoning",
            question="?", gold="g3", answer="a3",
            judgments={"j": {"label": "WRONG", "correct": False, "reasoning": "", "judge_model": "j"}},
            answer_latency_ms=0, ingest_turns=1, ingest_memories=1, retrieved_count=1,
            gold_session_ids=("answer_c",),
            retrieved_session_ids=("noise",),
            retrieval_hit=False, retrieval_overlap=0,
            predicted_mode="fact",
        ),
    ]
    dim = _summarize_dimensions(samples)
    # Retrieval: 2 hits out of 3 with gold_session_ids → 66.67%
    assert dim["retrieval"] == {"hits": 2, "total": 3, "precision": 66.67}
    # Mode distribution: 2 fact, 1 recommendation, 0 unknown
    assert dim["mode_distribution"]["counts"] == {"fact": 2, "recommendation": 1, "unknown": 0}
    # Personalization: 1 correct out of 1 sample (only the rec-mode one)
    assert dim["personalization"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    # Per-predicted-mode answer accuracy
    assert dim["by_predicted_mode"]["fact"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert dim["by_predicted_mode"]["recommendation"] == {"correct": 1, "total": 1, "accuracy": 100.0}


@pytest.mark.unit
def test_summarize_dimensions_empty_safe() -> None:
    assert _summarize_dimensions([]) == {}


@pytest.mark.unit
def test_personalization_judge_prompt_has_required_placeholders() -> None:
    for ph in ["{question}", "{expected}", "{generated}", "{context}"]:
        assert ph in PERSONALIZATION_JUDGE_PROMPT, f"missing {ph}"
    # Anti-regression on key criteria text.
    assert "stored user fact" in PERSONALIZATION_JUDGE_PROMPT
    assert "Generic boilerplate" in PERSONALIZATION_JUDGE_PROMPT


@pytest.mark.unit
def test_legacy_is_recommendation_question_is_deprecated_noop() -> None:
    # The separate classifier was folded into the unified ANSWER_PROMPT;
    # the legacy helper stays as an import-compat shim and always returns
    # False so no one accidentally starts branching on it again.
    assert is_recommendation_question("Any tips for X?") is False
    assert is_recommendation_question("When did I go to Paris?") is False
    assert is_recommendation_question("") is False


@pytest.mark.unit
def test_coerce_sample_tolerates_stray_turn_keys() -> None:
    raw = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "?",
        "question_date": "",
        "answer": "yes",
        "answer_session_ids": ["a"],
        "haystack_dates": ["2023/01/01 (Sun) 00:00"],
        "haystack_session_ids": ["a"],
        "haystack_sessions": [[{"role": "user", "content": "hi", "extra": "ignored"}]],
    }
    s = _coerce_sample(raw)
    assert s.haystack_sessions[0][0].role == "user"
    assert s.haystack_sessions[0][0].content == "hi"
