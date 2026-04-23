"""Unit tests for attestor.longmemeval Phase 1 (schema, loader, date parser)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from attestor.longmemeval import (
    CATEGORY_NAMES,
    DATASET_VARIANTS,
    IngestStats,
    LMESample,
    LMETurn,
    TEMPORAL_CATEGORY,
    _coerce_sample,
    _format_turn_content,
    _iso_date,
    _short_date,
    ingest_history,
    load_longmemeval,
    namespace_for,
    parse_lme_date,
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
