"""Unit tests for attestor.longmemeval Phase 1 (schema, loader, date parser)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from attestor.longmemeval import (
    CATEGORY_NAMES,
    DATASET_VARIANTS,
    LMESample,
    LMETurn,
    TEMPORAL_CATEGORY,
    _coerce_sample,
    load_longmemeval,
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
