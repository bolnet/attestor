"""Unit tests for evals/abstention/synthetic_loader.py.

Pure tests on the fixture loader — no DB, no LLM. The end-to-end
AbstentionBench wiring is exercised by the user via the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.abstention.synthetic_loader import (
    SyntheticLoader, load_synthetic_samples,
)
from evals.abstention.types import AbstentionSample


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "evals" / "abstention" / "fixtures.json"


# ── load_synthetic_samples — happy path ───────────────────────────────


@pytest.mark.unit
def test_load_returns_30_samples() -> None:
    samples = load_synthetic_samples(FIXTURES)
    assert len(samples) == 30


@pytest.mark.unit
def test_samples_split_15_answerable_15_not() -> None:
    samples = load_synthetic_samples(FIXTURES)
    answerable = [s for s in samples if s.answerable]
    unanswerable = [s for s in samples if not s.answerable]
    assert len(answerable) == 15
    assert len(unanswerable) == 15


@pytest.mark.unit
def test_all_sample_ids_unique() -> None:
    samples = load_synthetic_samples(FIXTURES)
    ids = [s.sample_id for s in samples]
    assert len(set(ids)) == len(ids)


@pytest.mark.unit
def test_all_samples_are_frozen_dataclasses() -> None:
    """AbstentionSample is frozen — confirms our loader produces the
    right type (not e.g. a dict that quacks)."""
    samples = load_synthetic_samples(FIXTURES)
    s = samples[0]
    assert isinstance(s, AbstentionSample)
    with pytest.raises(Exception):
        s.query = "mutated"   # type: ignore[misc]


@pytest.mark.unit
def test_categories_cover_six_types() -> None:
    """6 answerable categories + 6 unanswerable categories. Specific
    enumeration prevents typos in the JSON from going unnoticed."""
    samples = load_synthetic_samples(FIXTURES)
    cats = {s.category for s in samples}
    expected = {
        # Answerable
        "personal_attribute", "decision_record", "preference",
        "relationship", "count", "professional",
        # Unanswerable
        "unknown_topic", "false_premise", "future_event",
        "underspecified", "subjective_opinion", "absent_relationship",
    }
    assert cats == expected


@pytest.mark.unit
def test_unanswerable_samples_have_null_expected_answer() -> None:
    samples = load_synthetic_samples(FIXTURES)
    for s in samples:
        if not s.answerable:
            assert s.expected_answer is None, (
                f"{s.sample_id}: unanswerable but has expected_answer="
                f"{s.expected_answer!r}"
            )


@pytest.mark.unit
def test_answerable_samples_have_substring_expected() -> None:
    samples = load_synthetic_samples(FIXTURES)
    for s in samples:
        if s.answerable:
            assert s.expected_answer, (
                f"{s.sample_id}: answerable=True but expected_answer is empty"
            )
            # Sanity: the expected answer should appear in the context too,
            # otherwise the "answerable" label is wrong.
            assert s.expected_answer.lower() in s.context.lower(), (
                f"{s.sample_id}: expected_answer={s.expected_answer!r} not "
                f"found in context — would mark a real correct answer wrong"
            )


@pytest.mark.unit
def test_unanswerable_metadata_records_why() -> None:
    """Every unanswerable case should document WHY in metadata.why
    so future maintainers understand the test intent."""
    samples = load_synthetic_samples(FIXTURES)
    for s in samples:
        if not s.answerable:
            assert "why" in s.metadata, (
                f"{s.sample_id}: unanswerable case missing metadata.why"
            )


# ── SyntheticLoader class wrapper ─────────────────────────────────────


@pytest.mark.unit
def test_synthetic_loader_is_callable() -> None:
    """The class must be a zero-arg callable per the DatasetLoader
    Protocol in evals/abstention/runner.py."""
    loader = SyntheticLoader()
    samples = loader()
    assert len(samples) == 30


@pytest.mark.unit
def test_synthetic_loader_uses_default_path() -> None:
    """Default constructor → bundled fixtures.json."""
    loader = SyntheticLoader()
    assert loader.fixtures_path == FIXTURES
    assert loader.fixtures_path.exists()


@pytest.mark.unit
def test_synthetic_loader_accepts_custom_path(tmp_path) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({
        "version": 1,
        "samples": [
            {
                "sample_id": "x",
                "category": "test",
                "answerable": True,
                "context": "ctx",
                "query": "q",
                "expected_answer": "ctx",
            },
        ],
    }))
    loader = SyntheticLoader(custom)
    samples = loader()
    assert len(samples) == 1
    assert samples[0].sample_id == "x"


# ── Error paths ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_rejects_missing_top_level_samples_key(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 1, "wrong_key": []}))
    with pytest.raises(ValueError, match="samples"):
        load_synthetic_samples(p)


@pytest.mark.unit
def test_load_rejects_empty_samples_list(tmp_path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"version": 1, "samples": []}))
    with pytest.raises(ValueError, match="no samples"):
        load_synthetic_samples(p)


@pytest.mark.unit
def test_load_rejects_sample_missing_required_field(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "version": 1,
        "samples": [
            {
                "sample_id": "x",
                "category": "test",
                # answerable is missing
                "context": "ctx",
                "query": "q",
            },
        ],
    }))
    with pytest.raises(ValueError, match="answerable"):
        load_synthetic_samples(p)
