"""Phase 9.1.1 — RegressionCase schema + YAML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.regression.cases import (
    RegressionCase, load_cases,
)


# ── Dataclass invariants ──────────────────────────────────────────────────


@pytest.mark.unit
def test_case_is_frozen() -> None:
    c = RegressionCase(id="x", description="d", category="c",
                       ingest=(), query="q?")
    with pytest.raises(Exception):  # FrozenInstanceError
        c.id = "y"  # type: ignore[misc]


@pytest.mark.unit
def test_case_requires_id() -> None:
    with pytest.raises(ValueError, match="id is required"):
        RegressionCase(id="", description="", category="c",
                       ingest=(), query="q?")


@pytest.mark.unit
def test_case_requires_query() -> None:
    with pytest.raises(ValueError, match="query is required"):
        RegressionCase(id="x", description="", category="c",
                       ingest=(), query="")


@pytest.mark.unit
def test_abstain_required_excludes_must_contain() -> None:
    """Logical contradiction: abstaining means recall returned nothing useful;
    asserting must_contain on top of that is incoherent."""
    with pytest.raises(ValueError, match="abstain_required"):
        RegressionCase(
            id="bad", description="", category="abstention",
            ingest=(), query="q?",
            must_contain=("foo",),
            abstain_required=True,
        )


@pytest.mark.unit
def test_time_window_must_be_paired() -> None:
    with pytest.raises(ValueError, match="time_window_start and time_window_end"):
        RegressionCase(
            id="bad", description="", category="temporal",
            ingest=(), query="q?",
            time_window_start="2026-01-01T00:00:00Z",
            # missing time_window_end
        )


# ── YAML loader ───────────────────────────────────────────────────────────


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "qa.yaml"
    p.write_text(body)
    return p


@pytest.mark.unit
def test_load_empty_yaml_returns_empty_list(tmp_path: Path) -> None:
    p = _write(tmp_path, "")
    assert load_cases(p) == []


@pytest.mark.unit
def test_load_minimal_case(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: simple_recall
  description: User states a fact, recall finds it
  category: preference
  ingest:
    - user: I love sushi
      assistant: Got it, sushi noted
  query: What food do I love?
  must_contain: ["sushi"]
""")
    cases = load_cases(p)
    assert len(cases) == 1
    c = cases[0]
    assert c.id == "simple_recall"
    assert c.category == "preference"
    assert len(c.ingest) == 1
    assert c.ingest[0].user == "I love sushi"
    assert c.ingest[0].assistant == "Got it, sushi noted"
    assert c.must_contain == ("sushi",)


@pytest.mark.unit
def test_load_rejects_non_list_top_level(tmp_path: Path) -> None:
    p = _write(tmp_path, "id: oops\nquery: q\n")
    with pytest.raises(ValueError, match="top-level must be a list"):
        load_cases(p)


@pytest.mark.unit
def test_load_rejects_unknown_case_keys(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: x
  query: q
  ingest: []
  bogus_field: 123
""")
    with pytest.raises(ValueError, match="unknown keys.*bogus_field"):
        load_cases(p)


@pytest.mark.unit
def test_load_rejects_unknown_round_keys(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: x
  query: q
  ingest:
    - user: hi
      assistant: hello
      typo_key: nope
""")
    with pytest.raises(ValueError, match="unknown keys.*typo_key"):
        load_cases(p)


@pytest.mark.unit
def test_load_rejects_round_missing_user(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: x
  query: q
  ingest:
    - assistant: only assistant present
""")
    with pytest.raises(ValueError, match="both 'user' and 'assistant'"):
        load_cases(p)


@pytest.mark.unit
def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: dup
  query: q1
  ingest: []
- id: dup
  query: q2
  ingest: []
""")
    with pytest.raises(ValueError, match="duplicate case id 'dup'"):
        load_cases(p)


@pytest.mark.unit
def test_load_round_with_timestamp(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: ts_case
  query: when?
  ingest:
    - user: meeting was monday
      assistant: noted
      ts: "2026-01-15T10:00:00Z"
""")
    cases = load_cases(p)
    assert cases[0].ingest[0].ts == "2026-01-15T10:00:00Z"


@pytest.mark.unit
def test_load_temporal_replay_fields(tmp_path: Path) -> None:
    p = _write(tmp_path, """
- id: replay
  query: what was the policy on jan 1?
  ingest: []
  as_of: "2026-01-01T00:00:00Z"
  time_window_start: "2026-01-01T00:00:00Z"
  time_window_end: "2026-01-31T23:59:59Z"
""")
    cases = load_cases(p)
    assert cases[0].as_of == "2026-01-01T00:00:00Z"
    assert cases[0].time_window_start == "2026-01-01T00:00:00Z"
    assert cases[0].time_window_end == "2026-01-31T23:59:59Z"
