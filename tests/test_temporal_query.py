"""Phase 5.1 — TemporalQueryExpander + TimeWindow tests.

Stub LLM throughout — these are unit tests for parsing + dataclass
invariants, not LLM behaviour validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from attestor.retrieval.temporal_query import (
    TIME_EXTRACTION_PROMPT,
    TemporalQueryExpander,
    TimeWindow,
    _parse_iso,
    _parse_window,
)


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM
# ──────────────────────────────────────────────────────────────────────────


class _StubChoice:
    def __init__(self, c: str) -> None:
        self.message = type("M", (), {"content": c})()


class _StubResponse:
    def __init__(self, c: str) -> None:
        self.choices = [_StubChoice(c)]


class _StubChat:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc
        self.calls: list = []

    def create(self, **kw: Any) -> _StubResponse:
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _StubResponse(self._payload)


class StubClient:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self.chat = type("Chat", (), {"completions": _StubChat(payload, exc=exc)})()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)

    @property
    def last_prompt(self) -> str:
        msgs = self.chat.completions.calls[-1]["messages"]
        return msgs[0]["content"]


# ──────────────────────────────────────────────────────────────────────────
# TimeWindow dataclass
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_timewindow_accepts_iso_strings() -> None:
    tw = TimeWindow(start="2026-04-26T00:00:00Z", end="2026-04-26T23:59:59Z")
    assert tw.start.year == 2026 and tw.start.month == 4
    assert tw.end.tzinfo is timezone.utc


@pytest.mark.unit
def test_timewindow_accepts_datetimes() -> None:
    a = datetime(2026, 4, 26, tzinfo=timezone.utc)
    b = datetime(2026, 4, 27, tzinfo=timezone.utc)
    tw = TimeWindow(start=a, end=b)
    assert tw.start == a
    assert tw.end == b


@pytest.mark.unit
def test_timewindow_open_ended_start() -> None:
    tw = TimeWindow(start=None, end=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert tw.is_open_ended
    assert not tw.is_unbounded


@pytest.mark.unit
def test_timewindow_open_ended_end() -> None:
    tw = TimeWindow(start=datetime(2026, 1, 1, tzinfo=timezone.utc), end=None)
    assert tw.is_open_ended


@pytest.mark.unit
def test_timewindow_rejects_inverted_range() -> None:
    a = datetime(2026, 4, 26, tzinfo=timezone.utc)
    b = a + timedelta(days=-1)  # b < a
    with pytest.raises(ValueError, match="start.*end"):
        TimeWindow(start=a, end=b)


@pytest.mark.unit
def test_timewindow_naive_datetime_gets_utc() -> None:
    tw = TimeWindow(start="2026-04-26T00:00:00", end="2026-04-26T23:59:59")
    assert tw.start.tzinfo is timezone.utc
    assert tw.end.tzinfo is timezone.utc


@pytest.mark.unit
def test_parse_iso_handles_z_suffix() -> None:
    dt = _parse_iso("2026-04-26T10:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.hour == 10


@pytest.mark.unit
def test_parse_iso_returns_none_on_garbage() -> None:
    assert _parse_iso(None) is None
    assert _parse_iso("") is None
    assert _parse_iso("null") is None
    assert _parse_iso("not a date") is None


# ──────────────────────────────────────────────────────────────────────────
# _parse_window — LLM payload → TimeWindow
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_window_no_constraint_returns_none() -> None:
    raw = json.dumps({
        "has_time_constraint": False,
        "start": None, "end": None, "interpretation": "no time",
    })
    assert _parse_window(raw) is None


@pytest.mark.unit
def test_parse_window_full_range() -> None:
    raw = json.dumps({
        "has_time_constraint": True,
        "start": "2026-04-21T00:00:00Z",
        "end": "2026-04-21T23:59:59Z",
        "interpretation": "last Tuesday",
    })
    tw = _parse_window(raw)
    assert tw is not None
    assert tw.start.day == 21
    assert tw.interpretation == "last Tuesday"


@pytest.mark.unit
def test_parse_window_open_end_only() -> None:
    """'Before I had kids' → start=null, end=<date>"""
    raw = json.dumps({
        "has_time_constraint": True,
        "start": None,
        "end": "2020-01-01T00:00:00Z",
        "interpretation": "before kids",
    })
    tw = _parse_window(raw)
    assert tw is not None
    assert tw.start is None
    assert tw.end.year == 2020


@pytest.mark.unit
def test_parse_window_both_null_returns_none() -> None:
    """has_time_constraint=true but no actual bounds → unhelpful, treat as None."""
    raw = json.dumps({
        "has_time_constraint": True,
        "start": None, "end": None,
    })
    assert _parse_window(raw) is None


@pytest.mark.unit
def test_parse_window_handles_markdown_fence() -> None:
    raw = (
        "```json\n"
        + json.dumps({"has_time_constraint": True,
                      "start": "2026-04-21T00:00:00Z", "end": None})
        + "\n```"
    )
    tw = _parse_window(raw)
    assert tw is not None
    assert tw.start.day == 21


@pytest.mark.unit
def test_parse_window_bad_json_returns_none() -> None:
    assert _parse_window("not json") is None
    assert _parse_window("") is None
    assert _parse_window("[]") is None  # array, not object


@pytest.mark.unit
def test_parse_window_inverted_range_returns_none() -> None:
    """If LLM hallucinates start > end, drop the window rather than crash."""
    raw = json.dumps({
        "has_time_constraint": True,
        "start": "2026-05-01T00:00:00Z",
        "end": "2026-04-01T00:00:00Z",
    })
    assert _parse_window(raw) is None


# ──────────────────────────────────────────────────────────────────────────
# Expander.expand — end-to-end with stub LLM
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_expand_happy_path() -> None:
    payload = json.dumps({
        "has_time_constraint": True,
        "start": "2026-04-21T00:00:00Z",
        "end": "2026-04-21T23:59:59Z",
        "interpretation": "last Tuesday",
    })
    stub = StubClient(payload)
    exp = TemporalQueryExpander(client=stub)
    tw = exp.expand("what did I say last Tuesday?")
    assert tw is not None
    assert tw.start.day == 21
    assert "last Tuesday" in tw.interpretation


@pytest.mark.unit
def test_expand_empty_query_skips_llm() -> None:
    stub = StubClient(payload="should not be called")
    exp = TemporalQueryExpander(client=stub)
    assert exp.expand("") is None
    assert exp.expand("   ") is None
    assert stub.call_count == 0


@pytest.mark.unit
def test_expand_no_constraint_returns_none() -> None:
    payload = json.dumps({
        "has_time_constraint": False,
        "start": None, "end": None,
        "interpretation": "no time",
    })
    stub = StubClient(payload)
    exp = TemporalQueryExpander(client=stub)
    assert exp.expand("what's my favorite color?") is None


@pytest.mark.unit
def test_expand_llm_error_returns_none() -> None:
    stub = StubClient(exc=RuntimeError("api down"))
    exp = TemporalQueryExpander(client=stub)
    assert exp.expand("any time-sensitive question") is None


@pytest.mark.unit
def test_expand_passes_now_into_prompt() -> None:
    """The reference time must reach the prompt — caller controls 'now'."""
    payload = json.dumps({"has_time_constraint": False})
    stub = StubClient(payload)
    exp = TemporalQueryExpander(client=stub)
    fixed_now = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    exp.expand("recent stuff", now=fixed_now)
    assert "2026-04-26T10:00:00" in stub.last_prompt


@pytest.mark.unit
def test_expand_default_now_is_current_utc() -> None:
    """No now passed → uses datetime.now(UTC). Just check the call works."""
    payload = json.dumps({"has_time_constraint": False})
    stub = StubClient(payload)
    exp = TemporalQueryExpander(client=stub)
    exp.expand("anything")
    assert stub.call_count == 1


@pytest.mark.unit
def test_prompt_template_includes_examples() -> None:
    """Examples in the prompt are load-bearing for accuracy — don't drift."""
    assert "last Tuesday" in TIME_EXTRACTION_PROMPT
    assert "Before I had kids" in TIME_EXTRACTION_PROMPT
    assert "favorite color" in TIME_EXTRACTION_PROMPT
    assert "Recent" in TIME_EXTRACTION_PROMPT
    # Schema fields
    assert "has_time_constraint" in TIME_EXTRACTION_PROMPT
    assert "interpretation" in TIME_EXTRACTION_PROMPT
