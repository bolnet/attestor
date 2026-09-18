"""SleepTimeConsolidator.consolidate_claimed — re-read a claimed row by id.

Fake queue, no Postgres. This is the seam the durable activity uses so
that raw turn text never crosses the Temporal workflow boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from attestor.consolidation.consolidator import (
    ConsolidationResult,
    SleepTimeConsolidator,
)
from attestor.consolidation.queue import QueuedEpisode

TS = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _ep(eid: str = "ep-1", user_id: str = "u-1") -> QueuedEpisode:
    return QueuedEpisode(
        id=eid, user_id=user_id, session_id="s", thread_id="t",
        user_turn_text="I moved to Lisbon", assistant_turn_text="Noted.",
        user_ts=TS, assistant_ts=TS,
    )


class _FakeQueue:
    def __init__(self, claimed: dict[tuple[str, str], QueuedEpisode]) -> None:
        self._claimed = dict(claimed)
        self.lookups: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    def fetch_claimed(self, episode_id: str, *, user_id: str) -> QueuedEpisode | None:
        self.lookups.append((episode_id, user_id))
        return self._claimed.get((episode_id, user_id))

    def mark_failed(self, episode_id: str, error: str) -> None:
        self.failed.append((episode_id, error))


class _FakeMem:
    _store = object()


def _consolidator(queue: _FakeQueue, monkeypatch, *, seen: list[QueuedEpisode]):
    cons = SleepTimeConsolidator(_FakeMem(), model="stub", queue=queue)

    def fake_one(ep: QueuedEpisode) -> ConsolidationResult:
        seen.append(ep)
        return ConsolidationResult(
            episode_id=ep.id, user_facts=[], agent_facts=[], applied=[],
        )

    monkeypatch.setattr(cons, "_consolidate_one", fake_one)
    return cons


@pytest.mark.unit
def test_consolidate_claimed_loads_row_and_runs_it(monkeypatch):
    seen: list[QueuedEpisode] = []
    queue = _FakeQueue({("ep-1", "u-1"): _ep()})
    result = _consolidator(queue, monkeypatch, seen=seen).consolidate_claimed("ep-1", "u-1")
    assert result.ok
    assert queue.lookups == [("ep-1", "u-1")]
    assert [e.user_turn_text for e in seen] == ["I moved to Lisbon"]


@pytest.mark.unit
def test_consolidate_claimed_reports_missing_row_as_permanent(monkeypatch):
    seen: list[QueuedEpisode] = []
    queue = _FakeQueue({})
    result = _consolidator(queue, monkeypatch, seen=seen).consolidate_claimed("ep-x", "u-1")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("missing:")
    assert "ep-x" in result.error
    assert seen == []
    assert queue.failed == []  # nothing to mark: the row is not ours / not claimed


@pytest.mark.unit
def test_consolidate_claimed_is_scoped_to_the_owning_user(monkeypatch):
    """A row claimed for one user must not be consolidated under another."""
    seen: list[QueuedEpisode] = []
    queue = _FakeQueue({("ep-1", "u-1"): _ep()})
    result = _consolidator(queue, monkeypatch, seen=seen).consolidate_claimed("ep-1", "u-other")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("missing:")
    assert seen == []
