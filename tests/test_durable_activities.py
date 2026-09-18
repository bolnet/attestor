"""consolidate_episode activity — wraps SleepTimeConsolidator.consolidate_claimed.

No Temporal server needed: the activity method is a plain callable when
invoked directly. Two contracts under test: the activity re-reads the
claimed row by id (turn text never crosses the workflow boundary), and
failure classification (retryable vs permanent).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from temporalio.exceptions import ApplicationError  # noqa: E402

from attestor.consolidation.consolidator import ConsolidationResult  # noqa: E402
from attestor.consolidation.queue import QueuedEpisode  # noqa: E402
from attestor.conversation.apply import AppliedDecision  # noqa: E402
from attestor.durable.models import EpisodeRef  # noqa: E402

TS = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _ref() -> EpisodeRef:
    return EpisodeRef(
        episode_id="ep-1", user_id="u-1", session_id="s-1", thread_id="t-1",
        user_ts=TS, assistant_ts=TS, project_id="p-1", agent_id="planner",
    )


class _FakeConsolidator:
    """Stands in for SleepTimeConsolidator: records the (id, user) lookups
    the activity performs and returns a scripted result."""

    def __init__(self, result: ConsolidationResult) -> None:
        self._result = result
        self.seen: list[tuple[str, str]] = []

    def consolidate_claimed(self, episode_id: str, user_id: str) -> ConsolidationResult:
        self.seen.append((episode_id, user_id))
        return self._result


def _applied(mid: str, op: str = "ADD") -> AppliedDecision:
    return AppliedDecision(operation=op, memory_id=mid)


@pytest.mark.unit
def test_activity_module_never_rebuilds_episode_from_payload():
    """Turn text must come from Postgres, not the Temporal payload."""
    import attestor.durable.activities.consolidation as mod

    assert not hasattr(mod, "episode_from_ref")
    assert QueuedEpisode is not None  # queue row type is what Postgres hands back


@pytest.mark.unit
def test_activity_reads_claimed_row_by_id_and_user():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[], applied=[],
    ))
    ConsolidationActivities(lambda: fake).consolidate_episode(_ref())
    assert fake.seen == [("ep-1", "u-1")]


@pytest.mark.unit
def test_activity_raises_non_retryable_when_row_is_missing():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[], applied=[],
        error="missing: episode ep-1 is not claimed for user u-1",
    ))
    with pytest.raises(ApplicationError) as exc:
        ConsolidationActivities(lambda: fake).consolidate_episode(_ref())
    assert exc.value.non_retryable
    assert exc.value.type == "ConsolidationPermanent"


@pytest.mark.unit
def test_activity_returns_outcome_on_success():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[],
        applied=[_applied("m-1"), _applied("m-2", "NOOP"), _applied("m-3", "UPDATE")],
    ))
    acts = ConsolidationActivities(lambda: fake)
    out = acts.consolidate_episode(_ref())
    assert out.ok
    assert out.episode_id == "ep-1"
    assert out.written_memory_ids == ("m-1", "m-3")
    assert fake.seen[0] == ("ep-1", "u-1")


@pytest.mark.unit
def test_activity_raises_retryable_on_transient_error():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[], applied=[], error="llm 503",
    ))
    with pytest.raises(ApplicationError) as exc:
        ConsolidationActivities(lambda: fake).consolidate_episode(_ref())
    assert not exc.value.non_retryable
    assert "llm 503" in str(exc.value)


@pytest.mark.unit
def test_activity_raises_non_retryable_on_permanent_error():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[], applied=[], error="rls: denied",
    ))
    with pytest.raises(ApplicationError) as exc:
        ConsolidationActivities(lambda: fake).consolidate_episode(_ref())
    assert exc.value.non_retryable


@pytest.mark.unit
def test_provider_is_lazy_and_asked_per_call():
    from attestor.durable.activities.consolidation import ConsolidationActivities

    fake = _FakeConsolidator(ConsolidationResult(
        episode_id="ep-1", user_facts=[], agent_facts=[], applied=[],
    ))
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        return fake

    acts = ConsolidationActivities(provider)
    assert calls == 0  # lazy: nothing built at registration
    acts.consolidate_episode(_ref())
    acts.consolidate_episode(_ref())
    # The provider owns caching (worker._memory_provider) and rebuilds a
    # memory whose backends failed to initialise; activities never pin it.
    assert calls == 2
