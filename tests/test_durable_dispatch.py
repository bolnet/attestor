"""SleepTimeConsolidator durable branch — one workflow per claimed episode.

Fake queue + fake Temporal client; no Postgres, no server. The last test
runs the real dispatch against the time-skipping server with a stub worker.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from attestor.config import DurableCfg
from attestor.consolidation.consolidator import SleepTimeConsolidator
from attestor.consolidation.queue import QueuedEpisode
from attestor.durable.models import (
    CONSOLIDATE_EPISODE_ACTIVITY,
    ConsolidationOutcome,
    EpisodeRef,
)

TS = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _ep(eid: str) -> QueuedEpisode:
    return QueuedEpisode(
        id=eid, user_id="u", session_id="s", thread_id="t",
        user_turn_text="q", assistant_turn_text="a", user_ts=TS, assistant_ts=TS,
    )


class _FakeQueue:
    def __init__(self, episodes: list[QueuedEpisode]) -> None:
        self._episodes = list(episodes)
        self.released: list[str] = []
        self.dequeue_limits: list[int] = []

    def dequeue_batch(self, limit: int = 20) -> list[QueuedEpisode]:
        self.dequeue_limits.append(limit)
        batch, self._episodes = self._episodes[:limit], self._episodes[limit:]
        return batch

    def release(self, episode_id: str) -> None:
        self.released.append(episode_id)


class _FakeMem:
    _store = object()


class _FakeClient:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.started: list[str] = []
        self.payloads: list[object] = []
        self._fail = fail_ids or set()

    async def start_workflow(self, workflow, arg, *, id, task_queue, **kw):
        if arg.episode.episode_id in self._fail:
            raise RuntimeError("rpc: unavailable")
        self.started.append(id)
        self.payloads.append(arg)
        return type("H", (), {"id": id})()


def _consolidator(queue: _FakeQueue) -> SleepTimeConsolidator:
    return SleepTimeConsolidator(_FakeMem(), model="stub", queue=queue, batch_size=2)


@pytest.mark.unit
async def test_dispatch_starts_one_workflow_per_claimed_episode():
    pytest.importorskip("temporalio")
    queue = _FakeQueue([_ep("a"), _ep("b"), _ep("c")])
    client = _FakeClient()
    cfg = DurableCfg(enabled=True, task_queue="tq")

    result = await _consolidator(queue).dispatch_durable(cfg, client=client)

    assert result.workflow_ids == ("consolidate-a", "consolidate-b")  # batch_size=2
    assert result.released_episode_ids == ()
    assert client.started == ["consolidate-a", "consolidate-b"]
    assert queue.dequeue_limits == [2]


@pytest.mark.unit
async def test_dispatch_payload_carries_ids_not_turn_text():
    """The claimed row's text stays in Postgres; only ids cross to Temporal."""
    import dataclasses

    pytest.importorskip("temporalio")
    ep = QueuedEpisode(
        id="pii", user_id="u", session_id="s", thread_id="t",
        user_turn_text="card 4111 1111 1111 1111", assistant_turn_text="ok",
        user_ts=TS, assistant_ts=TS,
    )
    client = _FakeClient()
    await _consolidator(_FakeQueue([ep])).dispatch_durable(
        DurableCfg(enabled=True), client=client,
    )
    payload = dataclasses.asdict(client.payloads[0])
    assert "4111" not in repr(payload)
    assert payload["episode"]["episode_id"] == "pii"
    assert payload["episode"]["user_id"] == "u"


@pytest.mark.unit
async def test_dispatch_releases_claim_when_start_fails():
    pytest.importorskip("temporalio")
    queue = _FakeQueue([_ep("a"), _ep("b")])
    client = _FakeClient(fail_ids={"b"})

    result = await _consolidator(queue).dispatch_durable(
        DurableCfg(enabled=True), client=client,
    )

    assert result.workflow_ids == ("consolidate-a",)
    assert result.released_episode_ids == ("b",)
    assert queue.released == ["b"]


@pytest.mark.unit
async def test_dispatch_refuses_when_disabled():
    from attestor.durable.config import DurableDisabledError

    pytest.importorskip("temporalio")
    queue = _FakeQueue([_ep("a")])
    with pytest.raises(DurableDisabledError):
        await _consolidator(queue).dispatch_durable(DurableCfg(enabled=False), client=_FakeClient())
    assert queue.dequeue_limits == []  # nothing claimed


@pytest.mark.unit
async def test_dispatch_releases_all_claims_when_server_unreachable(monkeypatch):
    pytest.importorskip("temporalio")
    from attestor.durable import client as durable_client
    from attestor.durable.config import DurableUnavailableError

    async def fail_connect(cfg):
        raise DurableUnavailableError("nope")

    monkeypatch.setattr(durable_client, "connect", fail_connect)
    queue = _FakeQueue([_ep("a"), _ep("b")])
    with pytest.raises(DurableUnavailableError):
        await _consolidator(queue).dispatch_durable(DurableCfg(enabled=True))
    assert sorted(queue.released) == ["a", "b"]


@pytest.mark.unit
async def test_run_forever_durable_branch_dispatches_and_returns():
    pytest.importorskip("temporalio")
    queue = _FakeQueue([_ep("a")])
    client = _FakeClient()
    cons = _consolidator(queue)

    result = await cons.run_forever(durable=DurableCfg(enabled=True), client=client)

    assert result.workflow_ids == ("consolidate-a",)


@pytest.mark.unit
async def test_run_forever_with_durable_disabled_stays_in_process(monkeypatch):
    """durable.enabled=false → legacy in-process loop (no Temporal import)."""
    import asyncio

    queue = _FakeQueue([])
    cons = _consolidator(queue)
    calls = {"run_once": 0}

    def fake_run_once(*, limit=None):
        calls["run_once"] += 1
        return []

    async def stop_loop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(cons, "run_once", fake_run_once)
    monkeypatch.setattr(asyncio, "sleep", stop_loop)
    with pytest.raises(asyncio.CancelledError):
        await cons.run_forever(durable=DurableCfg(enabled=False))
    assert calls["run_once"] == 1


@pytest.mark.integration
async def test_dispatch_end_to_end_on_test_server(temporal_env):
    """Real dispatch → real workflow → stub activity, on the time-skipping server."""
    from temporalio import activity

    from attestor.durable.client import consolidation_result
    from attestor.durable.worker import build_worker

    seen: list[str] = []

    @activity.defn(name=CONSOLIDATE_EPISODE_ACTIVITY)
    async def stub(ref: EpisodeRef) -> ConsolidationOutcome:
        seen.append(ref.episode_id)
        return ConsolidationOutcome(episode_id=ref.episode_id, ok=True)

    cfg = DurableCfg(enabled=True, task_queue=f"test-{uuid.uuid4().hex[:8]}")
    queue = _FakeQueue([_ep("e2e-1"), _ep("e2e-2")])
    async with build_worker(temporal_env.client, cfg, activities=[stub]):
        result = await _consolidator(queue).dispatch_durable(cfg, client=temporal_env.client)
        for wid in result.workflow_ids:
            out = await consolidation_result(temporal_env.client, wid)
            assert out.ok
    assert sorted(seen) == ["e2e-1", "e2e-2"]
