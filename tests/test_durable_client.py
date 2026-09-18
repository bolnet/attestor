"""Durable client — connect() gating, caching, and workflow start helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable.models import ConsolidateRequest, EpisodeRef  # noqa: E402


def _ref(eid: str = "ep-1") -> EpisodeRef:
    ts = datetime(2026, 9, 3, tzinfo=timezone.utc)
    return EpisodeRef(
        episode_id=eid, user_id="u", session_id="s", thread_id="t",
        user_ts=ts, assistant_ts=ts,
    )


@pytest.mark.unit
async def test_connect_refuses_when_disabled():
    from attestor.durable.client import connect
    from attestor.durable.config import DurableDisabledError

    with pytest.raises(DurableDisabledError):
        await connect(DurableCfg(enabled=False))


@pytest.mark.unit
async def test_connect_wraps_unreachable_server_as_durable_unavailable(monkeypatch):
    """A refused connection must surface as DurableUnavailableError (loud),
    never as a bare RuntimeError or a silent None."""
    import temporalio.client as tclient

    from attestor.durable import client as durable_client
    from attestor.durable.config import DurableUnavailableError

    async def fake_connect(*a, **kw):
        raise RuntimeError("Failed client connect: connection refused")

    monkeypatch.setattr(tclient.Client, "connect", fake_connect)
    durable_client.reset_client_cache()
    cfg = DurableCfg(enabled=True, address="127.0.0.1:1")
    with pytest.raises(DurableUnavailableError, match="127.0.0.1:1"):
        await durable_client.connect(cfg)


@pytest.mark.unit
async def test_connect_caches_per_target(monkeypatch):
    import temporalio.client as tclient

    from attestor.durable import client as durable_client

    calls: list[tuple] = []

    async def fake_connect(target_host, *, namespace, **kw):
        calls.append((target_host, namespace))
        return object()

    monkeypatch.setattr(tclient.Client, "connect", fake_connect)
    durable_client.reset_client_cache()
    cfg = DurableCfg(enabled=True, address="h:7233", namespace="ns")
    c1 = await durable_client.connect(cfg)
    c2 = await durable_client.connect(cfg)
    c3 = await durable_client.connect(DurableCfg(enabled=True, address="h:7233", namespace="other"))
    assert c1 is c2
    assert c3 is not c1
    assert calls == [("h:7233", "ns"), ("h:7233", "other")]
    durable_client.reset_client_cache()


@pytest.mark.unit
async def test_connect_cache_is_keyed_by_loop_identity_and_released_with_it(monkeypatch):
    """Two live loops never share a client, and a loop's entry disappears
    with the loop — so a recycled ``id()`` can never alias a stale client."""
    import asyncio
    import gc

    import temporalio.client as tclient

    from attestor.durable import client as durable_client

    calls: list[int] = []

    async def fake_connect(*a, **kw):
        calls.append(1)
        return object()

    monkeypatch.setattr(tclient.Client, "connect", fake_connect)
    durable_client.reset_client_cache()
    cfg = DurableCfg(enabled=True, address="h:7233", namespace="ns")
    here = await durable_client.connect(cfg)
    assert here is await durable_client.connect(cfg)

    def on_fresh_loop():
        loop = asyncio.new_event_loop()
        try:
            first = loop.run_until_complete(durable_client.connect(cfg))
            again = loop.run_until_complete(durable_client.connect(cfg))
            assert first is again
            assert durable_client.cache_size() == 2
            return first
        finally:
            loop.close()

    other = await asyncio.to_thread(on_fresh_loop)
    assert other is not here
    assert len(calls) == 2
    gc.collect()
    assert durable_client.cache_size() == 1  # the closed loop's entry went with it
    assert here is await durable_client.connect(cfg)
    assert len(calls) == 2
    durable_client.reset_client_cache()
    assert durable_client.cache_size() == 0


@pytest.mark.unit
async def test_connect_uses_pydantic_converter_and_tls_flag(monkeypatch):
    import temporalio.client as tclient
    from temporalio.contrib.pydantic import pydantic_data_converter

    from attestor.durable import client as durable_client

    seen: dict = {}

    async def fake_connect(target_host, **kw):
        seen.update(kw, target_host=target_host)
        return object()

    monkeypatch.setattr(tclient.Client, "connect", fake_connect)
    durable_client.reset_client_cache()
    await durable_client.connect(DurableCfg(enabled=True, tls=True))
    assert seen["data_converter"] is pydantic_data_converter
    assert seen["tls"] is True
    durable_client.reset_client_cache()


class _FakeHandle:
    def __init__(self, wid: str) -> None:
        self.id = wid


class _FakeClient:
    def __init__(self, *, already_started: bool = False) -> None:
        self.calls: list[dict] = []
        self._already = already_started

    async def start_workflow(self, workflow, arg, *, id, task_queue, **kw):
        self.calls.append({"workflow": workflow, "arg": arg, "id": id, "task_queue": task_queue})
        if self._already:
            from temporalio.exceptions import WorkflowAlreadyStartedError
            raise WorkflowAlreadyStartedError(id, "ConsolidateEpisode", run_id="r")
        return _FakeHandle(id)


@pytest.mark.unit
async def test_start_consolidation_uses_deterministic_id_and_task_queue():
    from attestor.durable.client import start_consolidation

    client = _FakeClient()
    cfg = DurableCfg(enabled=True, task_queue="tq")
    wid = await start_consolidation(client, cfg, ConsolidateRequest(episode=_ref("ep-9")))
    assert wid == "consolidate-ep-9"
    call = client.calls[0]
    assert call["id"] == "consolidate-ep-9"
    assert call["task_queue"] == "tq"
    assert isinstance(call["arg"], ConsolidateRequest)


@pytest.mark.unit
async def test_start_consolidation_is_idempotent_on_already_started():
    from attestor.durable.client import start_consolidation

    client = _FakeClient(already_started=True)
    wid = await start_consolidation(
        client, DurableCfg(enabled=True), ConsolidateRequest(episode=_ref("dup")),
    )
    assert wid == "consolidate-dup"
