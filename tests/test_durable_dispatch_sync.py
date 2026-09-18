"""attestor.durable.dispatch — fire-and-forget bridge from sync callers.

``AgentMemory.add()`` is synchronous and may run inside an event loop
(Starlette). ``schedule_derive`` therefore hands the start RPC to a
private background loop and returns the deterministic workflow id at
once; failures are logged, never raised into the ingest path — unless
the caller opts in to ``wait_seconds``.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from attestor.config import DurableCfg  # noqa: E402
from attestor.durable import dispatch  # noqa: E402
from attestor.durable.config import DurableUnavailableError  # noqa: E402
from attestor.durable.models import DeriveRef, DeriveRequest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_dispatch_loop():
    dispatch.reset_for_test()
    yield
    dispatch.reset_for_test()


def _req(mid: str = "m-1") -> DeriveRequest:
    return DeriveRequest(memory=DeriveRef(memory_id=mid, namespace="ns"))


@pytest.mark.unit
def test_schedule_derive_returns_id_immediately_and_starts_in_background(monkeypatch):
    started = threading.Event()
    seen: dict = {}

    async def fake_start(cfg, request):
        seen["cfg"] = cfg
        seen["request"] = request
        seen["thread"] = threading.current_thread().name
        started.set()
        return "derive-m-1"

    monkeypatch.setattr(dispatch, "start_derive_async", fake_start)
    cfg = DurableCfg(enabled=True, task_queue="tq")
    wid = dispatch.schedule_derive(cfg, _req())
    assert wid == "derive-m-1"
    assert started.wait(5), "start RPC never ran on the dispatch loop"
    assert seen["request"] == _req()
    assert seen["cfg"] == cfg
    assert seen["thread"] != threading.current_thread().name


@pytest.mark.unit
def test_schedule_derive_wait_blocks_for_result(monkeypatch):
    async def fake_start(cfg, request):
        await asyncio.sleep(0.01)
        return "derive-m-1"

    monkeypatch.setattr(dispatch, "start_derive_async", fake_start)
    wid = dispatch.schedule_derive(DurableCfg(enabled=True), _req(), wait_seconds=5)
    assert wid == "derive-m-1"


@pytest.mark.unit
def test_schedule_derive_logs_but_does_not_raise_when_server_unreachable(monkeypatch, caplog):
    failed = threading.Event()

    async def fake_start(cfg, request):
        try:
            raise DurableUnavailableError("Temporal server unreachable at x:1")
        finally:
            failed.set()

    monkeypatch.setattr(dispatch, "start_derive_async", fake_start)
    with caplog.at_level(logging.WARNING, logger="attestor.durable.dispatch"):
        wid = dispatch.schedule_derive(DurableCfg(enabled=True), _req())
        assert wid == "derive-m-1"
        assert failed.wait(5)
        # Give the done-callback a moment to run on the loop thread.
        deadline = threading.Event()
        for _ in range(50):
            if any("derive-m-1" in r.message for r in caplog.records):
                break
            deadline.wait(0.05)
    assert any("unreachable" in r.message for r in caplog.records)


@pytest.mark.unit
def test_schedule_derive_wait_re_raises_failure(monkeypatch):
    async def fake_start(cfg, request):
        raise DurableUnavailableError("nope")

    monkeypatch.setattr(dispatch, "start_derive_async", fake_start)
    with pytest.raises(DurableUnavailableError, match="nope"):
        dispatch.schedule_derive(DurableCfg(enabled=True), _req(), wait_seconds=5)


@pytest.mark.unit
def test_schedule_derive_refuses_when_disabled():
    from attestor.durable.config import DurableDisabledError

    with pytest.raises(DurableDisabledError):
        dispatch.schedule_derive(DurableCfg(enabled=False), _req())


@pytest.mark.unit
def test_schedule_derive_works_from_inside_a_running_event_loop(monkeypatch):
    """A sync caller inside an async server must not hit 'loop already running'."""

    async def fake_start(cfg, request):
        return "derive-m-1"

    monkeypatch.setattr(dispatch, "start_derive_async", fake_start)

    async def caller() -> str:
        return dispatch.schedule_derive(DurableCfg(enabled=True), _req(), wait_seconds=5)

    assert asyncio.run(caller()) == "derive-m-1"
