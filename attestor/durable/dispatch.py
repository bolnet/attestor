# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Fire-and-forget workflow starts from synchronous callers.

``AgentMemory.add()`` is synchronous and may itself be running inside an
event loop (Starlette API, MCP server), so it can neither ``await`` nor
``asyncio.run``. This module owns ONE private daemon thread with its own
event loop; ``schedule_derive`` submits the start RPC there and returns
the deterministic workflow id immediately. The Temporal client cache is
per event loop (``client._cache_key``), so the dispatch loop keeps its
own connection.

Failures on the loop are logged (``attestor.durable.dispatch``) and never
reach the ingest path — unless the caller passes ``wait_seconds`` (CLI /
tests), in which case the result or the error is returned in-line.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from attestor.durable import client as durable_client
from attestor.durable.config import require_enabled
from attestor.durable.models import (
    DeriveRequest,
    ForgetRequest,
    derive_workflow_id,
    forget_workflow_id,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from concurrent.futures import Future

    from attestor.config import DurableCfg

logger = logging.getLogger("attestor.durable.dispatch")

DISPATCH_THREAD_NAME = "attestor-durable-dispatch"

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _dispatch_loop() -> asyncio.AbstractEventLoop:
    """The background loop, started on first use (daemon: never blocks exit)."""
    global _loop, _thread
    with _lock:
        if _loop is not None and _thread is not None and _thread.is_alive():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_loop, args=(loop,), name=DISPATCH_THREAD_NAME, daemon=True,
        )
        thread.start()
        _loop, _thread = loop, thread
        return loop


def submit(coro: Coroutine[Any, Any, Any]) -> Future[Any]:
    """Run ``coro`` on the dispatch loop; returns a concurrent ``Future``."""
    return asyncio.run_coroutine_threadsafe(coro, _dispatch_loop())


def reset_for_test() -> None:
    """Stop the background loop so each test starts clean."""
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop, _thread = None, None
    if loop is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5)
    durable_client.reset_client_cache()


async def start_derive_async(cfg: DurableCfg, request: DeriveRequest) -> str:
    """Connect + start ``derive-{memory_id}`` (module attribute: tests patch it)."""
    client = await durable_client.connect(cfg)
    return await durable_client.start_derive(client, cfg, request)


def _log_outcome(workflow_id: str, future: Future[Any]) -> None:
    exc = future.exception()
    if exc is None:
        logger.info("started %s", workflow_id)
        return
    logger.warning(
        "could not start %s (derived state stays unrepaired until rebuild): %s: %s",
        workflow_id, type(exc).__name__, exc,
    )


def schedule_derive(
    cfg: DurableCfg, request: DeriveRequest, *, wait_seconds: float | None = None,
) -> str:
    """Start ``DeriveMemory`` for ``request`` from sync code; return its workflow id.

    Raises ``DurableDisabledError`` when ``cfg.enabled`` is false. With
    ``wait_seconds`` the call blocks for the start RPC and re-raises its
    failure; otherwise it returns at once and the outcome is logged.
    """
    require_enabled(cfg)
    workflow_id = derive_workflow_id(request.memory.memory_id)
    future = submit(start_derive_async(cfg, request))
    if wait_seconds is None:
        future.add_done_callback(lambda f: _log_outcome(workflow_id, f))
        return workflow_id
    return future.result(timeout=wait_seconds)


async def start_forget_async(cfg: DurableCfg, request: ForgetRequest) -> str:
    """Connect + start ``forget-{user_id}-{audit_id}`` (module attribute: tests patch it)."""
    client = await durable_client.connect(cfg)
    return await durable_client.start_forget(client, cfg, request)


def schedule_forget(cfg: DurableCfg, request: ForgetRequest, *, wait_seconds: float) -> str:
    """Start ``ForgetUser`` from sync code and WAIT for the start RPC.

    A GDPR delete must be confirmed started, so unlike ``schedule_derive``
    this always blocks (bounded by ``wait_seconds``) and re-raises the
    connect / start failure — callers must not fall back silently.
    """
    require_enabled(cfg)
    if wait_seconds <= 0:
        raise ValueError(f"wait_seconds must be > 0; got {wait_seconds!r}")
    future = submit(start_forget_async(cfg, request))
    workflow_id = future.result(timeout=wait_seconds)
    logger.info("started %s", forget_workflow_id(request.user_id, request.audit_id))
    return workflow_id
