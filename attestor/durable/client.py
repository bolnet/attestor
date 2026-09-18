# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Temporal client: cached ``connect()`` + workflow start helpers.

* ``pydantic_data_converter`` so the frozen dataclasses in
  ``attestor.durable.models`` (with ``datetime`` fields) round-trip.
* One client per ``(address, namespace, tls)`` per event loop — the
  dispatch loop (``attestor.durable.dispatch``) keeps its own. The
  per-loop table is keyed on the loop OBJECT (weakly), never ``id()``:
  CPython recycles addresses, so an id-keyed cache could hand a new loop
  a client bound to a dead one.
* Every failure surfaces as a ``DurableError`` subclass — never a bare
  ``RuntimeError`` and never a silent ``None``.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Any

from attestor.durable.config import (
    DurableUnavailableError,
    require_enabled,
    require_temporalio,
)
from attestor.durable.models import (
    ConsolidateRequest,
    ConsolidationOutcome,
    DeriveOutcome,
    DeriveRequest,
    ForgetRequest,
    ForgetUserOutcome,
    RebuildOutcome,
    RebuildRequest,
    consolidate_workflow_id,
    derive_workflow_id,
    forget_workflow_id,
)

if TYPE_CHECKING:
    from attestor.config import DurableCfg

logger = logging.getLogger("attestor.durable.client")

_TargetKey = tuple[str, str, bool]
_LoopCache = dict[_TargetKey, Any]
# Entries vanish with their loop; no address reuse can alias a stale client.
_CLIENTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopCache] = (
    weakref.WeakKeyDictionary()
)
# Loop types without weakref support (some C-implemented loops) are held
# strongly instead: the loop then outlives the cache entry, which keeps
# its id() from being recycled while the entry exists.
_PINNED_CLIENTS: dict[asyncio.AbstractEventLoop, _LoopCache] = {}


def _target_key(cfg: DurableCfg) -> _TargetKey:
    return (cfg.address, cfg.namespace, cfg.tls)


def _loop_cache(loop: asyncio.AbstractEventLoop) -> _LoopCache:
    """The client table for ``loop`` (created on first use)."""
    try:
        return _CLIENTS.setdefault(loop, {})
    except TypeError:  # loop is not weak-referenceable
        return _PINNED_CLIENTS.setdefault(loop, {})


def cache_size() -> int:
    """Number of cached clients across all live loops (tests / diagnostics)."""
    tables = list(_CLIENTS.values()) + list(_PINNED_CLIENTS.values())
    return sum(len(table) for table in tables)


def reset_client_cache() -> None:
    """Drop cached clients (tests / reconnect after config change)."""
    _CLIENTS.clear()
    _PINNED_CLIENTS.clear()


async def connect(cfg: DurableCfg) -> Any:
    """Return a connected ``temporalio.client.Client`` for ``cfg``.

    Raises ``DurableDisabledError`` when ``cfg.enabled`` is false and
    ``DurableUnavailableError`` when the SDK is missing or the server
    refuses the connection.
    """
    require_enabled(cfg)
    require_temporalio()
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    table = _loop_cache(asyncio.get_running_loop())
    key = _target_key(cfg)
    cached = table.get(key)
    if cached is not None:
        return cached
    try:
        client = await Client.connect(
            cfg.address,
            namespace=cfg.namespace,
            tls=cfg.tls,
            data_converter=pydantic_data_converter,
        )
    except Exception as exc:  # every connect failure is terminal here
        raise DurableUnavailableError(
            f"Temporal server unreachable at {cfg.address} "
            f"(namespace={cfg.namespace}, tls={cfg.tls}): {exc}"
        ) from exc
    table[key] = client
    logger.info("connected to Temporal at %s namespace=%s", cfg.address, cfg.namespace)
    return client


async def start_consolidation(
    client: Any, cfg: DurableCfg, request: ConsolidateRequest,
) -> str:
    """Start ``ConsolidateEpisode`` for ``request``; return the workflow id.

    Idempotent: a workflow with the same deterministic id already running
    (or already completed within the id-reuse window) is treated as
    dispatched — the queue row was claimed once, and Temporal dedups.
    """
    from temporalio.exceptions import WorkflowAlreadyStartedError

    from attestor.durable.workflows.consolidate import ConsolidateEpisode

    workflow_id = consolidate_workflow_id(request.episode.episode_id)
    try:
        await client.start_workflow(
            ConsolidateEpisode.run,
            request,
            id=workflow_id,
            task_queue=cfg.task_queue,
        )
    except WorkflowAlreadyStartedError:
        logger.info("workflow %s already started; treating as dispatched", workflow_id)
    return workflow_id


async def consolidation_result(client: Any, workflow_id: str) -> ConsolidationOutcome:
    """Await a ``ConsolidateEpisode`` run by id, typed (raw handles return dicts)."""
    handle = client.get_workflow_handle(workflow_id, result_type=ConsolidationOutcome)
    return await handle.result()


async def _start_idempotent(
    client: Any, cfg: DurableCfg, workflow: Any, arg: Any, workflow_id: str,
) -> str:
    """``start_workflow`` treating an already-started id as dispatched."""
    from temporalio.exceptions import WorkflowAlreadyStartedError

    try:
        await client.start_workflow(workflow, arg, id=workflow_id, task_queue=cfg.task_queue)
    except WorkflowAlreadyStartedError:
        logger.info("workflow %s already started; treating as dispatched", workflow_id)
    return workflow_id


async def start_derive(client: Any, cfg: DurableCfg, request: DeriveRequest) -> str:
    """Start ``DeriveMemory`` as ``derive-{memory_id}``; return the workflow id.

    Idempotent by id: an ``add()`` repair and a rebuild for the same
    memory collapse into one execution.
    """
    from attestor.durable.workflows.derive import DeriveMemory

    return await _start_idempotent(
        client, cfg, DeriveMemory.run, request,
        derive_workflow_id(request.memory.memory_id),
    )


async def derive_result(client: Any, workflow_id: str) -> DeriveOutcome:
    """Await a ``DeriveMemory`` run by id, typed."""
    handle = client.get_workflow_handle(workflow_id, result_type=DeriveOutcome)
    return await handle.result()


async def start_rebuild(
    client: Any, cfg: DurableCfg, request: RebuildRequest, *, workflow_id: str,
) -> str:
    """Start ``RebuildDerived`` under the operator-chosen ``workflow_id``."""
    from attestor.durable.workflows.rebuild import RebuildDerived

    return await _start_idempotent(client, cfg, RebuildDerived.run, request, workflow_id)


async def rebuild_result(client: Any, workflow_id: str) -> RebuildOutcome:
    """Await a ``RebuildDerived`` run by id (follows continue-as-new), typed."""
    handle = client.get_workflow_handle(workflow_id, result_type=RebuildOutcome)
    return await handle.result()


async def start_forget(client: Any, cfg: DurableCfg, request: ForgetRequest) -> str:
    """Start ``ForgetUser`` as ``forget-{user_id}-{audit_id}``; return the workflow id.

    Idempotent by id: the caller mints ``audit_id`` once, so a retried
    start (or a double click) collapses into one saga and one audit row.
    """
    from attestor.durable.workflows.forget import ForgetUser

    return await _start_idempotent(
        client, cfg, ForgetUser.run, request,
        forget_workflow_id(request.user_id, request.audit_id),
    )


async def forget_result(client: Any, workflow_id: str) -> ForgetUserOutcome:
    """Await a ``ForgetUser`` run by id, typed."""
    handle = client.get_workflow_handle(workflow_id, result_type=ForgetUserOutcome)
    return await handle.result()
