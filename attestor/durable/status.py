# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Read-only introspection for ``attestor durable status``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from attestor.durable import client as durable_client
from attestor.durable.config import DurableError, DurableUnavailableError

if TYPE_CHECKING:
    from datetime import datetime

    from attestor.config import DurableCfg


@dataclass(frozen=True)
class DurableProbe:
    """Connectivity check result."""

    address: str
    namespace: str
    reachable: bool
    error: str | None = None


@dataclass(frozen=True)
class WorkflowStatus:
    """Compact view of one workflow execution."""

    workflow_id: str
    status: str
    run_id: str | None = None
    start_time: datetime | None = None
    close_time: datetime | None = None


async def probe(cfg: DurableCfg) -> DurableProbe:
    """Try to connect; never raises for an unreachable server."""
    try:
        await durable_client.connect(cfg)
    except DurableUnavailableError as exc:
        return DurableProbe(
            address=cfg.address, namespace=cfg.namespace, reachable=False, error=str(exc),
        )
    return DurableProbe(address=cfg.address, namespace=cfg.namespace, reachable=True)


async def describe_workflow(cfg: DurableCfg, workflow_id: str) -> WorkflowStatus:
    """Describe one execution; raises ``DurableError`` when it does not exist."""
    from temporalio.service import RPCError

    client = await durable_client.connect(cfg)
    try:
        desc = await client.get_workflow_handle(workflow_id).describe()
    except RPCError as exc:
        raise DurableError(f"workflow {workflow_id!r} not found: {exc}") from exc
    status = desc.status.name if desc.status is not None else "UNKNOWN"
    return WorkflowStatus(
        workflow_id=workflow_id,
        status=status,
        run_id=desc.run_id,
        start_time=desc.start_time,
        close_time=desc.close_time,
    )
