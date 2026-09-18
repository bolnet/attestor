# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``AgentMemory.forget_user`` durable branch — start the ``ForgetUser`` saga.

When ``durable.enabled`` is true a real (non-dry-run) forget does NOT run
in-process: it mints an ``audit_id``, starts ``forget-{user_id}-{audit_id}``
and returns the workflow id so ``attestor durable status <id>`` can follow
it. Unlike the ``add()`` repair this is NOT fire-and-forget: the caller
must know the saga started, so the start RPC is awaited (bounded) and an
unreachable server raises ``DurableUnavailableError`` loudly — a GDPR
delete never silently downgrades to the in-process path.

``durable.enabled`` false (or no config) → ``None``: the caller stays on
today's in-process path, and ``attestor.durable`` is never imported.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from attestor.core.derive_service import resolve_durable_cfg

if TYPE_CHECKING:
    from attestor.config import DurableCfg
    from attestor.durable.models import ForgetRequest

logger = logging.getLogger("attestor.core.forget_service")

FORGET_START_TIMEOUT_SECONDS = 30.0
FOLLOW_HINT = "attestor durable status {workflow_id}"


def new_audit_id() -> str:
    return str(uuid.uuid4())


def start_forget(cfg: DurableCfg, request: ForgetRequest) -> str:
    """Start the saga from sync code (module attribute: tests patch it)."""
    from attestor.durable import dispatch

    return dispatch.schedule_forget(cfg, request, wait_seconds=FORGET_START_TIMEOUT_SECONDS)


def start_durable_forget(user_id: str, *, initiated_by: str | None) -> dict[str, Any] | None:
    """Start ``ForgetUser`` when durable is enabled; ``None`` means "run in-process"."""
    cfg = resolve_durable_cfg()
    if cfg is None or not cfg.enabled:
        return None
    from attestor.durable.models import ForgetRequest

    request = ForgetRequest(user_id=user_id, audit_id=new_audit_id(), initiated_by=initiated_by)
    workflow_id = start_forget(cfg, request)
    logger.info("forget_user %s started durably as %s", user_id, workflow_id)
    return {
        "user_id": user_id,
        "audit_id": request.audit_id,
        "workflow_id": workflow_id,
        "task_queue": cfg.task_queue,
        "durable": True,
        "dry_run": False,
        "status": "started",
        "follow": FOLLOW_HINT.format(workflow_id=workflow_id),
    }
