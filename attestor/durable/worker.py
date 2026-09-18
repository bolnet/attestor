# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``attestor worker`` — one Temporal worker on the governance task queue.

Sandbox: the whole ``attestor`` prefix is passed through
(``SandboxRestrictions.default.with_passthrough_modules("attestor")``)
because the package ``__init__`` has import-time side effects the
sandbox forbids (spike 2026-09-03). Workflows still import only
``attestor.durable.models``; drivers and LLM clients stay in activities.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from attestor.durable import client as durable_client
from attestor.durable.activities.consolidation import ConsolidationActivities
from attestor.durable.activities.derive import DeriveActivities
from attestor.durable.activities.forget import ForgetActivities
from attestor.durable.activities.retention import RetentionActivities
from attestor.durable.activities.sessions import SessionActivities
from attestor.durable.config import require_enabled
from attestor.durable.workflows.consolidate import ConsolidateEpisode
from attestor.durable.workflows.derive import DeriveMemory
from attestor.durable.workflows.forget import ForgetUser
from attestor.durable.workflows.rebuild import RebuildDerived
from attestor.durable.workflows.retention import RetentionSweep
from attestor.durable.workflows.sessions import SessionSweep

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from attestor.config import DurableCfg

logger = logging.getLogger("attestor.durable.worker")

# Sync activities (psycopg2 + blocking LLM calls) run on this pool.
DEFAULT_ACTIVITY_THREADS = 4
PASSTHROUGH_MODULE_PREFIX = "attestor"
WORKFLOWS: tuple[type, ...] = (
    ConsolidateEpisode, DeriveMemory, RebuildDerived, ForgetUser, RetentionSweep, SessionSweep,
)


def _memory_provider(store_path: str) -> Callable[[], Any]:
    """One ``AgentMemory`` per worker, built on first activity call, not at start.

    The pool (``activity_threads``) shares this one Postgres connection.
    That is safe because every SQL path — ``PostgresBackend._execute``,
    the forget lanes, the retention helpers, ``SessionRepo`` — serialises
    on the connection's re-entrant lock (``attestor.store.conn_lock``),
    and the derive seam holds it across its whole RLS scope → read →
    write unit. Threads therefore overlap only on non-SQL work (LLM
    calls, Pinecone, Neo4j); Postgres statements never interleave.
    """
    cache: dict[str, Any] = {}
    # Activity threads race on the first call; two concurrent AgentMemory
    # constructions run schema DDL in parallel and deadlock Postgres.
    build_lock = threading.Lock()

    def make() -> Any:
        with build_lock:
            stale = cache.get("memory")
            if stale is not None and _backend_init_failed(stale):
                # Built during an outage: a derived store never initialised
                # and would stay None forever. Drop it so this attempt reconnects.
                logger.warning(
                    "durable worker: rebuilding AgentMemory (a backend failed to initialise)",
                )
                _close_quietly(stale)
                cache.pop("memory", None)
            if "memory" not in cache:
                from attestor.core import AgentMemory

                cache["memory"] = AgentMemory(store_path)
            return cache["memory"]

    return make


def _backend_init_failed(memory: Any) -> bool:
    return bool(
        getattr(memory, "_vector_init_failed", False)
        or getattr(memory, "_graph_init_failed", False)
    )


def _close_quietly(memory: Any) -> None:
    try:
        memory.close()
    except Exception as exc:
        logger.debug("durable worker: close of stale AgentMemory failed: %s", exc)


def _consolidator_provider(memory_provider: Callable[[], Any]) -> Callable[[], Any]:
    def make() -> Any:
        from attestor.consolidation import SleepTimeConsolidator

        return SleepTimeConsolidator(memory_provider())

    return make


def default_activities(store_path: str) -> list[Callable[..., Any]]:
    """The production activity set for ``store_path`` (lazy — no I/O yet)."""
    memory = _memory_provider(store_path)
    consolidation = ConsolidationActivities(_consolidator_provider(memory))
    derive = DeriveActivities(memory)
    forget = ForgetActivities(memory)
    retention = RetentionActivities(memory)
    sessions = SessionActivities(memory)
    return [
        consolidation.consolidate_episode,
        derive.embed_and_upsert,
        derive.graph_extract,
        derive.list_memory_ids,
        forget.write_forget_audit,
        forget.forget_doc,
        forget.forget_vector,
        forget.forget_graph,
        forget.forget_state,
        retention.list_due_policies,
        retention.apply_policy,
        sessions.sweep_idle,
        sessions.sweep_ended,
        sessions.sweep_archived,
    ]


def build_worker(
    client: Any,
    cfg: DurableCfg,
    *,
    activities: Sequence[Callable[..., Any]],
    activity_executor: Executor | None = None,
    activity_threads: int = DEFAULT_ACTIVITY_THREADS,
) -> Any:
    """Construct (but do not run) a ``temporalio.worker.Worker``."""
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import (
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )

    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        PASSTHROUGH_MODULE_PREFIX,
    )
    executor = activity_executor or ThreadPoolExecutor(
        max_workers=activity_threads, thread_name_prefix="attestor-activity",
    )
    return Worker(
        client,
        task_queue=cfg.task_queue,
        workflows=list(WORKFLOWS),
        activities=list(activities),
        activity_executor=executor,
        workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
    )


async def run_worker(
    cfg: DurableCfg,
    *,
    store_path: str,
    activities: Sequence[Callable[..., Any]] | None = None,
) -> None:
    """Connect (loudly) and poll ``cfg.task_queue`` until cancelled."""
    require_enabled(cfg)
    client = await durable_client.connect(cfg)
    acts = list(activities) if activities is not None else default_activities(store_path)
    worker = build_worker(client, cfg, activities=acts)
    logger.info(
        "attestor worker polling task_queue=%s namespace=%s address=%s workflows=%s",
        cfg.task_queue, cfg.namespace, cfg.address, [w.__name__ for w in WORKFLOWS],
    )
    await worker.run()
