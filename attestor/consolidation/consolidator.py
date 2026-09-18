# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""SleepTimeConsolidator — per-episode background re-extraction (Phase 7.2).

Drains the ConsolidationQueue and re-runs the conversation pipeline
with a stronger model. Three artifacts per episode (roadmap §E.1):

  1. Refined facts via the existing ADD/UPDATE/INVALIDATE/NOOP path
     (with mode='consolidation' provenance)
  2. Optional session summary when this episode closes a thread
  3. Cross-thread reflection (Phase 7.3, hooked here but implemented
     in attestor.consolidation.reflection)

The consolidator is intentionally synchronous — the queue handles
parallelism via SKIP LOCKED. ``run_once`` processes a batch and
returns; ``run_forever`` is an asyncio loop for daemons.

Durable mode (``durable.enabled: true`` in configs/attestor.yaml):
``run_forever(durable=cfg)`` claims one batch and starts one
``ConsolidateEpisode`` Temporal workflow per episode
(``dispatch_durable``) instead of looping in-process. The
``attestor.durable`` import is lazy so plain installs never load
``temporalio``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from attestor.consolidation.queue import ConsolidationQueue, QueuedEpisode
from attestor.conversation.apply import AppliedDecision, apply_decisions
from attestor.conversation.turns import ConversationTurn
from attestor.extraction.conflict_resolver import resolve_conflicts
from attestor.extraction.round_extractor import (
    DEFAULT_MAX_TOKENS,
    ExtractedFact,
    extract_agent_facts,
    extract_user_facts,
)
from attestor.models import Memory

if TYPE_CHECKING:
    from attestor.config import DurableCfg
    from attestor.durable.models import DurableDispatch

logger = logging.getLogger("attestor.consolidation.consolidator")

def _default_consolidation_model() -> str:
    """Resolve the consolidation model from ``configs/attestor.yaml``.

    Consolidation is a heavier reasoning task than synchronous extraction
    so we use the verifier slot (Claude Sonnet by default in the
    canonical stack).
    """
    from attestor.config import get_stack
    return get_stack().models.verifier


# Eager-resolved at import for the in-module class default below.
# YAML lookup failures fall back to an empty string so importing the
# module in a YAML-less test environment doesn't crash; the SleepTime
# consolidator instances will surface a clear error at call time
# instead. Production callers always have YAML loaded.
try:
    DEFAULT_CONSOLIDATION_MODEL = _default_consolidation_model()
except Exception:  # noqa: BLE001
    DEFAULT_CONSOLIDATION_MODEL = ""


@dataclass(frozen=True)
class ConsolidationResult:
    """One episode consolidation outcome."""
    episode_id: str
    user_facts: list[ExtractedFact]
    agent_facts: list[ExtractedFact]
    applied: list[AppliedDecision]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def written_memory_ids(self) -> list[str]:
        return [a.memory_id for a in self.applied
                if a.memory_id and a.operation in {"ADD", "UPDATE", "INVALIDATE"}]


class SleepTimeConsolidator:
    """Per-episode worker that re-extracts with a stronger model.

    Constructor takes an AgentMemory plus the consolidation model name.
    For tests, inject ``extraction_client`` / ``resolver_client`` to use
    stubs.

    Idempotency: ConsolidationQueue.mark_done is called only after the
    apply step succeeds. A crashed worker's row gets reclaimed after
    queue_lock_seconds (default 600).
    """

    def __init__(
        self,
        mem: Any,                                  # AgentMemory
        *,
        model: str = DEFAULT_CONSOLIDATION_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cadence_seconds: int = 300,
        batch_size: int = 20,
        extraction_client: Any | None = None,
        resolver_client: Any | None = None,
        queue: ConsolidationQueue | None = None,
    ) -> None:
        self._mem = mem
        self._model = model
        self._max_tokens = max_tokens
        self._cadence = cadence_seconds
        self._batch_size = batch_size
        self._extraction_client = extraction_client
        self._resolver_client = resolver_client
        # Allow injection so tests can use a custom queue connection
        self._queue = queue or ConsolidationQueue(self._mem._store._conn)

    # ── Public ──────────────────────────────────────────────────────────

    def run_once(self, *, limit: int | None = None) -> list[ConsolidationResult]:
        """Drain one batch and return per-episode results."""
        n = limit or self._batch_size
        batch = self._queue.dequeue_batch(limit=n)
        if not batch:
            return []
        return [self._consolidate_one(ep) for ep in batch]

    async def run_forever(
        self,
        *,
        durable: DurableCfg | None = None,
        client: Any | None = None,
    ) -> DurableDispatch | None:
        """Daemon loop. Sleeps cadence_seconds when the queue is empty.

        Durable branch: when ``durable.enabled`` is true this does NOT
        loop — it claims one batch, starts one ``ConsolidateEpisode``
        workflow per episode (id ``consolidate-{episode_id}``) and
        returns the :class:`DurableDispatch`. The Temporal worker
        (``attestor worker``) owns retries + completion from there.
        ``durable=None`` or ``enabled=False`` keeps the legacy in-process
        loop byte-for-byte.

        A transient psycopg2 / Neo4j error inside ``run_once`` (e.g.
        ``dequeue_batch`` raising on a dropped connection) used to crash
        the entire daemon. The outer try/except now logs and sleeps,
        so transient failures recover automatically.
        """
        if durable is not None and durable.enabled:
            return await self.dispatch_durable(durable, client=client)
        while True:
            try:
                results = self.run_once()
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "consolidator.run_once failed: %s; sleeping before retry",
                    e,
                )
                await asyncio.sleep(self._cadence)
                continue
            if not results:
                await asyncio.sleep(self._cadence)
                continue
            ok = sum(1 for r in results if r.ok)
            logger.info(
                "consolidated %d/%d episodes (errors=%d)",
                ok, len(results), len(results) - ok,
            )

    async def dispatch_durable(
        self,
        durable: DurableCfg,
        *,
        client: Any | None = None,
        limit: int | None = None,
    ) -> DurableDispatch:
        """Claim one batch and start one Temporal workflow per episode.

        The queue row claim stays in Postgres (zero schema change). If
        the server is unreachable every claim is released; if a single
        start fails that row is released and the rest proceed — nothing
        is left stranded in ``processing`` until the lease expires.
        """
        from attestor.durable import client as durable_client
        from attestor.durable.config import require_enabled
        from attestor.durable.models import (
            ConsolidateRequest,
            DurableDispatch,
            EpisodeRef,
        )

        require_enabled(durable)
        batch = self._queue.dequeue_batch(limit=limit or self._batch_size)
        if not batch:
            return DurableDispatch()
        try:
            temporal = client if client is not None else await durable_client.connect(durable)
        except Exception:
            self._release_all(batch)
            raise

        started: list[str] = []
        released: list[str] = []
        for ep in batch:
            request = ConsolidateRequest(episode=EpisodeRef.from_queued(ep))
            try:
                started.append(
                    await durable_client.start_consolidation(temporal, durable, request)
                )
            except Exception as e:  # release + report, keep dispatching
                logger.warning("durable dispatch of episode %s failed: %s", ep.id, e)
                self._queue.release(ep.id)
                released.append(ep.id)
        logger.info(
            "durable dispatch: started %d workflow(s), released %d",
            len(started), len(released),
        )
        return DurableDispatch(
            workflow_ids=tuple(started), released_episode_ids=tuple(released),
        )

    def _release_all(self, batch: list[QueuedEpisode]) -> None:
        for ep in batch:
            try:
                self._queue.release(ep.id)
            except Exception as e:
                logger.error("release of episode %s failed: %s", ep.id, e)

    # ── Per-episode ─────────────────────────────────────────────────────

    def consolidate_claimed(self, episode_id: str, user_id: str) -> ConsolidationResult:
        """Re-read a claimed row by ``(episode_id, user_id)`` and consolidate it.

        Entry point for the durable activity: Temporal hands over ids
        only, so the conversation text is read from Postgres here. A row
        that is not ``processing`` for that user (never claimed, lease
        reclaimed, already done, or owned by someone else) is a
        permanent ``missing:`` error — retries cannot change it and the
        row is not ours to mark failed.
        """
        ep = self._queue.fetch_claimed(episode_id, user_id=user_id)
        if ep is None:
            error = f"missing: episode {episode_id} is not claimed for user {user_id}"
            logger.warning("consolidate_claimed: %s", error)
            return ConsolidationResult(
                episode_id=episode_id, user_facts=[], agent_facts=[],
                applied=[], error=error,
            )
        return self._consolidate_one(ep)

    def _consolidate_one(self, ep: QueuedEpisode) -> ConsolidationResult:
        # RLS scope: the consolidator must operate as the episode's user.
        # The queue runs on an admin connection (BYPASSRLS); we explicitly
        # set the var here so the apply step's writes/reads are scoped.
        if hasattr(self._mem._store, "_set_rls_user"):
            try:
                self._mem._store._set_rls_user(ep.user_id)
            except Exception as e:
                self._queue.mark_failed(ep.id, f"set_rls_user failed: {e}")
                return ConsolidationResult(
                    episode_id=ep.id, user_facts=[], agent_facts=[],
                    applied=[], error=f"rls: {e}",
                )

        try:
            user_turn = ConversationTurn(
                thread_id=ep.thread_id, speaker="user", role="user",
                content=ep.user_turn_text, ts=ep.user_ts,
            )
            assistant_turn = ConversationTurn(
                thread_id=ep.thread_id,
                speaker=ep.agent_id or "assistant",
                role="assistant",
                content=ep.assistant_turn_text, ts=ep.assistant_ts,
            )

            user_facts = extract_user_facts(
                user_turn, model=self._model,
                max_tokens=self._max_tokens,
                client=self._extraction_client,
            )
            agent_facts = extract_agent_facts(
                assistant_turn, model=self._model,
                max_tokens=self._max_tokens,
                client=self._extraction_client,
            )
            all_facts = user_facts + agent_facts

            existing = self._retrieve_similar(all_facts)
            decisions = resolve_conflicts(
                new_facts=all_facts,
                existing=existing,
                evidence_episode_id=ep.id,
                model=self._model,
                max_tokens=self._max_tokens,
                client=self._resolver_client,
            )

            applied = apply_decisions(
                decisions, mem=self._mem,
                user_id=ep.user_id, project_id=ep.project_id,
                session_id=ep.session_id, scope="user",
                extraction_model=f"consolidation:{self._model}",
                parent_agent_id=ep.agent_id,
            )
            self._queue.mark_done(ep.id)
            return ConsolidationResult(
                episode_id=ep.id,
                user_facts=user_facts, agent_facts=agent_facts,
                applied=applied,
            )
        except Exception as e:
            logger.warning("consolidate %s failed: %s", ep.id, e)
            self._queue.mark_failed(ep.id, str(e))
            return ConsolidationResult(
                episode_id=ep.id, user_facts=[], agent_facts=[],
                applied=[], error=str(e),
            )

    # ── Helpers ────────────────────────────────────────────────────────

    def _retrieve_similar(
        self, new_facts: list[ExtractedFact],
    ) -> list[Memory]:
        """Reuse the same fallback logic as ConversationIngest — but
        without going through mem.recall (which would require resolving
        identity again). We just hit the doc store directly.
        """
        if not new_facts:
            return []
        store = getattr(self._mem, "_store", None)
        if store is None or not hasattr(store, "list_memories"):
            return []
        seen: set = set()
        out: list[Memory] = []
        for fact in new_facts:
            try:
                if fact.entity:
                    rows = store.list_memories(
                        category=fact.category, entity=fact.entity,
                        status="active", limit=5,
                    )
                else:
                    rows = store.list_memories(
                        category=fact.category, status="active", limit=5,
                    )
            except Exception as e:
                logger.debug("similar-lookup failed: %s", e)
                continue
            for m in rows:
                if m.id in seen:
                    continue
                seen.add(m.id)
                out.append(m)
                if len(out) >= 5:
                    return out
        return out
