"""Retrieval orchestrator — public class composing helpers + postprocess
+ debug mixins.

Pipeline (permanent as of 2026-04-19):
    1. Vector semantic search (top-K=50)
    2. Graph narrow — soft boost by hop-distance from each hit's entity
       to the question entities (BFS depth ≤ 2 via Neo4j)
    3. Inject typed-edge triples as synthetic memories (for LLM reasoning)
    4. MMR diversity rerank (if enabled)
    5. Confidence decay + temporal boost (if enabled)
    6. Fit to token budget

Every call writes a JSONL trace to ``logs/attestor_trace.jsonl``
(disable via ``ATTESTOR_TRACE=0``).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from attestor.models import RetrievalResult
from attestor.retrieval.orchestrator.config import RetrievalRuntimeConfig
from attestor.retrieval.orchestrator.debug import _OrchestratorDebugMixin
from attestor.retrieval.orchestrator.helpers import _OrchestratorHelpersMixin
from attestor.retrieval.orchestrator.postprocess import (
    _OrchestratorPostProcessMixin,
)
from attestor.store.base import DocumentStore


class RetrievalOrchestrator(
    _OrchestratorHelpersMixin,
    _OrchestratorPostProcessMixin,
    _OrchestratorDebugMixin,
):
    """Semantic-first retrieval with graph narrowing.

    Tuning knobs live on :class:`RetrievalRuntimeConfig`, which is
    sourced from ``configs/attestor.yaml`` via
    :meth:`RetrievalRuntimeConfig.from_stack`. Pass ``config=`` to
    override; omit it to fall back to the historical literals (so
    unit tests that construct an orchestrator without YAML still work).

    For backward-compatibility, ``mmr_lambda`` and ``vector_top_k``
    remain readable / writable as instance attributes — the recall
    cascade reads them directly. ``AgentMemory`` continues to overwrite
    them post-construction; that path is preserved.
    """

    def __init__(
        self,
        store: DocumentStore,
        min_results: int = 3,
        vector_store=None,
        graph=None,
        enable_temporal_boost: bool = True,
        bm25_lane=None,
        *,
        config: RetrievalRuntimeConfig | None = None,
    ):
        self.store = store
        self.min_results = min_results
        self.vector_store = vector_store
        self.graph = graph
        self.enable_temporal_boost = enable_temporal_boost
        self.bm25_lane = bm25_lane            # Phase 4.3 — keyword/FTS lane
        self.bm25_top_k: int = 30             # how many BM25 hits to feed to RRF
        self.bm25_min_rank: float = 0.0       # drop hits below this rank
        self.confidence_gate: float = 0.0
        self.confidence_decay_rate: float = 0.001
        self.confidence_boost_rate: float = 0.03
        self.enable_mmr: bool = True
        # Retained for callers; has no effect in semantic-first pipeline.
        self.fusion_mode: str = "semantic_first"

        # Tuning config — single source of truth for the score-blending
        # constants (vector_weight, graph_weight, graph_affinity_bonus,
        # graph_unreachable_penalty) plus vector_top_k + mmr_lambda.
        self.config: RetrievalRuntimeConfig = (
            config if config is not None else RetrievalRuntimeConfig()
        )

        # Mirror the two most-frequently-overridden tuning fields onto
        # instance attributes so AgentMemory's post-construction
        # ``orch.vector_top_k = ...`` / ``orch.mmr_lambda = ...``
        # writes still work without needing to mutate a frozen dataclass.
        self.vector_top_k: int = self.config.vector_top_k
        self.mmr_lambda: float = self.config.mmr_lambda
        self.mmr_top_n: int | None = None  # None = no cap

        # Multi-query lane (Phase 3 PR-C). Defaults to None which means
        # "single-query path" — the legacy behavior. AgentMemory wires
        # the resolved MultiQueryCfg from configs/attestor.yaml after
        # construction. When ``multi_query_cfg.enabled`` is True, Step 1
        # runs N+1 vector searches (original + N rewrites) and merges
        # them via RRF before the rest of the cascade.
        self.multi_query_cfg = None  # type: ignore[assignment]

        # Temporal pre-filter (Phase 3 RC4). Defaults to None — no
        # behavior change. AgentMemory wires the resolved
        # TemporalPrefilterCfg post-construction. When
        # ``temporal_prefilter_cfg.enabled`` is True and the question
        # contains a relative time phrase, recall() builds a
        # ``TimeWindow`` and passes it through the existing
        # ``time_window`` kwarg before Step 1 runs.
        self.temporal_prefilter_cfg = None  # type: ignore[assignment]

        # HyDE retrieval (Phase 3 PR-D). Sibling of multi_query.
        # Defaults to None — no behavior change. AgentMemory wires
        # the resolved HydeCfg post-construction. Mutually exclusive
        # with multi_query in this PR — if both are enabled, the
        # orchestrator prefers multi_query (it shipped first + has
        # the longer track record) and logs a warning.
        self.hyde_cfg = None  # type: ignore[assignment]

    # ── Public API ──

    def recall(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: str | None = None,
        as_of: Any | None = None,           # datetime; Phase 5.3
        time_window: Any | None = None,     # TimeWindow; Phase 5.3
    ) -> list[RetrievalResult]:
        """Semantic-first recall with graph narrowing.

        Phase 5.3 — bi-temporal:
          as_of       — replay past belief: see only memories the system
                        BELIEVED were true at that timestamp
          time_window — pre-filter by EVENT-time overlap (when the fact
                        was true in the world)

        Both kwargs are passed through to the vector + BM25 lanes; lanes
        that don't support them ignore them silently. Behavior unchanged
        when both are None.
        """
        # Audit invariants A2 + A5 are enforced via two scopes opened
        # at the very top of the recall — recall_started_at_scope sets
        # the t_created ceiling that flows into store WHERE clauses;
        # trace.recall_scope tags every event below with a unique
        # recall_id + monotonic seq so the audit dashboard can
        # reconstruct the per-recall event tree from the JSONL log
        # even when concurrent recalls interleave. ContextVar
        # propagation makes both work for sync callers (here) and
        # async callers (recall_async below).
        from attestor import trace as _tr
        from attestor.recall_context import recall_started_at_scope

        with recall_started_at_scope(), _tr.recall_scope():
            t_total = time.monotonic()
            question_entities = self._question_entities(query)
            if _tr.is_enabled():
                _tr.event("recall.start", query=query[:200], namespace=namespace,
                          token_budget=token_budget,
                          question_entities=question_entities,
                          as_of=str(as_of) if as_of else None,
                          time_window=str(time_window) if time_window else None)

            # ── Step 0: Temporal pre-filter (RC4) ──
            time_window = self._step0_temporal_prefilter(query, time_window)

            # ── Step 1: Vector top-K (single | hyde | multi_query) ──
            vector_hits_raw, mq_used, path = self._compute_lane_vector(
                query, namespace, as_of, time_window,
            )

            # ── Step 1b: BM25 lane ──
            bm25_hits_raw = self._compute_lane_bm25(query, as_of, time_window)

            # ── Steps 2–6 — extracted into ``_post_process_candidates`` so
            # the same logic is reused by both the sync ``recall()`` and
            # the async ``recall_async()`` (Phase 6 of the async retrieval
            # rollout, see docs/plans/async-retrieval/PLAN.md).
            return self._post_process_candidates(
                query=query, namespace=namespace, as_of=as_of,
                time_window=time_window,
                question_entities=question_entities,
                vector_hits_raw=vector_hits_raw,
                bm25_hits_raw=bm25_hits_raw,
                mq_used=mq_used, path=path,
                token_budget=token_budget, t_total=t_total,
            )

    # ──────────────────────────────────────────────────────────────────
    # Phase 6 — async sibling. Vector lane and BM25 lane run in parallel
    # via asyncio.gather + asyncio.to_thread. Steps 2-6 (CPU-bound) run
    # serially after both lanes return; no parallelism opportunity there.
    #
    # Audit-preservation argument:
    #   - Opens a recall_started_at_scope_async() so the Postgres ceiling
    #     (audit invariant A2) is active for the entire recall.
    #   - The recall_id contextvar from trace.recall_scope, if opened by
    #     a caller, propagates through asyncio.to_thread (contextvars
    #     are copied onto threads via ContextVar.run() semantics).
    #   - Lane failures are isolated via gather(return_exceptions=True);
    #     a failing lane degrades to an empty list rather than aborting
    #     the recall.
    # ──────────────────────────────────────────────────────────────────

    async def recall_async(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: str | None = None,
        as_of: Any | None = None,
        time_window: Any | None = None,
    ) -> list[RetrievalResult]:
        """Async sibling of ``recall()``. Runs the vector lane and BM25
        lane concurrently via ``asyncio.gather`` + ``asyncio.to_thread``.

        Latency story (with vector ~80ms and BM25 ~50ms):
            sync:  ~130ms (vector → BM25 sequential)
            async:  ~80ms (vector ‖ BM25 under gather, max wins)
        """
        from attestor.recall_context import recall_started_at_scope_async

        from attestor import trace as _tr
        # Both audit scopes opened together — A2 ceiling + A5 trace tree.
        async with recall_started_at_scope_async():
            with _tr.recall_scope():
                t_total = time.monotonic()
                question_entities = self._question_entities(query)
                if _tr.is_enabled():
                    _tr.event(
                        "recall.start", query=query[:200], namespace=namespace,
                        token_budget=token_budget,
                        question_entities=question_entities,
                        as_of=str(as_of) if as_of else None,
                        time_window=str(time_window) if time_window else None,
                        mode="async",
                    )

                # Step 0 (sync, regex — cheap).
                time_window = self._step0_temporal_prefilter(query, time_window)

                # Steps 1 + 1b — the P6 win: parallel via gather.
                lane_results = await asyncio.gather(
                    asyncio.to_thread(
                        self._compute_lane_vector,
                        query, namespace, as_of, time_window,
                    ),
                    asyncio.to_thread(
                        self._compute_lane_bm25,
                        query, as_of, time_window,
                    ),
                    return_exceptions=True,
                )

                if isinstance(lane_results[0], Exception):
                    vector_hits_raw, mq_used, path = [], None, "single"
                else:
                    vector_hits_raw, mq_used, path = lane_results[0]

                if isinstance(lane_results[1], Exception):
                    bm25_hits_raw = []
                else:
                    bm25_hits_raw = lane_results[1]

                # Steps 2-6 — sync post-processing (no I/O parallelism inside).
                return self._post_process_candidates(
                    query=query, namespace=namespace, as_of=as_of,
                    time_window=time_window,
                    question_entities=question_entities,
                    vector_hits_raw=vector_hits_raw,
                    bm25_hits_raw=bm25_hits_raw,
                    mq_used=mq_used, path=path,
                    token_budget=token_budget, t_total=t_total,
                )

    def recall_as_context(
        self,
        query: str,
        token_budget: int = 2000,
        namespace: str | None = None,
    ) -> str:
        results = self.recall(query, token_budget, namespace=namespace)
        if not results:
            return ""
        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return "\n".join(lines)
