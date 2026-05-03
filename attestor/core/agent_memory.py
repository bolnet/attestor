"""AgentMemory — public API for the memory layer.

Composes Identity / Quota / Provenance mixins (split from a 1563-line
``core.py`` for FAANG-grade modularity). Public method signatures are
unchanged — this is a pure file-organization refactor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attestor.mode import AttestorMode, detect_mode
from attestor.models import Memory, RetrievalResult
from attestor.retrieval.orchestrator import RetrievalOrchestrator
from attestor.store.base import DocumentStore, GraphStore, VectorStore
from attestor.temporal.manager import TemporalManager
from attestor.utils.config import MemoryConfig, load_config, save_config

from attestor.core.identity_service import _IdentityMixin
from attestor.core.provenance_service import _ProvenanceMixin
from attestor.core.quota_service import _QuotaMixin


def _registry():
    """Late lookup of registry symbols through the ``attestor.core`` package.

    Tests patch ``attestor.core.instantiate_backend`` / ``resolve_backends``
    / ``DEFAULT_BACKENDS`` (see tests/test_embedder_dim_check.py). Routing
    every reference through the package namespace at call time ensures
    those patches are observed without losing the symbol re-export.
    """
    import attestor.core as _pkg
    return _pkg

logger = logging.getLogger("attestor")


class AgentMemory(_IdentityMixin, _QuotaMixin, _ProvenanceMixin):
    """Memory for AI agents, backed by Postgres (doc+pgvector) + Neo4j (graph).

    Usage:
        mem = AgentMemory("./my-agent", config={"backend_configs": {...}})
        mem.add("User prefers Python", tags=["preference"], category="preference")
        results = mem.recall("what language?")
    """

    def __init__(
        self,
        path: str | Path,
        config: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Load config
        if config:
            self.config = MemoryConfig.from_dict(config)
        else:
            self.config = load_config(self.path)
        save_config(self.path, self.config)

        # Backend configuration
        _reg = _registry()
        backends = getattr(self.config, "backends", None) or _reg.DEFAULT_BACKENDS
        backend_configs: dict[str, dict[str, Any]] = (
            getattr(self.config, "backend_configs", None) or {}
        )

        # Resolve role -> backend_name mapping
        role_assignments = _reg.resolve_backends(backends)

        # Docker manager (lazy, only needed for container-based backends)
        self._docker = None

        # Track instantiated backends so the same backend isn't constructed
        # twice across roles (defensive — the canonical stack assigns one
        # backend per role, so this only matters if a future config maps the
        # same backend to multiple roles).
        _instances: dict[str, Any] = {}

        def _get_or_create(backend_name: str) -> Any:
            if backend_name in _instances:
                return _instances[backend_name]
            bcfg = backend_configs.get(backend_name, {})
            instance = _reg.instantiate_backend(backend_name, self.path, bcfg)
            _instances[backend_name] = instance
            return instance

        # Initialize document store (required — no graceful degradation)
        doc_backend = role_assignments["document"]
        self._store: DocumentStore = _get_or_create(doc_backend)

        # Embedder/schema dim guard — fail-fast if the embedder produces
        # D-dim vectors but the pgvector schema declares vector(N) with
        # N != D. Without this guard, every UPDATE on memories.embedding
        # silently no-ops (the doc-write path swallows non-fatal vector
        # errors) and Attestor accepts writes that store nothing in the
        # vector lane. Skips on non-Postgres stores, on first-init
        # (table not created yet), and on introspection errors.
        from attestor.store.embedder_dim_check import (
            assert_embedder_dim_matches_schema,
        )
        assert_embedder_dim_matches_schema(self._store)

        # Initialize vector store (optional — graceful degradation)
        self._vector_store: VectorStore | None = None
        if "vector" in role_assignments:
            try:
                self._vector_store = _get_or_create(role_assignments["vector"])
            except Exception as e:
                logger.warning("Vector store init failed (%s): %s",
                               role_assignments["vector"], e)

        # Initialize graph store (optional — graceful degradation)
        self._graph: GraphStore | None = None
        if "graph" in role_assignments:
            try:
                self._graph = _get_or_create(role_assignments["graph"])
            except Exception as e:
                logger.warning("Graph store init failed (%s): %s",
                               role_assignments["graph"], e)

        # Initialize managers
        self._temporal = TemporalManager(self._store)
        # BM25 / FTS lane (Phase 4.3) — opt-in via v4 schema's content_tsv
        # column. Only safe when the doc store exposes a psycopg2 conn.
        bm25_lane = None
        if (
            getattr(self._store, "_v4", False)
            and getattr(self._store, "_conn", None) is not None
        ):
            try:
                from attestor.retrieval.bm25 import BM25Lane
                bm25_lane = BM25Lane(self._store._conn)
            except Exception as e:
                logger.debug("BM25 lane init skipped: %s", e)

        # Build the orchestrator's tuning config from YAML.
        # ``RetrievalRuntimeConfig`` is the orchestrator-side dataclass;
        # ``stack.retrieval`` (a ``RetrievalCfg``) is the YAML-loaded
        # superset. The runtime config picks the score-blending +
        # vector_top_k + mmr_lambda subset that the recall hot path
        # actually reads. We deliberately don't make AgentMemory crash
        # if the stack loader fails — this is a tunable, not a
        # correctness constraint, and the runtime config's own defaults
        # match the historical literals.
        from attestor.retrieval.orchestrator import RetrievalRuntimeConfig
        _retrieval_cfg = None
        _runtime_cfg: RetrievalRuntimeConfig | None = None
        try:
            from attestor.config import get_stack
            _stack = get_stack(strict=False)
            _retrieval_cfg = _stack.retrieval
            _runtime_cfg = RetrievalRuntimeConfig.from_stack(_stack)
        except Exception as _e:
            logger.debug("retrieval cfg not applied: %s", _e)

        self._retrieval = RetrievalOrchestrator(
            self._store,
            min_results=self.config.min_results,
            vector_store=self._vector_store,
            graph=self._graph,
            bm25_lane=bm25_lane,
            config=_runtime_cfg,
        )
        # Wire MemoryConfig overrides onto the orchestrator. These are
        # caller-controlled knobs that override the YAML values when
        # set explicitly on MemoryConfig. The orchestrator already
        # picked YAML defaults via the RetrievalRuntimeConfig above;
        # the writes below are last-mile overrides for back-compat
        # with the embedded-mode MemoryConfig surface.
        self._retrieval.enable_mmr = self.config.enable_mmr
        self._retrieval.mmr_lambda = self.config.mmr_lambda
        self._retrieval.fusion_mode = self.config.fusion_mode
        self._retrieval.confidence_gate = self.config.confidence_gate
        self._retrieval.confidence_decay_rate = self.config.confidence_decay_rate
        self._retrieval.confidence_boost_rate = self.config.confidence_boost_rate

        # Lane-level cfgs (multi_query / temporal_prefilter / hyde)
        # are still attached as separate attributes — they configure
        # independent lanes, not the score-blending hot path. Skip if
        # the stack loader failed above.
        if _retrieval_cfg is not None:
            self._retrieval.mmr_top_n = _retrieval_cfg.mmr_top_n
            self._retrieval.multi_query_cfg = _retrieval_cfg.multi_query
            self._retrieval.temporal_prefilter_cfg = (
                _retrieval_cfg.temporal_prefilter
            )
            self._retrieval.hyde_cfg = _retrieval_cfg.hyde

        # Operation ring buffer for latency observability
        self._ops_log: deque[dict[str, Any]] = deque(maxlen=200)

        # v4 provenance signing (Phase 8.1) — opt-in via config["signing"].
        # When enabled, every memory written through self.add() gets an
        # Ed25519 signature stored in memories.signature. Verification is
        # callable via mem.verify_memory(memory_id).
        self._signer = None
        signing_cfg = (
            config.get("signing") if config else None
        ) or getattr(self.config, "signing", None)
        if signing_cfg:
            try:
                from attestor.identity.signing import Signer
                self._signer = Signer.from_config(signing_cfg)
            except Exception as e:
                logger.warning("provenance signing init failed: %s", e)

        # v4 quota enforcement (Phase 8.3) — auto-enabled on v4 + Postgres
        # since the user_quotas table is part of the v4 schema. NULL limits
        # = unlimited, so this is a no-op until SetLimits is called.
        self._quotas = None
        if (
            getattr(self._store, "_v4", False)
            and getattr(self._store, "_conn", None) is not None
        ):
            try:
                from attestor.quotas import QuotaRepo
                self._quotas = QuotaRepo(self._store._conn)
            except Exception as e:
                logger.debug("QuotaRepo init skipped: %s", e)

        # v4 operating mode (SOLO / HOSTED / SHARED). Detected from env on
        # first construction; tests can override via config["mode"]. The
        # mode controls how _resolve() fills missing identity params.
        self._mode = (
            AttestorMode(config.get("mode")) if config and config.get("mode")
            else detect_mode()
        )
        # In SOLO mode, ensure the singleton user + Inbox exist so the
        # zero-config "AgentMemory().add('foo')" path works without the
        # caller doing any identity setup. Only viable for v4 + Postgres
        # (the identity repos need _conn). For other modes / backends, the
        # caller is expected to provide IDs explicitly.
        self._default_user = None
        if (
            self._mode is AttestorMode.SOLO
            and getattr(self._store, "_v4", False)
            and getattr(self._store, "_conn", None) is not None
        ):
            try:
                from attestor.identity.defaults import ensure_solo_user
                self._default_user = ensure_solo_user(self._user_repo())
                # Pre-create Inbox so the first add() doesn't pay for it.
                self._project_repo().ensure_inbox(self._default_user.id)
                # Set RLS to the default user so subsequent reads admit rows.
                if hasattr(self._store, "_set_rls_user"):
                    self._store._set_rls_user(self._default_user.id)
            except Exception as e:
                # Non-fatal: AgentMemory still works in v3 / explicit-id
                # mode even if the SOLO bootstrap fails (e.g. tables not
                # provisioned yet). Log and move on.
                logger.warning("SOLO default-user bootstrap skipped: %s", e)

    def close(self) -> None:
        """Close all database connections."""
        if self._vector_store:
            try:
                self._vector_store.close()
            except Exception:
                pass
        if self._graph:
            try:
                self._graph.save()
            except Exception:
                pass
            if hasattr(self._graph, "close"):
                try:
                    self._graph.close()
                except Exception:
                    pass
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- v4 round-level conversation ingest (Phase 3.5) --

    def ingest_round(
        self,
        user_turn: Any,                 # ConversationTurn
        assistant_turn: Any,            # ConversationTurn
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        scope: str = "user",
        agent_id: str | None = None,
        recent_context: str = "(none)",
        config: Any | None = None,        # IngestConfig
        extraction_client: Any | None = None,
        resolver_client: Any | None = None,
    ) -> Any:                            # RoundResult
        """End-to-end ingest of one conversational round.

        Resolves identity (SOLO defaults apply), writes the verbatim
        episode, runs speaker-locked extraction in two passes, resolves
        conflicts against existing similar memories, and applies the
        ADD/UPDATE/INVALIDATE/NOOP decisions through the supersession
        path.

        Returns a ``RoundResult`` with episode + decisions + applied
        outcomes — enough for tests + audit dashboards.
        """
        from attestor.conversation.ingest import ConversationIngest

        rc = self._resolve(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            autostart=True,
        )
        ingest = ConversationIngest(
            self,
            config=config,
            extraction_client=extraction_client,
            resolver_client=resolver_client,
        )
        return ingest.ingest_round(
            user_turn=user_turn,
            assistant_turn=assistant_turn,
            user_id=rc.user.id,
            project_id=rc.project.id,
            session_id=rc.session.id if rc.session else None,
            scope=scope,
            agent_id=agent_id,
            recent_context=recent_context,
        )

    # -- v4 sleep-time consolidation (Phase 7) --

    def consolidate(
        self,
        *,
        limit: int = 20,
        model: str | None = None,
        extraction_client: Any | None = None,
        resolver_client: Any | None = None,
    ) -> list[Any]:
        """Drain one batch from the consolidation queue in-process.

        For long-running daemons use ``SleepTimeConsolidator.run_forever``
        directly. This method is for tests and one-shot manual triggers
        (e.g., ``attestor consolidate run``).

        Returns a list of ``ConsolidationResult`` (one per processed
        episode). Empty list when the queue is drained.
        """
        self._require_v4()
        from attestor.consolidation import SleepTimeConsolidator
        kwargs: dict[str, Any] = {"batch_size": limit}
        if model:
            kwargs["model"] = model
        if extraction_client is not None:
            kwargs["extraction_client"] = extraction_client
        if resolver_client is not None:
            kwargs["resolver_client"] = resolver_client
        cons = SleepTimeConsolidator(self, **kwargs)
        return cons.run_once(limit=limit)

    # -- v4 GDPR delete + export (Phase 8.5) --

    def export_user(self, external_id: str) -> dict[str, Any]:
        """Full data portability dump for a user. JSON-serializable."""
        self._require_v4()
        from attestor.gdpr import export_user
        return export_user(self._store._conn, external_id).to_dict()

    def purge_user(
        self,
        external_id: str,
        *,
        reason: str = "gdpr_request",
        deleted_by: str | None = None,
    ) -> dict[str, Any]:
        """Hard-delete a user. CASCADEs through Postgres; vector/graph
        stores are the caller's responsibility (they don't CASCADE)."""
        self._require_v4()
        from attestor.gdpr import purge_user
        result = purge_user(
            self._store._conn, external_id,
            reason=reason, deleted_by=deleted_by,
        )
        return {
            "user_existed": result.user_existed,
            "audit_id": result.audit_id,
            "counts": result.counts,
        }

    def deletion_audit_log(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Recent GDPR deletion audit entries (read-only)."""
        self._require_v4()
        from attestor.gdpr import list_audit_log
        return list_audit_log(self._store._conn, limit=limit)

    # -- Write --

    @staticmethod
    def _content_hash(content: str) -> str:
        """Compute SHA-256 hash of normalized content for dedup."""
        return hashlib.sha256(content.strip().encode()).hexdigest()

    def add(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str = "general",
        entity: str | None = None,
        namespace: str = "default",
        event_date: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        # ── v4 tenancy params (Phase 1 chunk 3) ──
        # When user_id is provided AND the backend is in v4 mode, the memory
        # is written through the v4 path (RLS-scoped, bi-temporal). When
        # user_id is None, the legacy v3 path runs unchanged.
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        scope: str = "user",
        agent_id: str | None = None,
        source_episode_id: str | None = None,
    ) -> Memory:
        """Store a new memory, handling contradictions automatically.

        v4 callers should pass ``user_id`` (and optionally project_id,
        session_id, scope). v3 callers can omit them — behavior unchanged."""
        t_total = time.monotonic()
        store_timings: dict[str, float] = {}

        # v4: route through the resolution chain so zero-config calls work.
        # In v3 / non-Postgres mode this branch is skipped entirely.
        v4_active = getattr(self._store, "_v4", False)
        if v4_active:
            rc = self._resolve(
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                autostart=True,
            )
            user_id = rc.user.id
            project_id = rc.project.id
            session_id = rc.session.id if rc.session else None
            # Quota check BEFORE the insert so we don't half-write
            if self._quotas is not None:
                self._quotas.check_memory_quota(user_id)

        # Dedup: check for exact content match (scoped by namespace)
        chash = self._content_hash(content)
        if hasattr(self._store, "get_by_hash"):
            existing = self._store.get_by_hash(chash, namespace=namespace)
            if existing:
                logger.debug("Dedup hit: content_hash=%s -> id=%s", chash[:8], existing.id)
                return existing

        # In v4 the schema has no `namespace` column (tenancy moved to
        # user_id + project_id + scope). Stamp the namespace into metadata
        # so it survives the round-trip — Memory.from_row() reads it back
        # from metadata["_namespace"] when there's no top-level column.
        # Without this, any caller that uses namespace as a sub-tenancy key
        # (LME bench writes namespace="lme_<sample>"; ad-hoc multi-tenant
        # callers passing namespace=...) silently drops 100% of recall
        # candidates because every read returns namespace="default".
        _final_metadata = dict(metadata or {})
        if v4_active and namespace and namespace != "default":
            _final_metadata.setdefault("_namespace", namespace)

        # Couple ``valid_from`` to ``event_date`` when the caller provided
        # one. Without this the dataclass default factory stamps NOW(), and
        # the temporal manager loses the real event time — supersession
        # tiebreakers, BM25 time-window filters, and as_of replay all see
        # ingest time instead. The v4 schema also drops ``event_date``
        # entirely, so ``valid_from`` is the only column carrying the date
        # in v4 mode. See tests/test_temporal_supersession_gaps.py.
        memory_kwargs: dict[str, Any] = dict(
            content=content,
            tags=tags or [],
            category=category,
            entity=entity,
            namespace=namespace,
            event_date=event_date,
            confidence=confidence,
            content_hash=chash,
            metadata=_final_metadata,
            # v4 fields (None / defaults when caller didn't pass them)
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            scope=scope,
            agent_id=agent_id,
            source_episode_id=source_episode_id,
        )
        if event_date:
            memory_kwargs["valid_from"] = event_date
        memory = Memory(**memory_kwargs)

        # Check for contradictions before insert
        contradictions = self._temporal.check_contradictions(memory)

        # Insert new memory first (so FK reference is valid).
        # In v4 mode the document backend assigns a fresh UUID and returns
        # a new (frozen) Memory; we MUST capture that return so the local
        # `memory` variable carries the DB-generated id forward to vector
        # add, graph extract, signing UPDATE, and the caller.
        t0 = time.monotonic()
        memory = self._store.insert(memory) or memory
        store_timings["document_ms"] = round((time.monotonic() - t0) * 1000, 2)
        from attestor import trace as _tr
        if _tr.is_enabled():
            _tr.event("ingest.write.pg",
                      memory_id=memory.id, namespace=namespace,
                      content_len=len(content), tags=list(tags or []),
                      category=category, entity=entity,
                      latency_ms=store_timings["document_ms"])

        # v4 provenance signing (Phase 8.1) — sign AFTER insert so the
        # canonical payload includes the DB-generated id + t_created.
        if self._signer is not None and v4_active:
            try:
                sig = self._signer.sign(memory)
                memory = replace(memory, signature=sig)
                with self._store._conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memories SET signature = %s WHERE id = %s",
                        (sig, memory.id),
                    )
            except Exception as e:
                logger.warning("provenance sign failed for %s: %s", memory.id, e)

        # Then supersede old contradicting memories
        for old in contradictions:
            self._temporal.supersede(old, memory.id)

        # Store in vector DB. Non-fatal — the document path is the source
        # of truth and recall degrades gracefully without vectors. Surface
        # the exception via logger.warning so the failure is debuggable
        # (silent pass made dim-mismatch / embedder-down / schema-drift bugs
        # invisible until recall returned 0 hits).
        if self._vector_store:
            try:
                t0 = time.monotonic()
                self._vector_store.add(memory.id, content, namespace=namespace)
                store_timings["vector_ms"] = round((time.monotonic() - t0) * 1000, 2)
                if _tr.is_enabled():
                    _tr.event("ingest.write.vector",
                              memory_id=memory.id, namespace=namespace,
                              latency_ms=store_timings["vector_ms"], ok=True)
            except Exception as e:
                store_timings["vector_ms"] = -1  # failed
                logger.warning(
                    "vector add failed for memory %s: %s: %s",
                    memory.id, type(e).__name__, e,
                )
                if _tr.is_enabled():
                    _tr.event("ingest.write.vector",
                              memory_id=memory.id, namespace=namespace,
                              ok=False, error=f"{type(e).__name__}: {e}")

        # Update entity graph (also non-fatal; surface exceptions for the
        # same reason — silent drops here mean recall layer 2 returns
        # nothing without the operator knowing).
        if self._graph:
            try:
                from attestor.graph.extractor import extract_entities_and_relations
                t0 = time.monotonic()
                nodes, edges = extract_entities_and_relations(
                    content, tags or [], entity, category,
                    namespace=namespace,
                )
                if _tr.is_enabled():
                    _tr.event("ingest.extract",
                              memory_id=memory.id, namespace=namespace,
                              entity_count=len(nodes), relation_count=len(edges),
                              entity_names=[n["name"] for n in nodes][:10])
                # Tag every entity / relation with the writer's namespace
                # so the graph layer enforces tenancy alongside Postgres.
                # Older graph backends without the kwarg still accept the
                # call via the TypeError fallback below.
                for node in nodes:
                    try:
                        self._graph.add_entity(
                            node["name"],
                            entity_type=node.get("type", "general"),
                            attributes=node.get("attributes"),
                            namespace=namespace,
                        )
                    except TypeError:
                        self._graph.add_entity(
                            node["name"],
                            entity_type=node.get("type", "general"),
                            attributes=node.get("attributes"),
                        )
                for edge in edges:
                    try:
                        self._graph.add_relation(
                            edge["from"],
                            edge["to"],
                            relation_type=edge.get("type", "related_to"),
                            metadata=edge.get("metadata"),
                            namespace=namespace,
                        )
                    except TypeError:
                        self._graph.add_relation(
                            edge["from"],
                            edge["to"],
                            relation_type=edge.get("type", "related_to"),
                            metadata=edge.get("metadata"),
                        )
                store_timings["graph_ms"] = round((time.monotonic() - t0) * 1000, 2)
                if _tr.is_enabled():
                    _tr.event("ingest.write.graph",
                              memory_id=memory.id, namespace=namespace,
                              entity_count=len(nodes), relation_count=len(edges),
                              latency_ms=store_timings["graph_ms"], ok=True)
            except Exception as e:
                store_timings["graph_ms"] = -1  # failed
                logger.warning(
                    "graph add failed for memory %s: %s: %s",
                    memory.id, type(e).__name__, e,
                )
                if _tr.is_enabled():
                    _tr.event("ingest.write.graph",
                              memory_id=memory.id, namespace=namespace,
                              ok=False, error=f"{type(e).__name__}: {e}")

        total_ms = round((time.monotonic() - t_total) * 1000, 2)
        store_timings["total_ms"] = total_ms

        # Record in ops log
        self._ops_log.append({
            "op": "add",
            "ts": datetime.now(timezone.utc).isoformat(),
            "latency_ms": total_ms,
            "store_timings": store_timings,
            "input": content[:120],
            "result_count": 1,
            "stores": [k.replace("_ms", "") for k, v in store_timings.items()
                       if k != "total_ms" and v >= 0],
        })

        return memory

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
        entity: str | None = None,
    ) -> Memory | None:
        """Update an existing memory's fields. Returns updated memory or None."""
        memory = self._store.get(memory_id)
        if not memory:
            return None

        updates: dict[str, Any] = {}
        if content is not None:
            updates["content"] = content
            updates["content_hash"] = self._content_hash(content)
        if tags is not None:
            updates["tags"] = tags
        if category is not None:
            updates["category"] = category
        if entity is not None:
            updates["entity"] = entity
        if updates:
            memory = replace(memory, **updates)

        self._store.update(memory)

        # Re-index in vector store if content changed. Non-fatal; surface
        # exceptions through logger.warning (same reasoning as add()).
        if content is not None and self._vector_store:
            try:
                self._vector_store.add(
                    memory.id, content, namespace=memory.namespace,
                )
            except Exception as e:
                logger.warning(
                    "vector re-index failed for memory %s on update: %s: %s",
                    memory.id, type(e).__name__, e,
                )

        return memory

    # -- Read --

    def recall(
        self,
        query: str,
        budget: int | None = None,
        namespace: str | None = None,
        user_id: str | None = None,
        as_of: datetime | None = None,
        time_window: Any | None = None,    # TimeWindow
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories for a query using 3-layer cascade.

        v4: when ``user_id`` is provided AND the backend is in v4 mode, the
        RLS variable is set on the connection so policies filter to this
        user. v3 callers omit user_id — behavior unchanged.

        v4 + Phase 5.3 — bi-temporal:
          as_of       — point-in-time replay (returns past belief)
          time_window — event-time overlap pre-filter
        Both pass through to the orchestrator and on to the lanes."""
        # v4: route through _resolve() so zero-config recall works in SOLO
        # mode. Recall doesn't autostart a session — read-only ops only need
        # user+project scope.
        if getattr(self._store, "_v4", False):
            rc = self._resolve(
                user_id=user_id,
                project_id=None,
                session_id=None,
                autostart=False,
            )
            user_id = rc.user.id

        t0 = time.monotonic()
        token_budget = budget or self.config.default_token_budget
        # Pass temporal kwargs only when present so legacy v3 orchestrator
        # signatures aren't disturbed.
        recall_kwargs: dict[str, Any] = {"namespace": namespace}
        if as_of is not None:
            recall_kwargs["as_of"] = as_of
        if time_window is not None:
            recall_kwargs["time_window"] = time_window
        results = self._retrieval.recall(query, token_budget, **recall_kwargs)
        total_ms = round((time.monotonic() - t0) * 1000, 2)

        # Track access for confidence decay/boost
        real_ids = [
            r.memory.id
            for r in results
            if r.memory.category != "graph_relation"
        ]
        if real_ids and hasattr(self._store, "increment_access"):
            try:
                self._store.increment_access(real_ids)
            except Exception:
                pass  # Non-fatal

        # Record in ops log
        stores = ["document"]
        if self._vector_store:
            stores.append("vector")
        if self._graph:
            stores.append("graph")
        self._ops_log.append({
            "op": "recall",
            "ts": datetime.now(timezone.utc).isoformat(),
            "latency_ms": total_ms,
            "input": query[:120],
            "result_count": len(results),
            "budget": token_budget,
            "stores": stores,
        })

        return results

    def recall_as_context(
        self,
        query: str,
        budget: int | None = None,
        namespace: str | None = None,
    ) -> str:
        """Recall and format as a context string for prompt injection.

        Legacy v3 surface — returns plain text. New code should prefer
        ``recall_as_pack`` which returns a structured ``ContextPack``
        with citations + Chain-of-Note prompt (Phase 6).
        """
        token_budget = budget or self.config.default_token_budget
        return self._retrieval.recall_as_context(
            query, token_budget, namespace=namespace
        )

    def recall_as_pack(
        self,
        query: str,
        budget: int | None = None,
        user_id: str | None = None,
        as_of: datetime | None = None,
        time_window: Any | None = None,
        chain_of_note_prompt: str | None = None,
    ):
        """Recall as a structured ``ContextPack`` for Chain-of-Note agents.

        The returned pack carries:
          - memories sorted by score (orchestrator decides ranking)
          - per-memory: id, content, validity window, confidence,
            source_episode_id (for citation)
          - default Chain-of-Note prompt with ABSTAIN clause

        v4 + Phase 5 callers can pass ``as_of`` / ``time_window`` for
        bi-temporal replay; the pack reflects the snapshot accordingly.

        Phase 6.2 — roadmap §D.1.
        """
        from attestor.models import ContextPack, ContextPackEntry
        from attestor.prompts.chain_of_note import DEFAULT_CHAIN_OF_NOTE_PROMPT

        results = self.recall(
            query, budget=budget, user_id=user_id,
            as_of=as_of, time_window=time_window,
        )
        # Skip synthetic graph_relation rows — they aren't real memories
        # and have no citation target.
        real = [r for r in results if r.memory.category != "graph_relation"]
        entries = [
            ContextPackEntry(
                id=r.memory.id,
                content=r.memory.content,
                category=r.memory.category,
                entity=r.memory.entity,
                valid_from=r.memory.valid_from,
                valid_until=r.memory.valid_until,
                confidence=r.memory.confidence,
                source_episode_id=r.memory.source_episode_id,
                score=r.score,
            )
            for r in real
        ]
        # Rough token-count estimate: ~4 chars/token
        chars = sum(len(e.content) for e in entries)
        token_count = chars // 4

        return ContextPack(
            query=query,
            memories=entries,
            as_of=as_of.isoformat() if as_of is not None else None,
            token_count=token_count,
            chain_of_note_prompt=(
                chain_of_note_prompt or DEFAULT_CHAIN_OF_NOTE_PROMPT
            ),
        )

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
        status: str = "active",
        after: str | None = None,
        before: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories with filters."""
        # If there's a text query and vector store, use semantic search
        if query and self._vector_store:
            try:
                vec_results = self._vector_store.search(
                    query, limit=limit * 2, namespace=namespace
                )
                if vec_results:
                    # Get full memory objects, apply filters
                    memories = []
                    for vr in vec_results:
                        mem = self._store.get(vr["memory_id"])
                        if not mem or mem.status != status:
                            continue
                        if category and mem.category != category:
                            continue
                        if entity and mem.entity != entity:
                            continue
                        if namespace and mem.namespace != namespace:
                            continue
                        if after and mem.created_at < after:
                            continue
                        if before and mem.created_at > before:
                            continue
                        memories.append(mem)
                        if len(memories) >= limit:
                            break
                    return memories
            except Exception:
                pass  # Fall through to document store search

        return self._store.list_memories(
            status=status,
            category=category,
            entity=entity,
            namespace=namespace,
            after=after,
            before=before,
            limit=limit,
        )

    # -- Timeline --

    def timeline(
        self, entity: str, namespace: str | None = None
    ) -> list[Memory]:
        """Get all memories about an entity in chronological order."""
        return self._temporal.timeline(entity, namespace=namespace)

    def current_facts(
        self,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
    ) -> list[Memory]:
        """Get only active, non-superseded memories."""
        return self._temporal.current_facts(
            category=category, entity=entity, namespace=namespace
        )

    # -- Extraction --

    def extract(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        use_llm: bool = False,
        namespace: str = "default",
    ) -> list[Memory]:
        """Extract and store memories from conversation messages.

        ``model`` defaults to ``stack.models.extraction`` from
        ``configs/attestor.yaml``.
        """
        from attestor.extraction.extractor import extract_memories

        # extract_memories already resolves None → stack default; pass
        # through unchanged.
        extracted = extract_memories(messages, use_llm=use_llm, model=model)
        stored = []
        for mem in extracted:
            stored_mem = self.add(
                content=mem.content,
                tags=mem.tags,
                category=mem.category,
                entity=mem.entity,
                namespace=namespace,
            )
            stored.append(stored_mem)
        return stored

    # -- Batch Operations --

    def batch_embed(self, batch_size: int = 100) -> int:
        """Batch-index all active memories into the vector store.

        Returns count of memories processed.
        """
        if not self._vector_store:
            return 0
        memories = self._store.list_memories(status="active", limit=1_000_000)
        count = 0
        for mem in memories:
            try:
                self._vector_store.add(mem.id, mem.content)
                count += 1
            except Exception:
                pass
        return count

    # -- Maintenance --

    def forget(self, memory_id: str) -> bool:
        """Archive a specific memory."""
        memory = self._store.get(memory_id)
        if memory:
            memory = replace(memory, status="archived")
            self._store.update(memory)
            return True
        return False

    def forget_before(self, date: str) -> int:
        """Archive memories created before a date."""
        return self._store.archive_before(date)

    def compact(self) -> int:
        """Permanently remove archived memories."""
        return self._store.compact()

    @property
    def ops_log(self) -> list[dict[str, Any]]:
        """Return a snapshot of the operation ring buffer (most recent last)."""
        return list(self._ops_log)

    def stats(self) -> dict[str, Any]:
        """Get store statistics."""
        return self._store.stats()

    def _try_recover_stores(self) -> dict[str, str]:
        """Attempt to re-initialize failed vector/graph stores.

        Called by health() when stores are None but config says they should
        exist. On success, wires the recovered store into the retrieval
        pipeline so the running process heals without a restart.

        Returns:
            Dict of role -> outcome ("recovered" | "failed: <reason>")
        """
        _reg = _registry()
        backends = getattr(self.config, "backends", None) or _reg.DEFAULT_BACKENDS
        backend_configs: dict[str, dict[str, Any]] = (
            getattr(self.config, "backend_configs", None) or {}
        )
        role_assignments = _reg.resolve_backends(backends)
        results: dict[str, str] = {}

        # Try to recover vector store
        if self._vector_store is None and "vector" in role_assignments:
            backend_name = role_assignments["vector"]
            try:
                self._vector_store = _reg.instantiate_backend(
                    backend_name, self.path, backend_configs.get(backend_name),
                )
                self._retrieval.vector_store = self._vector_store
                logger.info("Recovered vector store (%s)", backend_name)
                results["vector"] = "recovered"
            except Exception as e:
                logger.warning("Vector store recovery failed (%s): %s", backend_name, e)
                results["vector"] = f"failed: {e}"

        # Try to recover graph store
        if self._graph is None and "graph" in role_assignments:
            backend_name = role_assignments["graph"]
            try:
                self._graph = _reg.instantiate_backend(
                    backend_name, self.path, backend_configs.get(backend_name),
                )
                self._retrieval.graph = self._graph
                logger.info("Recovered graph store (%s)", backend_name)
                results["graph"] = "recovered"
            except Exception as e:
                logger.warning("Graph store recovery failed (%s): %s", backend_name, e)
                results["graph"] = f"failed: {e}"

        return results

    def health(self) -> dict[str, Any]:
        """Check health of all components. Returns structured status report.

        If vector or graph stores failed at startup, attempts recovery before
        reporting. This lets long-running processes (like the MCP server)
        self-heal without a restart.

        Checks: Document Store (Postgres), Vector Store (pgvector), Graph
        Store (Neo4j), Retrieval Pipeline.
        """
        t0_health = time.monotonic()

        # Attempt recovery of failed stores before reporting
        recovery: dict[str, str] = {}
        if self._vector_store is None or self._graph is None:
            recovery = self._try_recover_stores()

        report: dict[str, Any] = {
            "healthy": True,
            "checks": [],
        }

        def _check(name: str, status: str, **details):
            entry = {"name": name, "status": status, **details}
            report["checks"].append(entry)
            if status == "error":
                report["healthy"] = False

        # -- Document Store --
        try:
            t0 = time.monotonic()
            store_stats = self._store.stats()
            latency = round((time.monotonic() - t0) * 1000, 1)
            check_name = type(self._store).__name__
            details: dict[str, Any] = {
                "memory_count": store_stats.get("total_memories", 0),
                "latency_ms": latency,
            }
            _check(check_name, "ok", **details)
        except Exception as e:
            _check("Document Store", "error", error=str(e))

        # -- Vector Store --
        if self._vector_store:
            try:
                vec_name = type(self._vector_store).__name__
                # Skip if same instance as document store (multi-role backend)
                if self._vector_store is not self._store:
                    t0 = time.monotonic()
                    vector_count = self._vector_store.count()
                    vec_latency = round((time.monotonic() - t0) * 1000, 1)
                    vec_details: dict[str, Any] = {
                        "vector_count": vector_count,
                        "latency_ms": vec_latency,
                    }
                    if hasattr(self._vector_store, "provider"):
                        vec_details["embedding_provider"] = self._vector_store.provider
                    if "vector" in recovery:
                        vec_details["note"] = "recovered at health check"
                    _check(vec_name, "ok", **vec_details)
                else:
                    _check(f"{vec_name} (vector)", "ok",
                           note="shared instance with document store")
            except Exception as e:
                _check("Vector Store", "error", error=str(e))
        else:
            error_msg = "Not initialized"
            if "vector" in recovery:
                error_msg += f" (recovery {recovery['vector']})"
            _check("Vector Store", "error", error=error_msg)

        # -- Graph Store --
        if self._graph:
            try:
                graph_name = type(self._graph).__name__
                t0 = time.monotonic()
                graph_stats = (
                    self._graph.graph_stats()
                    if hasattr(self._graph, "graph_stats")
                    else self._graph.stats()
                )
                graph_latency = round((time.monotonic() - t0) * 1000, 1)
                graph_details: dict[str, Any] = {
                    "nodes": graph_stats["nodes"],
                    "edges": graph_stats["edges"],
                    "latency_ms": graph_latency,
                }
                # Skip if same instance as document store (multi-role backend)
                if self._graph is not self._store:
                    if hasattr(self._graph, "graph_path"):
                        graph_details["graph_file"] = str(self._graph.graph_path)
                    if "graph" in recovery:
                        graph_details["note"] = "recovered at health check"
                    _check(graph_name, "ok", **graph_details)
                else:
                    _check(f"{graph_name} (graph)", "ok", **graph_details,
                           note="shared instance with document store")
            except Exception as e:
                _check("Graph Store", "error", error=str(e))
        else:
            error_msg = "Not initialized"
            if "graph" in recovery:
                error_msg += f" (recovery {recovery['graph']})"
            _check("Graph Store", "error", error=error_msg)

        # -- Retrieval Pipeline --
        layers = ["tag_match"]
        if self._graph:
            layers.append("graph_expansion")
        if self._vector_store:
            layers.append("vector_similarity")
        _check("Retrieval Pipeline", "ok",
               active_layers=len(layers), max_layers=3, layers=layers)

        health_ms = round((time.monotonic() - t0_health) * 1000, 2)

        # Include ops log summary in health report
        report["ops_log_size"] = len(self._ops_log)
        if self._ops_log:
            latencies = [op["latency_ms"] for op in self._ops_log]
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            report["latency_percentiles"] = {
                "p50": latencies_sorted[n // 2] if n else 0,
                "p95": latencies_sorted[int(n * 0.95)] if n else 0,
                "p99": latencies_sorted[int(n * 0.99)] if n else 0,
            }
            report["latency_sparkline"] = [
                op["latency_ms"] for op in list(self._ops_log)[-50:]
            ]

        # Record health check itself in ops log
        self._ops_log.append({
            "op": "health",
            "ts": datetime.now(timezone.utc).isoformat(),
            "latency_ms": health_ms,
            "input": "health check",
            "result_count": len(report["checks"]),
            "stores": ["document"]
                + (["vector"] if self._vector_store else [])
                + (["graph"] if self._graph else []),
        })

        return report

    # -- Export / Import --

    def export_json(self, filepath: str) -> None:
        """Export all memories to a JSON file."""
        memories = self._store.list_memories(limit=1_000_000)
        data = [m.to_dict() for m in memories]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_json(self, filepath: str) -> int:
        """Import memories from a JSON file. Returns count imported.

        Each entry in the JSON payload is mapped onto a fresh ``Memory``
        and handed to ``self._store.insert``. Two safety details matter:

        - ``id`` is regenerated when the JSON entry omits the field OR
          provides ``null`` / empty-string. ``dict.get("id", DEFAULT)``
          only uses the default for *missing* keys, so a payload with
          ``"id": null`` would otherwise pass ``None`` straight to the
          store and silently drop the row when the PK rejects it.
        - Insert errors that are NOT a duplicate-content collision are
          logged at WARNING. The previous bare ``except Exception: pass``
          masked PK violations, NOT-NULL violations, and connection
          errors as if they were dedup hits.
        """
        with open(filepath) as f:
            data = json.load(f)
        count = 0
        for item in data:
            content = item["content"]
            chash = self._content_hash(content)

            # Skip if duplicate content already exists
            if hasattr(self._store, "get_by_hash") and self._store.get_by_hash(chash):
                continue

            # ``or Memory().id`` (not ``get(..., default)``) so null / empty
            # ids in the JSON payload still get a fresh value — see
            # tests/test_client_import_gaps.py.
            row_id = item.get("id") or Memory().id
            memory = Memory(
                id=row_id,
                content=content,
                tags=item.get("tags", []),
                category=item.get("category", "general"),
                entity=item.get("entity"),
                created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                event_date=item.get("event_date"),
                valid_from=item.get("valid_from", datetime.now(timezone.utc).isoformat()),
                valid_until=item.get("valid_until"),
                superseded_by=item.get("superseded_by"),
                confidence=item.get("confidence", 1.0),
                status=item.get("status", "active"),
                content_hash=chash,
                metadata=item.get("metadata", {}),
            )
            try:
                self._store.insert(memory)
                count += 1
            except Exception as e:
                # Don't mask non-dedup failures (PK violations, NOT-NULL
                # violations, connection errors). Log and keep going so a
                # bad row doesn't abort the rest of the import.
                logger.warning(
                    "import_json: insert failed for id=%s (%s): %s",
                    row_id, type(e).__name__, e,
                )
        return count

    # -- Graph --

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank scores from the entity graph. Returns empty dict if no graph."""
        if self._graph and hasattr(self._graph, "pagerank"):
            return self._graph.pagerank(alpha=alpha)
        return {}

    # -- Raw SQL --

    def execute(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Execute raw SQL. Use with caution."""
        return self._store.execute(sql, params)
