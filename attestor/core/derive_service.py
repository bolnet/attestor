# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Derived-state writes (vector + graph) shared by ``add()`` and the durable repair path.

Postgres is the source of truth; the Pinecone vector and the Neo4j graph
are *derived* from it and must be rebuildable by memory id. This mixin
owns that single code path so ``AgentMemory.add()`` (synchronous
fast-path) and the ``DeriveMemory`` Temporal activities (retried repair /
``attestor durable rebuild``) write byte-identical derived state.

Public re-read seam (called by the activities):

* :meth:`_DeriveMixin.derive_vector` — re-read the row by id, rebuild the
  embedding payload (including a stored contextual prefix), upsert.
* :meth:`_DeriveMixin.derive_graph` — re-read the row by id, re-extract
  entities / relations, MERGE them into the graph.

Both raise :class:`MemoryNotFoundError` (permanent) when the id is gone
and :class:`DerivedStoreUnavailableError` (transient) when the derived
store is not initialised — the activity layer maps these onto Temporal
retry semantics.

Tenant scope: every re-read / write runs AS THE ROW'S OWNER. The seam
sets the Postgres RLS session user (``_set_rls_user``, mirroring
``consolidation/consolidator.py``) to the caller's ``user_id`` before the
re-read, applies the ``visibility='private'`` requester guard with
``agent_id``, refuses a row owned by another user (a BYPASSRLS worker
role could otherwise read it), and resets the session to the row's own
owner before the derived write. A v4 store with no scope at all resolves
the SOLO default user or raises :class:`RebuildScopeError`; a v3 store
(no ``user_id`` column, no RLS) stays unscoped.

The RLS session user is SESSION state on a connection the durable
worker shares across activity threads, so each derive / list holds the
store's re-entrant connection lock (``_get_conn_lock``) for the WHOLE
scope-set → read → ownership check → write sequence; a concurrent
activity for another tenant cannot flip the session user mid-flight.

Repair scheduling (plan Phase 2): when a derived write fails inside
``add()`` and ``durable.enabled`` is true, ``_schedule_derive_repair``
starts ``derive-{memory_id}`` fire-and-forget. When durable is disabled
the call returns ``None`` without importing ``attestor.durable``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from attestor import trace as _tr

if TYPE_CHECKING:
    from attestor.config import DurableCfg
    from attestor.models import Memory

logger = logging.getLogger("attestor.derive")

LANE_VECTOR = "vector"
LANE_GRAPH = "graph"
CONTEXT_PREFIX_METADATA_KEY = "_context_prefix"
REPAIR_TRACE_EVENT = "ingest.write.repair_scheduled"


class MemoryNotFoundError(LookupError):
    """No memory row with that id — re-deriving cannot succeed."""


class DerivedStoreUnavailableError(RuntimeError):
    """The vector / graph store is not initialised right now (retry later)."""


class RebuildScopeError(PermissionError):
    """Multi-tenant store, no user scope — refuse to derive / list across tenants."""


class RebuildUnsupportedError(TypeError):
    """The document store cannot page memory ids (rebuild needs Postgres)."""


REBUILD_SCOPE_HINT = "attestor durable rebuild --user <user_id>"


@dataclass(frozen=True)
class DerivedIdPage:
    """One page of rebuildable ids plus the tenant scope they were listed under."""

    ids: tuple[str, ...]
    next_after_id: str | None
    user_id: str | None


@dataclass(frozen=True)
class DerivedVector:
    memory_id: str
    namespace: str
    payload_len: int


@dataclass(frozen=True)
class DerivedGraph:
    memory_id: str
    namespace: str
    entity_count: int
    relation_count: int


def embed_payload_for(memory: Memory) -> str:
    """The exact text ``add()`` embedded for ``memory``.

    A contextual prefix (``ingest.contextual_embedding``) is persisted on
    ``metadata["_context_prefix"]`` so a re-derive reproduces the same
    ``[CTX] <prefix>\\n\\n<content>`` wire payload.
    """
    prefix = (memory.metadata or {}).get(CONTEXT_PREFIX_METADATA_KEY)
    if not prefix:
        return memory.content
    from attestor.ingest.contextual import ContextualEmbedder

    return ContextualEmbedder._format_payload(memory.content, prefix)


def resolve_durable_cfg() -> DurableCfg | None:
    """``durable:`` block from the YAML (+ ``TEMPORAL_*`` env when enabled), or ``None``.

    Never raises: a missing / unparsable config means "stay in-process",
    which is exactly today's behaviour.
    """
    try:
        from attestor.config import get_stack

        cfg = getattr(get_stack(strict=False), "durable", None)
    except Exception as exc:
        logger.debug("durable config unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    if cfg is None or not cfg.enabled:
        return cfg
    from attestor.durable.config import resolve_durable

    return resolve_durable(cfg)


def _add_entities(graph: Any, nodes: list[dict[str, Any]], namespace: str) -> None:
    # Older graph backends without the ``namespace`` kwarg still accept
    # the call via the TypeError fallback.
    for node in nodes:
        try:
            graph.add_entity(
                node["name"],
                entity_type=node.get("type", "general"),
                attributes=node.get("attributes"),
                namespace=namespace,
            )
        except TypeError:
            graph.add_entity(
                node["name"],
                entity_type=node.get("type", "general"),
                attributes=node.get("attributes"),
            )


def _add_relations(graph: Any, edges: list[dict[str, Any]], namespace: str) -> None:
    for edge in edges:
        try:
            graph.add_relation(
                edge["from"],
                edge["to"],
                relation_type=edge.get("type", "related_to"),
                metadata=edge.get("metadata"),
                namespace=namespace,
            )
        except TypeError:
            graph.add_relation(
                edge["from"],
                edge["to"],
                relation_type=edge.get("type", "related_to"),
                metadata=edge.get("metadata"),
            )


class _DeriveMixin:
    """Vector + graph derived-state writes. Mixed into ``AgentMemory``."""

    _store: Any
    _vector_store: Any
    _graph: Any
    _ingest_cfg: Any

    # ── lanes shared with add() ─────────────────────────────────────────

    def _vector_sink(self) -> Any | None:
        """The vector store, or the document store's pgvector mixin as fallback."""
        sink = self._vector_store
        if sink is None and hasattr(self._store, "_embedding_dim"):
            sink = self._store
        return sink

    def _write_vector(self, memory_id: str, payload: str, namespace: str) -> None:
        sink = self._vector_sink()
        if sink is None:
            raise DerivedStoreUnavailableError("no vector store initialised")
        sink.add(memory_id, payload, namespace=namespace)

    def _extract_graph(
        self,
        content: str,
        tags: list[str],
        entity: str | None,
        category: str,
        namespace: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Entity / relation extraction, honouring ``ingest.llm_entity_extraction``.

        When the LLM path is enabled it falls back to the regex extractor
        on LLM failure (never to empty); when disabled it is the plain
        regex extractor.
        """
        le_cfg = getattr(self._ingest_cfg, "llm_entity_extraction", None)
        if le_cfg is not None and le_cfg.enabled:
            from attestor.extraction.llm_entity_extractor import extract_or_regex

            return extract_or_regex(
                content, tags, entity, category,
                namespace=namespace,
                llm_enabled=True,
                llm_model=le_cfg.model,
                llm_timeout=le_cfg.timeout_s,
            )
        from attestor.graph.extractor import extract_entities_and_relations

        return extract_entities_and_relations(
            content, tags, entity, category, namespace=namespace,
        )

    def _write_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        namespace: str,
    ) -> None:
        graph = self._graph
        if graph is None:
            raise DerivedStoreUnavailableError("no graph store initialised")
        _add_entities(graph, nodes, namespace)
        _add_relations(graph, edges, namespace)

    # ── tenant scope (RLS) ──────────────────────────────────────────────

    def _tenancy_enforced(self) -> bool:
        """v4 schema: ``memories.user_id`` + RLS policies keyed on the session user."""
        return bool(getattr(self._store, "_v4", False))

    def _resolve_scope_user(self, user_id: str | None) -> str | None:
        """The tenant a derive / list call runs as.

        Explicit id → itself. v3 store → ``None`` (nothing to scope).
        v4 with a SOLO default user → that user. v4 otherwise → refuse:
        never touch rows across tenants implicitly.
        """
        if user_id:
            return str(user_id)
        if not self._tenancy_enforced():
            return None
        default = getattr(self, "_default_user", None)
        if default is not None:
            return str(default.id)
        raise RebuildScopeError(
            "the document store enforces per-user isolation; derive / rebuild "
            f"needs an explicit user scope ({REBUILD_SCOPE_HINT})"
        )

    def _scope_lock(self) -> Any:
        """The store's connection lock (re-entrant), or a no-op for lock-less stores."""
        getter = getattr(self._store, "_get_conn_lock", None)
        return getter() if getter is not None else nullcontext()

    def _set_scope(self, user_id: str | None) -> None:
        """Set the Postgres RLS session user (no-op when there is no scope)."""
        if user_id is None:
            return
        setter = getattr(self._store, "_set_rls_user", None)
        if setter is None:
            raise RebuildScopeError(
                f"document store {type(self._store).__name__} enforces tenancy "
                "but exposes no _set_rls_user; refusing to derive unscoped"
            )
        setter(user_id)

    # ── re-read seam for the durable activities ─────────────────────────

    def _require_memory(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None,
    ) -> Memory:
        """Re-read one row as ``user_id`` and leave the session scoped to its owner."""
        scope = self._resolve_scope_user(user_id)
        self._set_scope(scope)
        if agent_id:
            memory = self._store.get(memory_id, requester_agent_id=agent_id)
        else:
            memory = self._store.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(f"memory {memory_id} not found in the document store")
        owner = str(memory.user_id) if memory.user_id else None
        if scope is not None and owner is not None and owner != scope:
            # Readable only because the role bypasses RLS — still not ours.
            raise MemoryNotFoundError(f"memory {memory_id} is not owned by user {scope}")
        self._set_scope(owner or scope)
        return memory

    def _recover_if_missing(self, lane: str) -> None:
        missing = self._vector_sink() is None if lane == LANE_VECTOR else self._graph is None
        if missing and hasattr(self, "_try_recover_stores"):
            self._try_recover_stores()

    def derive_vector(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None,
    ) -> DerivedVector:
        """Re-embed + upsert one memory from its Postgres row (idempotent by id)."""
        with self._scope_lock():
            memory = self._require_memory(memory_id, user_id=user_id, agent_id=agent_id)
            self._recover_if_missing(LANE_VECTOR)
            payload = embed_payload_for(memory)
            self._write_vector(memory.id, payload, memory.namespace)
        return DerivedVector(
            memory_id=memory.id, namespace=memory.namespace, payload_len=len(payload),
        )

    def derive_graph(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None,
    ) -> DerivedGraph:
        """Re-extract + MERGE one memory's entities / relations (idempotent by name)."""
        with self._scope_lock():
            memory = self._require_memory(memory_id, user_id=user_id, agent_id=agent_id)
            self._recover_if_missing(LANE_GRAPH)
            if self._graph is None:
                raise DerivedStoreUnavailableError("no graph store initialised")
            nodes, edges = self._extract_graph(
                memory.content, list(memory.tags or []), memory.entity,
                memory.category, memory.namespace,
            )
            self._write_graph(nodes, edges, memory.namespace)
        return DerivedGraph(
            memory_id=memory.id, namespace=memory.namespace,
            entity_count=len(nodes), relation_count=len(edges),
        )

    def list_derivable_ids(
        self,
        *,
        user_id: str | None,
        since: Any,
        namespace: str | None,
        after_id: str | None,
        limit: int,
    ) -> DerivedIdPage:
        """One keyset page of active memory ids for ONE tenant (``RebuildDerived``).

        Sets the RLS session user to the resolved scope and filters the
        listing by it, so a BYPASSRLS role still pages a single tenant.
        """
        lister = getattr(self._store, "list_memory_ids", None)
        if lister is None:
            raise RebuildUnsupportedError(
                f"document store {type(self._store).__name__} cannot list memory ids; "
                "rebuild requires the Postgres document backend"
            )
        scope = self._resolve_scope_user(user_id)
        with self._scope_lock():
            self._set_scope(scope)
            ids, next_after_id = lister(
                since=since, namespace=namespace, user_id=scope, after_id=after_id,
                limit=limit,
            )
        return DerivedIdPage(ids=tuple(ids), next_after_id=next_after_id, user_id=scope)

    # ── durable repair ──────────────────────────────────────────────────

    def _schedule_derive_repair(self, memory: Memory, *, lanes: tuple[str, ...]) -> str | None:
        """Start ``derive-{memory.id}`` when durable is enabled; else ``None``.

        Fire-and-forget: never blocks ingest, never raises. The payload is
        ``DeriveRef.from_memory`` — ids plus owner / agent scope, never
        content. Emits ``ingest.write.repair_scheduled`` on success.
        """
        cfg = resolve_durable_cfg()
        if cfg is None or not cfg.enabled:
            return None
        from attestor.durable import dispatch
        from attestor.durable.models import DeriveRef, DeriveRequest

        memory_id, namespace = memory.id, memory.namespace
        request = DeriveRequest(memory=DeriveRef.from_memory(memory))
        try:
            workflow_id = dispatch.schedule_derive(cfg, request)
        except Exception as exc:
            logger.warning(
                "durable repair dispatch failed for memory %s (lanes=%s): %s: %s",
                memory_id, ",".join(lanes), type(exc).__name__, exc,
            )
            return None
        logger.info(
            "durable repair scheduled for memory %s (lanes=%s) as %s",
            memory_id, ",".join(lanes), workflow_id,
        )
        if _tr.is_enabled():
            _tr.event(
                REPAIR_TRACE_EVENT,
                memory_id=memory_id, namespace=namespace, lanes=list(lanes),
                workflow_id=workflow_id, task_queue=cfg.task_queue,
            )
        return workflow_id
