"""AgentMemory main class -- the public API."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from attestor.mode import AttestorMode, detect_mode
from attestor.models import Memory, Project, RetrievalResult, Session, User
from attestor.retrieval.orchestrator import RetrievalOrchestrator
from attestor.store.base import DocumentStore, GraphStore, VectorStore
from attestor.store.registry import (
    DEFAULT_BACKENDS,
    instantiate_backend,
    resolve_backends,
)
from attestor.temporal.manager import TemporalManager
from attestor.utils.config import MemoryConfig, load_config, save_config

logger = logging.getLogger("attestor")


@dataclass(frozen=True)
class ResolvedContext:
    """Result of AgentMemory._resolve() — the identity tuple for a call.

    Public methods take optional user_id/project_id/session_id; _resolve()
    fills any that are missing using the mode's defaults (SOLO singleton
    user, Inbox project, daily session) and returns the fully-resolved
    triple plus a derived AgentContext for downstream provenance.

    session can be None when autostart=False (e.g. for read-only recall
    that doesn't need a session anchor)."""
    user: User
    project: Project
    session: Optional[Session]


class AgentMemory:
    """Memory for AI agents, backed by Postgres (doc+pgvector) + Neo4j (graph).

    Usage:
        mem = AgentMemory("./my-agent", config={"backend_configs": {...}})
        mem.add("User prefers Python", tags=["preference"], category="preference")
        results = mem.recall("what language?")
    """

    def __init__(
        self,
        path: str | Path,
        config: Optional[Dict[str, Any]] = None,
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
        backends = getattr(self.config, "backends", None) or DEFAULT_BACKENDS
        backend_configs: Dict[str, Dict[str, Any]] = (
            getattr(self.config, "backend_configs", None) or {}
        )

        # Resolve role -> backend_name mapping
        role_assignments = resolve_backends(backends)

        # Docker manager (lazy, only needed for container-based backends)
        self._docker = None

        # Track instantiated backends so multi-role backends (e.g., ArangoDB)
        # reuse the same instance
        _instances: Dict[str, Any] = {}

        def _get_or_create(backend_name: str) -> Any:
            if backend_name in _instances:
                return _instances[backend_name]
            bcfg = backend_configs.get(backend_name, {})
            self._ensure_docker(backend_name, bcfg)
            instance = instantiate_backend(backend_name, self.path, bcfg)
            _instances[backend_name] = instance
            return instance

        # Initialize document store (required — no graceful degradation)
        doc_backend = role_assignments["document"]
        self._store: DocumentStore = _get_or_create(doc_backend)

        # Initialize vector store (optional — graceful degradation)
        self._vector_store: Optional[VectorStore] = None
        if "vector" in role_assignments:
            try:
                self._vector_store = _get_or_create(role_assignments["vector"])
            except Exception as e:
                logger.warning("Vector store init failed (%s): %s",
                               role_assignments["vector"], e)

        # Initialize graph store (optional — graceful degradation)
        self._graph: Optional[GraphStore] = None
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

        self._retrieval = RetrievalOrchestrator(
            self._store,
            min_results=self.config.min_results,
            vector_store=self._vector_store,
            graph=self._graph,
            bm25_lane=bm25_lane,
        )
        # Wire retrieval tuning from config
        self._retrieval.enable_mmr = self.config.enable_mmr
        self._retrieval.mmr_lambda = self.config.mmr_lambda
        self._retrieval.fusion_mode = self.config.fusion_mode
        self._retrieval.confidence_gate = self.config.confidence_gate
        self._retrieval.confidence_decay_rate = self.config.confidence_decay_rate
        self._retrieval.confidence_boost_rate = self.config.confidence_boost_rate

        # Operation ring buffer for latency observability
        self._ops_log: Deque[Dict[str, Any]] = deque(maxlen=200)

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
        self._default_user: Optional[User] = None
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

    # -- v4 identity (Phase 1 chunk 4) --
    #
    # The repos are constructed lazily and share the document store's
    # psycopg2 connection. That keeps the RLS variable consistent: any
    # SELECT / INSERT issued through a repo runs in the same session as
    # the memory writes that follow.
    #
    # Provisioning (create user / ensure inbox) requires admin / RLS-
    # bypassing context — the policies do not match yet because the user
    # row hasn't been seen by the policy yet. Callers running in v4 mode
    # are expected to use a Postgres role that owns the tables (RLS skipped
    # for table owners) for the bootstrap path. Once a user exists we set
    # the RLS variable and the rest of the surface is tenant-scoped.

    def _require_v4(self) -> None:
        if not getattr(self._store, "_v4", False):
            raise RuntimeError(
                "Identity APIs require v4 mode. Set ATTESTOR_V4=1 or pass "
                "v4=True in the postgres backend config."
            )

    def _pg_conn(self) -> Any:
        """Return the underlying psycopg2 connection on the document store.

        Raises if the store doesn't expose ``_conn`` — only Postgres-backed
        v4 deployments are supported in this chunk; ArangoDB / Cosmos / etc.
        get identity wiring in Phase 1.5."""
        conn = getattr(self._store, "_conn", None)
        if conn is None:
            raise RuntimeError(
                "Identity APIs currently require the Postgres backend "
                "(no _conn on store=%s)" % type(self._store).__name__
            )
        return conn

    def _user_repo(self) -> Any:
        from attestor.identity.users import UserRepo
        if not hasattr(self, "_users"):
            self._users = UserRepo(self._pg_conn())
        return self._users

    def _project_repo(self) -> Any:
        from attestor.identity.projects import ProjectRepo
        if not hasattr(self, "_projects"):
            self._projects = ProjectRepo(self._pg_conn())
        return self._projects

    def _session_repo(self) -> Any:
        from attestor.identity.sessions import SessionRepo
        if not hasattr(self, "_sessions"):
            self._sessions = SessionRepo(self._pg_conn())
        return self._sessions

    def ensure_user(
        self,
        external_id: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Idempotent first-login provisioning. Returns the user; safe to
        call on every request. Inbox project is created on first call so
        the caller never has to think about it."""
        self._require_v4()
        user = self._user_repo().create_or_get(
            external_id=external_id,
            email=email,
            display_name=display_name,
            metadata=metadata,
        )
        # Auto-provision Inbox so chat can start a session immediately.
        self._project_repo().ensure_inbox(user.id)
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        self._require_v4()
        return self._user_repo().get(user_id)

    def find_user_by_external_id(self, external_id: str) -> Optional[User]:
        self._require_v4()
        return self._user_repo().find_by_external_id(external_id)

    def ensure_inbox(self, user_id: str) -> Project:
        """Idempotent. The Inbox is the default project for sessions that
        haven't been assigned to one. Lives forever; cannot be deleted."""
        self._require_v4()
        return self._project_repo().ensure_inbox(user_id)

    def create_project(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Project:
        self._require_v4()
        if self._quotas is not None:
            self._quotas.check_project_quota(user_id)
        return self._project_repo().create(
            user_id=user_id, name=name,
            description=description, metadata=metadata,
        )

    def list_projects(
        self, user_id: str, include_inbox: bool = False, limit: int = 100,
    ) -> List[Project]:
        self._require_v4()
        return self._project_repo().list_for_user(
            user_id, include_inbox=include_inbox, limit=limit,
        )

    def start_session(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Open a new session. If ``project_id`` is None, drops it into the
        user's Inbox so the caller never has to provision one upfront."""
        self._require_v4()
        if self._quotas is not None:
            self._quotas.check_session_quota(user_id)
        if project_id is None:
            project_id = self.ensure_inbox(user_id).id
        return self._session_repo().create(
            user_id=user_id, project_id=project_id,
            title=title, metadata=metadata,
        )

    def get_or_create_daily_session(
        self, user_id: str, day: str, project_id: Optional[str] = None,
    ) -> Session:
        """SOLO-mode helper — one session per (user, ISO-date)."""
        self._require_v4()
        if project_id is None:
            project_id = self.ensure_inbox(user_id).id
        return self._session_repo().get_or_create_daily(
            user_id=user_id, project_id=project_id, day=day,
        )

    def end_session(self, session_id: str) -> Optional[Session]:
        self._require_v4()
        return self._session_repo().end(session_id)

    def list_sessions(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        status: str = "active",
        limit: int = 20,
    ) -> List[Session]:
        self._require_v4()
        return self._session_repo().list_for_user(
            user_id=user_id, project_id=project_id,
            status=status, limit=limit,
        )

    # -- v4 resolution chain (Phase 2 chunk 3, defaults.md §5) --

    def _resolve(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        autostart: bool = True,
    ) -> ResolvedContext:
        """Single entry point for identity resolution. Public methods call
        this first to get the (user, project, session) triple they should
        operate against.

        Resolution rules (defaults.md §5):
          User:    explicit user_id wins. Else SOLO singleton. Else raise.
          Project: explicit project_id wins. Else session.project_id. Else Inbox.
          Session: explicit session_id wins. Else autostart (daily for SOLO,
                   fresh for HOSTED). If autostart=False → None.

        v3 / non-Postgres backends: this method is a no-op and is not
        called. Public methods only invoke it when v4 is on.
        """
        self._require_v4()
        from attestor.identity.defaults import (
            ensure_solo_user,
            resolve_project,
            resolve_session,
        )

        # ── User ────────────────────────────────────────────────────────
        if user_id is None:
            if self._mode is AttestorMode.SOLO:
                # Lazy-recover: if SOLO bootstrap was skipped at __init__,
                # try once more here. Keeps the zero-config path working
                # even if the schema wasn't ready at construction time.
                if self._default_user is None:
                    self._default_user = ensure_solo_user(self._user_repo())
                user = self._default_user
            else:
                raise PermissionError(
                    f"user_id is required in {self._mode.value} mode"
                )
        else:
            # Pre-set RLS so the users-table lookup itself is admitted —
            # the policy on `users` is `id = current_setting(...)`. Without
            # this, even a valid user_id would resolve to "not found".
            if hasattr(self._store, "_set_rls_user"):
                self._store._set_rls_user(user_id)
            user = self._user_repo().get(user_id)
            if user is None or user.status != "active":
                raise LookupError(f"user {user_id} not found")

        # Set RLS to the resolved user (no-op when we already set it above).
        if hasattr(self._store, "_set_rls_user"):
            self._store._set_rls_user(user.id)

        # ── Project ────────────────────────────────────────────────────
        project = resolve_project(
            user_id=user.id,
            project_id=project_id,
            session_id=session_id,
            project_repo=self._project_repo(),
            session_repo=self._session_repo(),
        )

        # ── Session ────────────────────────────────────────────────────
        session = resolve_session(
            user_id=user.id,
            project_id=project.id,
            session_id=session_id,
            session_repo=self._session_repo(),
            autostart=autostart,
            mode_is_solo=(self._mode is AttestorMode.SOLO),
        )

        return ResolvedContext(user=user, project=project, session=session)

    # -- v4 round-level conversation ingest (Phase 3.5) --

    def ingest_round(
        self,
        user_turn: Any,                 # ConversationTurn
        assistant_turn: Any,            # ConversationTurn
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "user",
        agent_id: Optional[str] = None,
        recent_context: str = "(none)",
        config: Optional[Any] = None,        # IngestConfig
        extraction_client: Optional[Any] = None,
        resolver_client: Optional[Any] = None,
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
        model: Optional[str] = None,
        extraction_client: Optional[Any] = None,
        resolver_client: Optional[Any] = None,
    ) -> List[Any]:
        """Drain one batch from the consolidation queue in-process.

        For long-running daemons use ``SleepTimeConsolidator.run_forever``
        directly. This method is for tests and one-shot manual triggers
        (e.g., ``attestor consolidate run``).

        Returns a list of ``ConsolidationResult`` (one per processed
        episode). Empty list when the queue is drained.
        """
        self._require_v4()
        from attestor.consolidation import SleepTimeConsolidator
        kwargs: Dict[str, Any] = {"batch_size": limit}
        if model:
            kwargs["model"] = model
        if extraction_client is not None:
            kwargs["extraction_client"] = extraction_client
        if resolver_client is not None:
            kwargs["resolver_client"] = resolver_client
        cons = SleepTimeConsolidator(self, **kwargs)
        return cons.run_once(limit=limit)

    # -- v4 GDPR delete + export (Phase 8.5) --

    def export_user(self, external_id: str) -> Dict[str, Any]:
        """Full data portability dump for a user. JSON-serializable."""
        self._require_v4()
        from attestor.gdpr import export_user
        return export_user(self._store._conn, external_id).to_dict()

    def purge_user(
        self,
        external_id: str,
        *,
        reason: str = "gdpr_request",
        deleted_by: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def deletion_audit_log(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        """Recent GDPR deletion audit entries (read-only)."""
        self._require_v4()
        from attestor.gdpr import list_audit_log
        return list_audit_log(self._store._conn, limit=limit)

    # -- v4 quota management (Phase 8.3) --

    def set_quota(
        self,
        user_id: str,
        *,
        max_memories: Optional[int] = None,
        max_sessions: Optional[int] = None,
        max_projects: Optional[int] = None,
        max_writes_per_day: Optional[int] = None,
    ) -> Any:
        """Set per-user quotas. NULL/omitted limits leave that limit
        unchanged (no implicit unlimited reset)."""
        self._require_v4()
        if self._quotas is None:
            raise RuntimeError("quota repo unavailable on this backend")
        return self._quotas.set_limits(
            user_id,
            max_memories=max_memories,
            max_sessions=max_sessions,
            max_projects=max_projects,
            max_writes_per_day=max_writes_per_day,
        )

    def get_quota(self, user_id: str) -> Optional[Any]:
        """Return the current UserQuota row (counters + limits)."""
        self._require_v4()
        if self._quotas is None:
            return None
        return self._quotas.get(user_id)

    # -- v4 provenance signing (Phase 8.1) --

    def verify_memory(self, memory_id: str) -> bool:
        """Verify the Ed25519 signature on a stored memory.

        Returns True if signed AND the signature matches; False if
        unsigned, missing, or tampered.

        Raises RuntimeError if signing isn't configured for this
        instance — verification needs the public key in the same place.
        """
        if self._signer is None:
            raise RuntimeError(
                "verify_memory requires config['signing'] to be set "
                "(public_key needed for verification)"
            )
        row = self._store.get(memory_id)
        if row is None:
            return False
        return self._signer.verify(row)

    @property
    def signing_enabled(self) -> bool:
        return self._signer is not None

    @property
    def mode(self) -> AttestorMode:
        """The detected operating mode. Settable via ``config["mode"]`` or
        the ``ATTESTOR_MODE`` env var."""
        return self._mode

    @property
    def default_user(self) -> Optional[User]:
        """The SOLO singleton user, if SOLO mode is active. None otherwise."""
        return self._default_user

    def _ensure_docker(self, backend_name: str, bcfg: Dict[str, Any]) -> None:
        """Start a Docker container for backends that require one.

        Docker auto-management is opt-in: requires bcfg["docker"] = True.
        Falls back to a no-op if the optional infra module is unavailable.
        """
        if bcfg.get("mode") == "cloud" or bcfg.get("url", "").startswith("https://"):
            return
        if not bcfg.get("docker"):
            return

        # Opt-in user said docker=true — propagate install instructions if the
        # extra isn't installed rather than silently no-op-ing.
        from attestor.infra.docker import DockerManager

        if self._docker is None:
            self._docker = DockerManager()
        docker_images = {
            "arangodb": ("arangodb:3.12", 8529, {"ARANGO_NO_AUTH": "1"}),
        }
        if backend_name in docker_images:
            image, default_port, env = docker_images[backend_name]
            port = bcfg.get("port", default_port)
            self._docker.ensure_running(backend_name, image, port, env)

    # -- Write --

    @staticmethod
    def _content_hash(content: str) -> str:
        """Compute SHA-256 hash of normalized content for dedup."""
        return hashlib.sha256(content.strip().encode()).hexdigest()

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        entity: Optional[str] = None,
        namespace: str = "default",
        event_date: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        # ── v4 tenancy params (Phase 1 chunk 3) ──
        # When user_id is provided AND the backend is in v4 mode, the memory
        # is written through the v4 path (RLS-scoped, bi-temporal). When
        # user_id is None, the legacy v3 path runs unchanged.
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "user",
        agent_id: Optional[str] = None,
        source_episode_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory, handling contradictions automatically.

        v4 callers should pass ``user_id`` (and optionally project_id,
        session_id, scope). v3 callers can omit them — behavior unchanged."""
        t_total = time.monotonic()
        store_timings: Dict[str, float] = {}

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

        memory = Memory(
            content=content,
            tags=tags or [],
            category=category,
            entity=entity,
            namespace=namespace,
            event_date=event_date,
            confidence=confidence,
            content_hash=chash,
            metadata=metadata or {},
            # v4 fields (None / defaults when caller didn't pass them)
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            scope=scope,
            agent_id=agent_id,
            source_episode_id=source_episode_id,
        )

        # Check for contradictions before insert
        contradictions = self._temporal.check_contradictions(memory)

        # Insert new memory first (so FK reference is valid)
        t0 = time.monotonic()
        self._store.insert(memory)
        store_timings["document_ms"] = round((time.monotonic() - t0) * 1000, 2)

        # v4 provenance signing (Phase 8.1) — sign AFTER insert so the
        # canonical payload includes the DB-generated id + t_created.
        if self._signer is not None and v4_active:
            try:
                sig = self._signer.sign(memory)
                memory.signature = sig
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
        # (silent pass made dim-mismatch / Ollama-down / schema-drift bugs
        # invisible until recall returned 0 hits).
        if self._vector_store:
            try:
                t0 = time.monotonic()
                self._vector_store.add(memory.id, content, namespace=namespace)
                store_timings["vector_ms"] = round((time.monotonic() - t0) * 1000, 2)
            except Exception as e:
                store_timings["vector_ms"] = -1  # failed
                logger.warning(
                    "vector add failed for memory %s: %s: %s",
                    memory.id, type(e).__name__, e,
                )

        # Update entity graph (also non-fatal; surface exceptions for the
        # same reason — silent drops here mean recall layer 2 returns
        # nothing without the operator knowing).
        if self._graph:
            try:
                from attestor.graph.extractor import extract_entities_and_relations
                t0 = time.monotonic()
                nodes, edges = extract_entities_and_relations(
                    content, tags or [], entity, category,
                )
                for node in nodes:
                    self._graph.add_entity(
                        node["name"],
                        entity_type=node.get("type", "general"),
                        attributes=node.get("attributes"),
                    )
                for edge in edges:
                    self._graph.add_relation(
                        edge["from"],
                        edge["to"],
                        relation_type=edge.get("type", "related_to"),
                        metadata=edge.get("metadata"),
                    )
                store_timings["graph_ms"] = round((time.monotonic() - t0) * 1000, 2)
            except Exception as e:
                store_timings["graph_ms"] = -1  # failed
                logger.warning(
                    "graph add failed for memory %s: %s: %s",
                    memory.id, type(e).__name__, e,
                )

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

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> Optional[Memory]:
        """Update an existing memory's fields. Returns updated memory or None."""
        memory = self._store.get(memory_id)
        if not memory:
            return None

        if content is not None:
            memory.content = content
            memory.content_hash = self._content_hash(content)
        if tags is not None:
            memory.tags = tags
        if category is not None:
            memory.category = category
        if entity is not None:
            memory.entity = entity

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
        budget: Optional[int] = None,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
        time_window: Optional[Any] = None,    # TimeWindow
    ) -> List[RetrievalResult]:
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
        recall_kwargs: Dict[str, Any] = {"namespace": namespace}
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
        budget: Optional[int] = None,
        namespace: Optional[str] = None,
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
        budget: Optional[int] = None,
        user_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
        time_window: Optional[Any] = None,
        chain_of_note_prompt: Optional[str] = None,
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
        query: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        namespace: Optional[str] = None,
        status: str = "active",
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
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
        self, entity: str, namespace: Optional[str] = None
    ) -> List[Memory]:
        """Get all memories about an entity in chronological order."""
        return self._temporal.timeline(entity, namespace=namespace)

    def current_facts(
        self,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Memory]:
        """Get only active, non-superseded memories."""
        return self._temporal.current_facts(
            category=category, entity=entity, namespace=namespace
        )

    # -- Extraction --

    def extract(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        use_llm: bool = False,
        namespace: str = "default",
    ) -> List[Memory]:
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
            memory.status = "archived"
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
    def ops_log(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the operation ring buffer (most recent last)."""
        return list(self._ops_log)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self._store.stats()

    def _try_recover_stores(self) -> Dict[str, str]:
        """Attempt to re-initialize failed vector/graph stores.

        Called by health() when stores are None but config says they should
        exist. On success, wires the recovered store into the retrieval
        pipeline so the running process heals without a restart.

        Returns:
            Dict of role -> outcome ("recovered" | "failed: <reason>")
        """
        backends = getattr(self.config, "backends", None) or DEFAULT_BACKENDS
        backend_configs: Dict[str, Dict[str, Any]] = (
            getattr(self.config, "backend_configs", None) or {}
        )
        role_assignments = resolve_backends(backends)
        results: Dict[str, str] = {}

        # Try to recover vector store
        if self._vector_store is None and "vector" in role_assignments:
            backend_name = role_assignments["vector"]
            try:
                self._vector_store = instantiate_backend(
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
                self._graph = instantiate_backend(
                    backend_name, self.path, backend_configs.get(backend_name),
                )
                self._retrieval.graph = self._graph
                logger.info("Recovered graph store (%s)", backend_name)
                results["graph"] = "recovered"
            except Exception as e:
                logger.warning("Graph store recovery failed (%s): %s", backend_name, e)
                results["graph"] = f"failed: {e}"

        return results

    def health(self) -> Dict[str, Any]:
        """Check health of all components. Returns structured status report.

        If vector or graph stores failed at startup, attempts recovery before
        reporting. This lets long-running processes (like the MCP server)
        self-heal without a restart.

        Checks: Document Store (Postgres), Vector Store (pgvector), Graph
        Store (Neo4j), Retrieval Pipeline.
        """
        t0_health = time.monotonic()

        # Attempt recovery of failed stores before reporting
        recovery: Dict[str, str] = {}
        if self._vector_store is None or self._graph is None:
            recovery = self._try_recover_stores()

        report: Dict[str, Any] = {
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
            details: Dict[str, Any] = {
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
                    vec_details: Dict[str, Any] = {
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
                graph_details: Dict[str, Any] = {
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
        """Import memories from a JSON file. Returns count imported."""
        with open(filepath) as f:
            data = json.load(f)
        count = 0
        for item in data:
            content = item["content"]
            chash = self._content_hash(content)

            # Skip if duplicate content already exists
            if hasattr(self._store, "get_by_hash") and self._store.get_by_hash(chash):
                continue

            memory = Memory(
                id=item.get("id", Memory().id),
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
            except Exception:
                # Skip duplicates
                pass
        return count

    # -- Graph --

    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Compute PageRank scores from the entity graph. Returns empty dict if no graph."""
        if self._graph and hasattr(self._graph, "pagerank"):
            return self._graph.pagerank(alpha=alpha)
        return {}

    # -- Raw SQL --

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL. Use with caution."""
        return self._store.execute(sql, params)
