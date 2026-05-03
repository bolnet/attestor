"""AgentContext -- shared state for multi-agent memory consumption.

This is the context object that gets passed between agents in a multi-agent
system. It carries agent identity, memory access, accumulated state, and
coordination metadata. Modeled after production patterns (e.g., LangGraph
state, Bedrock agent context).

Usage:
    # Orchestrator creates context
    ctx = AgentContext(
        agent_id="planner-01",
        session_id="sess-abc123",
        namespace="project:acme",
        memory=AgentMemory("./shared-store"),
    )

    # Pass to sub-agents -- each sets their identity
    ctx = ctx.as_agent("researcher-01")
    ctx.add_memory("User prefers Python over Go", tags=["preference"])

    # Sub-agent recalls with scoping
    results = ctx.recall("language preferences")

    # Orchestrator reads accumulated state
    print(ctx.memories_written)   # what this session produced
    print(ctx.entities_discovered)  # new entities found
    print(ctx.agent_trail)         # full agent handoff chain
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from attestor.models import Memory, MemoryScope, RetrievalResult


class Visibility(str, Enum):
    """Memory visibility in multi-agent systems."""
    PUBLIC = "public"          # All agents can read
    TEAM = "team"              # Only agents in same namespace can read
    PRIVATE = "private"        # Only the writing agent can read
    SYSTEM = "system"          # Infrastructure-level, always readable


class AgentRole(str, Enum):
    """Standard agent roles for access control."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    MONITOR = "monitor"


class RolePermission(str, Enum):
    """Capabilities granted by a role.

    READ      -- recall / search / timeline (always granted; recall path
                 stays open so a misconfigured role never blinds an agent)
    WRITE     -- add_memory
    FORGET    -- forget (archival; admin-tier capability)
    """

    READ = "read"
    WRITE = "write"
    FORGET = "forget"


# Role -> capability matrix. ORCHESTRATOR is the admin tier (full perms).
# REVIEWER + MONITOR are read-only by intent: a reviewer that writes
# contaminates the evidence trail; a monitor that writes is a feedback
# loop. PLANNER / EXECUTOR / RESEARCHER need write but not forget --
# immutability is a feature, only the orchestrator can archive.
ROLE_PERMISSIONS: dict[AgentRole, frozenset[RolePermission]] = {
    AgentRole.ORCHESTRATOR: frozenset(
        {RolePermission.READ, RolePermission.WRITE, RolePermission.FORGET}
    ),
    AgentRole.PLANNER:   frozenset({RolePermission.READ, RolePermission.WRITE}),
    AgentRole.EXECUTOR:  frozenset({RolePermission.READ, RolePermission.WRITE}),
    AgentRole.RESEARCHER: frozenset({RolePermission.READ, RolePermission.WRITE}),
    AgentRole.REVIEWER:  frozenset({RolePermission.READ}),
    AgentRole.MONITOR:   frozenset({RolePermission.READ}),
}


@dataclass
class AgentContext:
    """Shared context for multi-agent memory consumption.

    Designed to be passed between agents in a pipeline. Each agent reads
    from and writes to the shared memory store through this context,
    which tracks provenance, enforces scoping, and accumulates results.
    """

    # -- Identity --
    agent_id: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    namespace: str = "default"
    role: AgentRole = AgentRole.EXECUTOR

    # -- v4 tenancy (None when running in v3-compat / SOLO-bootstrap mode) --
    user_id: str | None = None
    project_id: str | None = None
    scope_default: MemoryScope = MemoryScope.USER

    # -- Memory access --
    # Either an AgentMemory instance (embedded) or an HTTP client (distributed).
    # Typed as Any to avoid circular import; runtime checks enforce the interface.
    memory: Any = None               # AgentMemory | MemoryClient
    memory_url: str | None = None  # HTTP endpoint for distributed mode

    # -- Token budget --
    token_budget: int = 20000         # Total budget for this session
    token_budget_used: int = 0        # Consumed so far
    max_writes_per_agent: int = 100   # Write quota per sub-agent

    # -- Agent tracking --
    current_agent: str = ""
    agent_trail: list[str] = field(default_factory=list)
    parent_agent_id: str | None = None

    # -- Accumulated state: memories written this session --
    memories_written: list[str] = field(default_factory=list)      # memory IDs
    memories_recalled: list[str] = field(default_factory=list)     # memory IDs
    entities_discovered: list[str] = field(default_factory=list)   # entity names

    # -- Recall cache (avoid redundant queries within a session) --
    _recall_cache: dict[str, list[RetrievalResult]] = field(
        default_factory=dict, repr=False
    )

    # -- Scratchpad: structured data passed between agents --
    # Like PlannerContext's snapshot/analysis_results/recommendations,
    # but generic. Each agent writes to its own key.
    scratchpad: dict[str, Any] = field(default_factory=dict)

    # -- Governance --
    visibility: Visibility = Visibility.PUBLIC
    compliance_tags: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    read_only: bool = False           # If True, agent can recall but not write

    # -- Timestamps --
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    def as_agent(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.EXECUTOR,
        read_only: bool | None = None,
        token_budget: int | None = None,
    ) -> AgentContext:
        """Create a child context for a sub-agent.

        Preserves shared *config* (memory backend, session id, namespace,
        budget) but gives the child its own *mutable* state copies so
        sibling agents don't stomp each other's scratchpad / recall cache /
        accumulated id lists. Immutable contract: never mutates ``self``.

        ``read_only`` defaults to ``None`` so the parent's value
        propagates: a read-only orchestrator must not be able to spawn
        a writeable sub-agent simply by omitting the kwarg. Pass
        ``read_only=False`` explicitly to override the parent (rare —
        usually a security regression).
        """
        effective_read_only = (
            self.read_only if read_only is None else read_only
        )
        return replace(
            self,
            **self._derived_fields(
                agent_id=agent_id,
                role=role,
                token_budget=token_budget or self.token_budget,
                read_only=effective_read_only,
            ),
        )

    def _derived_fields(
        self,
        *,
        agent_id: str,
        role: AgentRole,
        token_budget: int,
        read_only: bool,
    ) -> dict[str, Any]:
        """Compute the field overrides for a derived child context.

        Centralizes the deep-copy / shallow-copy policy so any future
        ``.as_<role>()`` factory uses the same isolation rules:

        - ``scratchpad`` and ``_recall_cache`` carry arbitrary user values
          (and lists of RetrievalResult); deep-copy so child writes can't
          leak back into the parent or to siblings.
        - ``agent_trail`` / ``memories_written`` / ``memories_recalled`` /
          ``entities_discovered`` / ``compliance_tags`` are lists of
          immutable ids/strings; a shallow copy is enough.
        """
        return {
            "agent_id": agent_id,
            "role": role,
            "token_budget": token_budget,
            "read_only": read_only,
            "current_agent": agent_id,
            "parent_agent_id": self.agent_id,
            "agent_trail": [*self.agent_trail, agent_id],
            "memories_written": list(self.memories_written),
            "memories_recalled": list(self.memories_recalled),
            "entities_discovered": list(self.entities_discovered),
            "compliance_tags": list(self.compliance_tags),
            "scratchpad": copy.deepcopy(self.scratchpad),
            "_recall_cache": copy.deepcopy(self._recall_cache),
            # The child is a fresh derivation; stamp its own creation time
            # so audit trails reflect when the sub-agent started, not when
            # the orchestrator did.
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def for_chat(
        cls,
        user_id: str,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str = "chat",
        role: AgentRole = AgentRole.EXECUTOR,
        scope_default: MemoryScope = MemoryScope.USER,
        memory: Any = None,
        memory_url: str | None = None,
        token_budget: int = 20000,
    ) -> AgentContext:
        """Build a v4 chat-flow context anchored to a real user.

        ``namespace`` is auto-derived as ``user:{uid}[/project:{pid}][/session:{sid}]``
        for backward compat with tag/log code that still reads it. The structured
        ``user_id`` / ``project_id`` / ``session_id`` are the source of truth for
        all DB queries; namespace is denormalized only.
        """
        return cls(
            agent_id=agent_id,
            session_id=session_id or uuid.uuid4().hex[:16],
            namespace=cls._build_namespace(user_id, project_id, session_id),
            role=role,
            user_id=user_id,
            project_id=project_id,
            scope_default=scope_default,
            memory=memory,
            memory_url=memory_url,
            token_budget=token_budget,
            current_agent=agent_id,
            agent_trail=[agent_id],
        )

    @staticmethod
    def _build_namespace(
        user_id: str,
        project_id: str | None,
        session_id: str | None,
    ) -> str:
        ns = f"user:{user_id}"
        if project_id:
            ns += f"/project:{project_id}"
        if session_id:
            ns += f"/session:{session_id}"
        return ns

    @classmethod
    def from_env(cls, agent_id: str, **overrides: Any) -> AgentContext:
        """Create context from environment variables.

        Reads: ATTESTOR_PATH, ATTESTOR_URL, ATTESTOR_NAMESPACE,
        ATTESTOR_TOKEN_BUDGET, ATTESTOR_SESSION_ID.
        """
        import os

        from attestor import _branding as brand

        path = os.environ.get(brand.ENV_STORE_PATH)
        url = os.environ.get(brand.ENV_URL)
        namespace = os.environ.get(brand.ENV_NAMESPACE, "default")
        budget = int(os.environ.get(brand.ENV_TOKEN_BUDGET, "20000"))
        session_id = os.environ.get(brand.ENV_SESSION_ID, uuid.uuid4().hex[:16])

        memory = None
        if path and not url:
            from attestor.core import AgentMemory
            memory = AgentMemory(path)

        return cls(
            agent_id=agent_id,
            session_id=session_id,
            namespace=namespace,
            memory=memory,
            memory_url=url,
            token_budget=budget,
            current_agent=agent_id,
            agent_trail=[agent_id],
            **overrides,
        )

    # ------------------------------------------------------------------ #
    #  Memory operations (delegate to memory store with provenance)        #
    # ------------------------------------------------------------------ #

    def _get_memory(self) -> Any:
        """Get the memory backend, initializing HTTP client if needed."""
        if self.memory is not None:
            return self.memory
        if self.memory_url:
            from attestor.client import MemoryClient
            self.memory = MemoryClient(self.memory_url, agent_id=self.agent_id)
            return self.memory
        raise RuntimeError(
            "No memory backend configured. Set memory= or memory_url= "
            "or use ATTESTOR_PATH / ATTESTOR_URL env vars."
        )

    # ------------------------------------------------------------------ #
    #  RBAC                                                                #
    # ------------------------------------------------------------------ #

    def _require_permission(self, perm: RolePermission) -> None:
        """Raise PermissionError if the current role lacks ``perm``.

        Looks up the role in ROLE_PERMISSIONS. ``read_only=True`` is an
        independent gate -- it strips WRITE/FORGET regardless of role
        (so an ORCHESTRATOR with read_only=True still cannot write).
        """
        granted = ROLE_PERMISSIONS.get(self.role, frozenset())
        if self.read_only and perm in (RolePermission.WRITE, RolePermission.FORGET):
            raise PermissionError(
                f"Agent '{self.agent_id}' is read-only in this context"
            )
        if perm not in granted:
            raise PermissionError(
                f"Agent '{self.agent_id}' (role={self.role.value}) "
                f"lacks permission '{perm.value}'"
            )

    def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str = "general",
        entity: str | None = None,
        event_date: str | None = None,
        confidence: float = 1.0,
        visibility: Visibility | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a memory with full provenance tracking.

        Attaches agent_id, session_id, namespace, and visibility to metadata.
        Enforces role permissions, write quota, and read_only constraints.
        """
        self._require_permission(RolePermission.WRITE)

        sum(
            1 for mid in self.memories_written
            # Count is approximate -- just length-based
        )
        if len(self.memories_written) >= self.max_writes_per_agent:
            raise RuntimeError(
                f"Agent '{self.agent_id}' exceeded write quota "
                f"({self.max_writes_per_agent})"
            )

        # Enrich metadata with provenance
        enriched_metadata = {
            **(metadata or {}),
            "_agent_id": self.agent_id,
            "_session_id": self.session_id,
            "_namespace": self.namespace,
            "_visibility": (visibility or self.visibility).value,
            "_role": self.role.value,
            "_parent_agent_id": self.parent_agent_id,
        }

        mem = self._get_memory()
        result = mem.add(
            content=content,
            tags=tags,
            category=category,
            entity=entity,
            namespace=self.namespace,
            event_date=event_date,
            confidence=confidence,
            metadata=enriched_metadata,
        )

        self.memories_written.append(result.id)

        # Track entities
        if entity and entity not in self.entities_discovered:
            self.entities_discovered.append(entity)

        return result

    def recall(
        self,
        query: str,
        budget: int | None = None,
        use_cache: bool = True,
    ) -> list[RetrievalResult]:
        """Recall memories with budget tracking and caching.

        Caches results within a session to avoid redundant vector searches.
        Tracks token budget consumption.
        """
        if use_cache and query in self._recall_cache:
            return self._recall_cache[query]

        effective_budget = budget or (self.token_budget - self.token_budget_used)
        if effective_budget <= 0:
            return []

        mem = self._get_memory()
        results = mem.recall(query, budget=effective_budget, namespace=self.namespace)

        # Track
        for r in results:
            if r.memory.id not in self.memories_recalled:
                self.memories_recalled.append(r.memory.id)

        # Estimate tokens consumed (rough: 1 token ≈ 4 chars)
        chars = sum(len(r.memory.content) for r in results)
        self.token_budget_used += chars // 4

        if use_cache:
            self._recall_cache[query] = results

        return results

    def recall_as_context(
        self, query: str, budget: int | None = None
    ) -> str:
        """Recall and format as injectable context string."""
        mem = self._get_memory()
        return mem.recall_as_context(
            query,
            budget=budget or (self.token_budget - self.token_budget_used),
            namespace=self.namespace,
        )

    def search(self, **kwargs: Any) -> list[Memory]:
        """Search with filters, scoped to this context's namespace."""
        kwargs.setdefault("namespace", self.namespace)
        mem = self._get_memory()
        return mem.search(**kwargs)

    def timeline(self, entity: str) -> list[Memory]:
        """Get chronological history for an entity, scoped to namespace."""
        mem = self._get_memory()
        return mem.timeline(entity, namespace=self.namespace)

    def forget(self, memory_id: str) -> bool:
        """Archive a memory. Requires the FORGET capability (admin-tier)."""
        self._require_permission(RolePermission.FORGET)
        mem = self._get_memory()
        return mem.forget(memory_id)

    # ------------------------------------------------------------------ #
    #  Scratchpad (inter-agent data passing)                               #
    # ------------------------------------------------------------------ #

    def set_scratchpad(self, key: str, value: Any) -> None:
        """Write to the shared scratchpad. Convention: use agent_id as key prefix."""
        self.scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Read from the shared scratchpad."""
        return self.scratchpad.get(key, default)

    # ------------------------------------------------------------------ #
    #  Governance                                                          #
    # ------------------------------------------------------------------ #

    def flag_for_review(self, reason: str) -> None:
        """Flag this context for human review."""
        self.requires_human_review = True
        self.compliance_tags.append(f"review:{reason}")

    def add_compliance_tag(self, tag: str) -> None:
        """Add a compliance/audit tag."""
        if tag not in self.compliance_tags:
            self.compliance_tags.append(tag)

    # ------------------------------------------------------------------ #
    #  Health & introspection                                              #
    # ------------------------------------------------------------------ #

    def health(self) -> dict[str, Any]:
        """Check memory backend health."""
        mem = self._get_memory()
        return mem.health()

    def session_summary(self) -> dict[str, Any]:
        """Summarize what happened in this session."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "agent_trail": self.agent_trail,
            "memories_written": len(self.memories_written),
            "memories_recalled": len(self.memories_recalled),
            "entities_discovered": self.entities_discovered,
            "token_budget_total": self.token_budget,
            "token_budget_used": self.token_budget_used,
            "token_budget_remaining": self.token_budget - self.token_budget_used,
            "requires_human_review": self.requires_human_review,
            "compliance_tags": self.compliance_tags,
            "scratchpad_keys": list(self.scratchpad.keys()),
        }

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the memory backend if we own it."""
        if self.memory and hasattr(self.memory, "close"):
            self.memory.close()

    def __enter__(self) -> AgentContext:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
