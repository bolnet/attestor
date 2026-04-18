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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from attestor.models import Memory, RetrievalResult


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

    # -- Memory access --
    # Either an AgentMemory instance (embedded) or an HTTP client (distributed).
    # Typed as Any to avoid circular import; runtime checks enforce the interface.
    memory: Any = None               # AgentMemory | MemoryClient
    memory_url: Optional[str] = None  # HTTP endpoint for distributed mode

    # -- Token budget --
    token_budget: int = 20000         # Total budget for this session
    token_budget_used: int = 0        # Consumed so far
    max_writes_per_agent: int = 100   # Write quota per sub-agent

    # -- Agent tracking --
    current_agent: str = ""
    agent_trail: List[str] = field(default_factory=list)
    parent_agent_id: Optional[str] = None

    # -- Accumulated state: memories written this session --
    memories_written: List[str] = field(default_factory=list)      # memory IDs
    memories_recalled: List[str] = field(default_factory=list)     # memory IDs
    entities_discovered: List[str] = field(default_factory=list)   # entity names

    # -- Recall cache (avoid redundant queries within a session) --
    _recall_cache: Dict[str, List[RetrievalResult]] = field(
        default_factory=dict, repr=False
    )

    # -- Scratchpad: structured data passed between agents --
    # Like PlannerContext's snapshot/analysis_results/recommendations,
    # but generic. Each agent writes to its own key.
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    # -- Governance --
    visibility: Visibility = Visibility.PUBLIC
    compliance_tags: List[str] = field(default_factory=list)
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
        read_only: bool = False,
        token_budget: Optional[int] = None,
    ) -> AgentContext:
        """Create a child context for a sub-agent.

        Preserves shared state (memory, session, scratchpad) but sets
        new identity and appends to the agent trail. Immutable -- returns
        a new context, never mutates the original.
        """
        trail = [*self.agent_trail, agent_id]
        return AgentContext(
            agent_id=agent_id,
            session_id=self.session_id,
            namespace=self.namespace,
            role=role,
            memory=self.memory,
            memory_url=self.memory_url,
            token_budget=token_budget or self.token_budget,
            token_budget_used=self.token_budget_used,
            max_writes_per_agent=self.max_writes_per_agent,
            current_agent=agent_id,
            agent_trail=trail,
            parent_agent_id=self.agent_id,
            memories_written=self.memories_written,
            memories_recalled=self.memories_recalled,
            entities_discovered=self.entities_discovered,
            _recall_cache=self._recall_cache,
            scratchpad=self.scratchpad,
            visibility=self.visibility,
            compliance_tags=self.compliance_tags,
            requires_human_review=self.requires_human_review,
            read_only=read_only,
        )

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

    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        entity: Optional[str] = None,
        event_date: Optional[str] = None,
        confidence: float = 1.0,
        visibility: Optional[Visibility] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory with full provenance tracking.

        Attaches agent_id, session_id, namespace, and visibility to metadata.
        Enforces write quota and read_only constraints.
        """
        if self.read_only:
            raise PermissionError(
                f"Agent '{self.agent_id}' is read-only in this context"
            )

        writes_by_me = sum(
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
        budget: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[RetrievalResult]:
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
        self, query: str, budget: Optional[int] = None
    ) -> str:
        """Recall and format as injectable context string."""
        mem = self._get_memory()
        return mem.recall_as_context(
            query,
            budget=budget or (self.token_budget - self.token_budget_used),
            namespace=self.namespace,
        )

    def search(self, **kwargs: Any) -> List[Memory]:
        """Search with filters, scoped to this context's namespace."""
        kwargs.setdefault("namespace", self.namespace)
        mem = self._get_memory()
        return mem.search(**kwargs)

    def timeline(self, entity: str) -> List[Memory]:
        """Get chronological history for an entity, scoped to namespace."""
        mem = self._get_memory()
        return mem.timeline(entity, namespace=self.namespace)

    def forget(self, memory_id: str) -> bool:
        """Archive a memory. Requires write access."""
        if self.read_only:
            raise PermissionError(
                f"Agent '{self.agent_id}' is read-only in this context"
            )
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

    def health(self) -> Dict[str, Any]:
        """Check memory backend health."""
        mem = self._get_memory()
        return mem.health()

    def session_summary(self) -> Dict[str, Any]:
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
