"""ConversationIngest — round-level orchestration (Phase 3.5).

Wires the foundation pieces:

  EpisodeRepo.write_round   — verbatim audit
  extract_user_facts        — user-side facts (with source_span)
  extract_agent_facts       — assistant-side facts
  resolve_conflicts         — ADD/UPDATE/INVALIDATE/NOOP per fact
  apply_decisions           — write through supersession

Public surface:

  IngestConfig    — knobs (model, recall_k, etc.)
  RoundResult     — episode_id + decisions + applied
  ConversationIngest.ingest_round(user_turn, assistant_turn, *, ctx)

Designed so AgentMemory.ingest_round() is a thin wrapper that resolves
identity, then delegates here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from attestor.conversation.apply import AppliedDecision, apply_decisions
from attestor.conversation.episodes import Episode, EpisodeRepo
from attestor.conversation.turns import ConversationTurn
from attestor.extraction.conflict_resolver import Decision, resolve_conflicts
from attestor.extraction.round_extractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ExtractedFact,
    extract_agent_facts,
    extract_user_facts,
)
from attestor.models import Memory

logger = logging.getLogger("attestor.conversation.ingest")


@dataclass(frozen=True)
class IngestConfig:
    """Tuning knobs for ingest_round.

    extract_user / extract_agent let callers turn off one side cheaply
    (e.g., audit-only assistants where user facts aren't useful)."""

    extract_user: bool = True
    extract_agent: bool = True
    resolve_conflicts: bool = True
    recall_k: int = 5  # how many existing memories to consider for conflict
    extraction_model: str = DEFAULT_MODEL
    extraction_max_tokens: int = DEFAULT_MAX_TOKENS
    resolver_model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class RoundResult:
    """Output of one ingest_round call."""

    episode: Episode
    user_facts: List[ExtractedFact]
    agent_facts: List[ExtractedFact]
    decisions: List[Decision]
    applied: List[AppliedDecision]

    @property
    def episode_id(self) -> str:
        return self.episode.id

    @property
    def written_memory_ids(self) -> List[str]:
        return [a.memory_id for a in self.applied if a.memory_id and a.operation in {"ADD", "UPDATE", "INVALIDATE"}]


class ConversationIngest:
    """Per-round extraction → conflict resolution → write.

    Keeps a reference to the AgentMemory (mem) but doesn't construct it.
    Tests use a duck-typed Mem stub; production code passes a real
    AgentMemory.
    """

    def __init__(
        self,
        mem: Any,                              # AgentMemory
        config: Optional[IngestConfig] = None,
        *,
        extraction_client: Optional[Any] = None,
        resolver_client: Optional[Any] = None,
    ) -> None:
        self._mem = mem
        self.config = config or IngestConfig()
        self._extraction_client = extraction_client
        self._resolver_client = resolver_client

    # ── Public API ──────────────────────────────────────────────────────

    def ingest_round(
        self,
        user_turn: ConversationTurn,
        assistant_turn: ConversationTurn,
        *,
        user_id: str,
        session_id: str,
        project_id: Optional[str] = None,
        scope: str = "user",
        agent_id: Optional[str] = None,
        recent_context: str = "(none)",
    ) -> RoundResult:
        """End-to-end ingest of one (user, assistant) round.

        Steps:
          1. Verbatim episode write (audit log)
          2. Speaker-locked extraction (user + agent in sequence)
          3. Conflict resolution against top-k similar existing memories
          4. Apply decisions through supersession path
        """
        episode = self._write_episode(
            user_turn=user_turn, assistant_turn=assistant_turn,
            user_id=user_id, session_id=session_id,
            project_id=project_id, agent_id=agent_id,
        )

        user_facts = self._extract(user_turn, recent_context, kind="user")
        agent_facts = self._extract(assistant_turn, recent_context, kind="agent")
        all_facts: List[ExtractedFact] = user_facts + agent_facts

        decisions = self._resolve(
            new_facts=all_facts,
            evidence_episode_id=episode.id,
            user_id=user_id,
        )

        applied = apply_decisions(
            decisions,
            mem=self._mem,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            scope=scope,
            extraction_model=self.config.extraction_model,
            parent_agent_id=agent_id,
        )
        return RoundResult(
            episode=episode,
            user_facts=user_facts,
            agent_facts=agent_facts,
            decisions=decisions,
            applied=applied,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _write_episode(
        self,
        *,
        user_turn: ConversationTurn,
        assistant_turn: ConversationTurn,
        user_id: str,
        session_id: str,
        project_id: Optional[str],
        agent_id: Optional[str],
    ) -> Episode:
        repo = EpisodeRepo(self._mem._store._conn)
        return repo.write_round(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            user_turn=user_turn,
            assistant_turn=assistant_turn,
            agent_id=agent_id,
        )

    def _extract(
        self, turn: ConversationTurn, recent_context: str, *, kind: str,
    ) -> List[ExtractedFact]:
        if kind == "user":
            if not self.config.extract_user:
                return []
            return extract_user_facts(
                turn, recent_context=recent_context,
                model=self.config.extraction_model,
                max_tokens=self.config.extraction_max_tokens,
                client=self._extraction_client,
            )
        if kind == "agent":
            if not self.config.extract_agent:
                return []
            return extract_agent_facts(
                turn, recent_context=recent_context,
                model=self.config.extraction_model,
                max_tokens=self.config.extraction_max_tokens,
                client=self._extraction_client,
            )
        raise ValueError(f"unknown kind: {kind!r}")

    def _resolve(
        self,
        *,
        new_facts: List[ExtractedFact],
        evidence_episode_id: str,
        user_id: str,
    ) -> List[Decision]:
        if not self.config.resolve_conflicts or not new_facts:
            return [
                Decision(
                    operation="ADD", new_fact=f, existing_id=None,
                    rationale="conflict resolution disabled",
                    evidence_episode_id=evidence_episode_id,
                )
                for f in new_facts
            ]
        existing = self._retrieve_similar(new_facts, user_id=user_id)
        return resolve_conflicts(
            new_facts=new_facts,
            existing=existing,
            evidence_episode_id=evidence_episode_id,
            model=self.config.resolver_model,
            max_tokens=self.config.extraction_max_tokens,
            client=self._resolver_client,
        )

    def _retrieve_similar(
        self, new_facts: List[ExtractedFact], *, user_id: str,
    ) -> List[Memory]:
        """Top-k existing memories most similar to the new facts.

        Two-tier lookup:
          1. Try ``mem.recall`` (full retrieval pipeline — vector/tag/graph).
          2. Fall back to a direct doc-store query by category+entity from
             the new facts. This keeps the resolver useful even when the
             vector store is unavailable (managed embedder offline, etc.).
        """
        if not new_facts or self._mem is None:
            return []

        # Tier 1: full recall pipeline
        try:
            query = " | ".join(f.text for f in new_facts)
            results = self._mem.recall(
                query, budget=2000, user_id=user_id,
            )
            out = self._dedupe_top_k([r.memory for r in results])
            if out:
                return out
        except Exception as e:
            logger.debug("recall lookup failed; falling back to doc-store: %s", e)

        # Tier 2: doc-store fallback by (category, entity)
        return self._fallback_doc_lookup(new_facts)

    def _fallback_doc_lookup(
        self, new_facts: List[ExtractedFact],
    ) -> List[Memory]:
        """Query the document store directly for memories that share a
        category or entity with any new fact. Best-effort; returns [] on
        failure or if the store doesn't support list_memories."""
        store = self._mem._store
        if not hasattr(store, "list_memories"):
            return []
        seen: set = set()
        out: List[Memory] = []
        for fact in new_facts:
            try:
                # Prefer entity-scoped lookup when available
                if fact.entity:
                    rows = store.list_memories(
                        category=fact.category, entity=fact.entity,
                        status="active", limit=self.config.recall_k,
                    )
                else:
                    rows = store.list_memories(
                        category=fact.category,
                        status="active", limit=self.config.recall_k,
                    )
            except Exception as e:
                logger.debug("doc-store fallback lookup failed: %s", e)
                continue
            for m in rows:
                if m.id in seen:
                    continue
                seen.add(m.id)
                out.append(m)
                if len(out) >= self.config.recall_k:
                    return out
        return out

    @staticmethod
    def _dedupe_top_k(memories: List[Memory], limit: int = 5) -> List[Memory]:
        seen: set = set()
        out: List[Memory] = []
        for m in memories:
            if m.id in seen:
                continue
            seen.add(m.id)
            out.append(m)
            if len(out) >= limit:
                break
        return out
