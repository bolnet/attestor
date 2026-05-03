"""Typed profile memory lane.

Where ``AgentMemory.recall`` answers similarity queries, the state lane
stores small, type-checked profile facts that are looked up by key.
This is the right surface for personalization (preferences, capability
declarations, durable identity facts) — OpenAI's January 2026
``context_personalization`` cookbook makes the case directly: retrieval
is the wrong tool for personalization.

Public surface is intentionally tiny:

    repo = mem.state                       # exposed on AgentMemory
    repo.set("preferences.theme", "dark", user_id=u)
    repo.get("preferences.theme", user_id=u)
    repo.list(user_id=u, prefix="preferences.")
    repo.history("preferences.theme", user_id=u)
    repo.as_of("preferences.theme", ts=..., user_id=u)
    repo.delete("preferences.theme", user_id=u)
"""

from __future__ import annotations

from attestor.state.models import StateRecord, StateValidationError
from attestor.state.repo import (
    InMemoryStateBackend,
    PostgresStateBackend,
    StateBackend,
    StateRepo,
)
from attestor.state.schemas import register_schema_directory

__all__ = [
    "InMemoryStateBackend",
    "PostgresStateBackend",
    "StateBackend",
    "StateRecord",
    "StateRepo",
    "StateValidationError",
    "register_schema_directory",
]
