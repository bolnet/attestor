"""Attestor — Embedded memory for AI agents."""

from attestor.context import AgentContext, AgentRole, Visibility
from attestor.core import AgentMemory
from attestor.models import (
    Memory,
    MemoryScope,
    Project,
    RetrievalResult,
    Session,
    User,
)

__all__ = [
    "AgentMemory",
    "AgentContext",
    "AgentRole",
    "Memory",
    "MemoryScope",
    "Project",
    "RetrievalResult",
    "Session",
    "User",
    "Visibility",
]
__version__ = "4.0.0a4"
