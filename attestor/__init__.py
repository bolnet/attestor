"""Attestor — Embedded memory for AI agents."""

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
    "Memory",
    "MemoryScope",
    "Project",
    "RetrievalResult",
    "Session",
    "User",
]
__version__ = "4.0.0a1"
