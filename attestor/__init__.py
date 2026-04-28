"""Attestor — Embedded memory for AI agents."""

from attestor.context import (
    ROLE_PERMISSIONS,
    AgentContext,
    AgentRole,
    RolePermission,
    Visibility,
)
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
    "RolePermission",
    "ROLE_PERMISSIONS",
    "Memory",
    "MemoryScope",
    "Project",
    "RetrievalResult",
    "Session",
    "User",
    "Visibility",
]
__version__ = "4.0.0"
