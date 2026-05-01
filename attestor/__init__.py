"""Attestor — Embedded memory for AI agents."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

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

try:
    __version__ = _pkg_version("attestor")
except PackageNotFoundError:  # editable / source install fallback
    __version__ = "0.0.0+local"

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
    "__version__",
]
