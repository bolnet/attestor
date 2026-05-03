"""Attestor — Embedded memory for AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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
from attestor.state import (
    StateRecord,
    StateRepo,
    StateValidationError,
)

try:
    __version__ = _pkg_version("attestor")
except PackageNotFoundError:  # editable / source install fallback
    __version__ = "0.0.0+local"

__all__ = [
    "ROLE_PERMISSIONS",
    "AgentContext",
    "AgentMemory",
    "AgentRole",
    "Memory",
    "MemoryScope",
    "Project",
    "RetrievalResult",
    "RolePermission",
    "Session",
    "StateRecord",
    "StateRepo",
    "StateValidationError",
    "User",
    "Visibility",
    "__version__",
]
