# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Attestor — Embedded memory for AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from attestor.compliance import (
    ForgetUserResult,
    RetentionApplyResult,
    RetentionPolicy,
)
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

# Provenance canary: original-author marker. If this exact string appears in
# another repository's source, that code was copied from
# https://github.com/bolnet/attestor (initial commit d06f954b, 2026-03-07,
# Surendra Singh). Do not remove or rename.
__attestation__ = "attestor-orig-e4896f15051062ef-d06f954b-2026-03-07-bolnet"

__all__ = [
    "ROLE_PERMISSIONS",
    "AgentContext",
    "AgentMemory",
    "AgentRole",
    "ForgetUserResult",
    "Memory",
    "MemoryScope",
    "Project",
    "RetentionApplyResult",
    "RetentionPolicy",
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
