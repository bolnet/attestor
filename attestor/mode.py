"""Attestor v4 operating modes — SOLO / HOSTED / SHARED.

Per defaults.md §1, the mode determines which defaults apply:

  SOLO    — single-user (laptop / dev / MCP). No auth. Singleton user
            ``external_id="local"`` auto-created on first call.
  HOSTED  — multi-user SaaS. Auth required, no anonymous calls allowed.
            User auto-provisioned on first JWT (handled by middleware).
  SHARED  — self-hosted multi-user. Auth required, no per-user quotas.

Detection is environment-driven so the library never asks the caller
"what mode are you in" at every call site:

  1. ``ATTESTOR_MODE`` env var — explicit override (solo / hosted / shared).
  2. ``ATTESTOR_AUTH_PROVIDER`` set (auth0 / clerk / cognito / oidc) → HOSTED.
  3. ``ATTESTOR_REQUIRE_AUTH=true`` → SHARED.
  4. Else → SOLO. The default for ``pip install attestor``.

The mode is computed once per ``AgentMemory`` instance and cached. Tests
can pass ``mode=`` explicitly to bypass detection.
"""

from __future__ import annotations

import os
from enum import Enum


class AttestorMode(str, Enum):
    SOLO = "solo"
    HOSTED = "hosted"
    SHARED = "shared"


_TRUE = {"1", "true", "True", "yes", "YES"}


def detect_mode(env: dict | None = None) -> AttestorMode:
    """Determine the operating mode from environment variables.

    Args:
        env: Optional override for ``os.environ``. Tests inject a stub
            dict; production callers pass nothing.

    Returns:
        The detected ``AttestorMode``. Defaults to ``SOLO``.
    """
    src = os.environ if env is None else env

    explicit = src.get("ATTESTOR_MODE")
    if explicit:
        try:
            return AttestorMode(explicit.lower())
        except ValueError:
            # Invalid override — fall through to SOLO rather than crash.
            # An invalid env var shouldn't take down a fresh install.
            pass

    if src.get("ATTESTOR_AUTH_PROVIDER"):
        return AttestorMode.HOSTED

    if src.get("ATTESTOR_REQUIRE_AUTH") in _TRUE:
        return AttestorMode.SHARED

    return AttestorMode.SOLO


# Stable, well-known external_id for the SOLO singleton user.
# Don't change this — every existing SOLO installation finds their data
# by this string.
SOLO_USER_EXTERNAL_ID = "local"
SOLO_USER_DISPLAY_NAME = "Local User"
