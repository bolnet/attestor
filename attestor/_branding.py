"""Brand, package, and identifier constants.

Single source of truth for names, paths, env vars, and URI schemes so
rename/rebrand operations touch one file rather than many.

Phase 0 adds these constants with both the new canonical names and the
legacy names held alongside them. Later phases switch call-sites to the
canonical names one at a time.
"""

from __future__ import annotations

from typing import Final

# -- Package + distribution names -----------------------------------------
PACKAGE_NAME: Final[str] = "attestor"
LEGACY_PACKAGE_NAME: Final[str] = "memwright"

# -- Default store directory (under home) ---------------------------------
DEFAULT_STORE_DIRNAME: Final[str] = ".attestor"
LEGACY_STORE_DIRNAME: Final[str] = ".memwright"

# -- Environment variables ------------------------------------------------
ENV_STORE_PATH: Final[str] = "ATTESTOR_PATH"
LEGACY_ENV_STORE_PATH: Final[str] = "MEMWRIGHT_PATH"

ENV_DATA_DIR: Final[str] = "ATTESTOR_DATA_DIR"
LEGACY_ENV_DATA_DIR: Final[str] = "MEMWRIGHT_DATA_DIR"

ENV_URL: Final[str] = "ATTESTOR_URL"
LEGACY_ENV_URL: Final[str] = "MEMWRIGHT_URL"

ENV_NAMESPACE: Final[str] = "ATTESTOR_NAMESPACE"
LEGACY_ENV_NAMESPACE: Final[str] = "MEMWRIGHT_NAMESPACE"

ENV_TOKEN_BUDGET: Final[str] = "ATTESTOR_TOKEN_BUDGET"
LEGACY_ENV_TOKEN_BUDGET: Final[str] = "MEMWRIGHT_TOKEN_BUDGET"

ENV_SESSION_ID: Final[str] = "ATTESTOR_SESSION_ID"
LEGACY_ENV_SESSION_ID: Final[str] = "MEMWRIGHT_SESSION_ID"

# -- MCP resource URI scheme ----------------------------------------------
MCP_URI_SCHEME: Final[str] = "attestor"
LEGACY_MCP_URI_SCHEME: Final[str] = "memwright"

# -- Docker + infra -------------------------------------------------------
CONTAINER_PREFIX: Final[str] = "attestor-"
LEGACY_CONTAINER_PREFIX: Final[str] = "memwright-"

# -- Cache directory (under ~/.cache) -------------------------------------
CACHE_DIRNAME: Final[str] = "attestor"
LEGACY_CACHE_DIRNAME: Final[str] = "memwright"
