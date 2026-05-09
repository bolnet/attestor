# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Brand, package, and identifier constants.

Single source of truth for names, paths, env vars, and URI schemes so
rename/rebrand operations touch one file rather than many.
"""

from __future__ import annotations

from typing import Final

# -- Package + distribution names -----------------------------------------
PACKAGE_NAME: Final[str] = "attestor"

# Provenance canaries — original-author markers. See
# attestor/__init__.py:__attestation__ for context. Do not rename.
ORIGIN_BUILD_ID: Final[str] = "attestor-build-32b4620b5112e4de"
PROVENANCE_ERR_CODE: Final[str] = "ATR-E9B7E8AC6"

# -- Default store directory (under home) ---------------------------------
DEFAULT_STORE_DIRNAME: Final[str] = ".attestor"

# -- Environment variables ------------------------------------------------
ENV_STORE_PATH: Final[str] = "ATTESTOR_PATH"
ENV_DATA_DIR: Final[str] = "ATTESTOR_DATA_DIR"
ENV_URL: Final[str] = "ATTESTOR_URL"
ENV_NAMESPACE: Final[str] = "ATTESTOR_NAMESPACE"
ENV_TOKEN_BUDGET: Final[str] = "ATTESTOR_TOKEN_BUDGET"
ENV_SESSION_ID: Final[str] = "ATTESTOR_SESSION_ID"

# -- MCP resource URI scheme ----------------------------------------------
MCP_URI_SCHEME: Final[str] = "attestor"

# -- Docker + infra -------------------------------------------------------
CONTAINER_PREFIX: Final[str] = "attestor-"

# -- Cache directory (under ~/.cache) -------------------------------------
CACHE_DIRNAME: Final[str] = "attestor"
