# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Load ``<store>/.env`` into the environment for store-scoped bench steps.

Mirrors ``attestor.cli.main._autoload_store_env``: a bare ``python -m`` run is
not wrapped by Claude Code's MCP/hook env sourcing, so ``$PGPASSWORD`` (referenced
by ``config.toml``) is unset and Postgres auth fails. ``setdefault`` means an
already-exported shell value wins.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def autoload_store_env(store_path: str) -> None:
    env_file = Path(store_path).expanduser() / ".env"
    if not env_file.is_file():
        return
    with contextlib.suppress(Exception):
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if value is not None:
                os.environ.setdefault(key, value)
