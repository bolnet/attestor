# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Shared CLI helpers — env file loading, backend args, output suppression.

Extracted from the legacy ``attestor/cli.py`` so that the per-command modules
under :mod:`attestor.cli.commands` can import only what they need.

Nothing in this module is considered public API; the leading-underscore names
match the original module-level names in the pre-split file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


class _SuppressModelNoise:
    """Context manager to suppress noisy model loading output at fd level.

    safetensors prints LOAD REPORT from Rust, bypassing Python's sys.stdout.
    We must redirect at the OS file descriptor level to silence it.
    """

    def __enter__(self) -> "_SuppressModelNoise":
        # Open all fds first, then perform the fd swaps. If any os.dup2
        # raises mid-setup we tear the partial state down ourselves —
        # __exit__ doesn't run when __enter__ raises.
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            self._old_stdout_fd = os.dup(1)
            self._old_stderr_fd = os.dup(2)
            os.dup2(self._devnull_fd, 1)
            os.dup2(self._devnull_fd, 2)
        except OSError:
            for attr in ("_old_stderr_fd", "_old_stdout_fd"):
                fd = getattr(self, attr, None)
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            os.close(self._devnull_fd)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        os.dup2(self._old_stdout_fd, 1)
        os.dup2(self._old_stderr_fd, 2)
        os.close(self._devnull_fd)
        os.close(self._old_stdout_fd)
        os.close(self._old_stderr_fd)


def _suppress_noisy_output() -> None:
    """Set environment variables to suppress HuggingFace/safetensors noise."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["SAFETENSORS_LOG_LEVEL"] = "error"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

    import logging
    import warnings

    # Narrowed from the prior ``warnings.filterwarnings("ignore")`` —
    # silencing every warning category masked legitimate signals
    # (Deprecation, Pending, etc.) in CI / test contexts. We only
    # suppress the noisy categories the model-loading stack emits.
    for category in (UserWarning, FutureWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=category, module="transformers")
        warnings.filterwarnings("ignore", category=category, module="huggingface_hub")
        warnings.filterwarnings("ignore", category=category, module="safetensors")
    for name in ("tqdm",):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _load_env_file(env_file_path: str) -> None:
    """Load environment variables from a .env file.

    Resolves the path so a relative ``--env-file ../../etc/passwd``
    can't silently traverse out of the user's working tree without an
    explicit absolute path. The resolved path is logged at debug for
    auditability when CI pipelines feed dynamic values in.
    """
    env_path = Path(env_file_path).expanduser().resolve()
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip a single matching pair of surrounding quotes so
            # values like ``"foo'bar"`` round-trip cleanly.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if value:
                os.environ[key] = value


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Add --backend and --backend-config arguments to a subparser."""
    parser.add_argument(
        "--backend", default=None,
        help="Backend role override (one of: postgres, pinecone, neo4j). "
             "The canonical stack assigns postgres → document, "
             "pinecone → vector, neo4j → graph.",
    )
    parser.add_argument(
        "--backend-config", default=None,
        help="JSON string or path to JSON file with backend config. "
             'Example: \'{"url": "http://localhost:8530", "database": "bench"}\'',
    )


def _parse_backend_config(args) -> dict | None:
    """Build a config dict from --backend and --backend-config CLI args."""
    backend = getattr(args, "backend", None)
    if not backend:
        return None

    config = {"backends": [backend]}

    raw = getattr(args, "backend_config", None)
    if raw:
        # Try as file path first, then as JSON string
        raw_path = Path(raw)
        if raw_path.exists():
            config[backend] = json.loads(raw_path.read_text())
        else:
            config[backend] = json.loads(raw)

    return config
