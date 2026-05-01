"""Env-var / required-key resolution helpers for the YAML loader."""

from __future__ import annotations

import os
from typing import Any


def _resolve_env_password(node: Any, *, strict: bool) -> str:
    """Resolve a password from ``password`` (literal) or ``password_env``."""
    if isinstance(node, dict):
        if node.get("password"):
            return str(node["password"])
        env_name = node.get("password_env")
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
            if strict:
                raise SystemExit(
                    f"[attestor.config] required env {env_name!r} not set"
                )
            return ""  # placeholder for non-strict (tests / CI without secrets)
    if strict:
        raise SystemExit(
            f"[attestor.config] could not resolve password from {node!r}"
        )
    return ""


def _require(block: dict[str, Any], key: str, yaml_path: str) -> Any:
    """Pull a required key out of a YAML block, raising with the dotted
    path so the user knows exactly which key to add.

    Fallback constants were removed in favor of fail-loud behavior:
    every required key in ``configs/attestor.yaml`` must be present.
    """
    if key not in block:
        raise ValueError(
            f"{yaml_path} missing in configs/attestor.yaml — "
            "required since fallback constants were removed"
        )
    return block[key]
