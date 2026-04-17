"""Layered config loader: defaults → file → env vars → CLI overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent_memory.config.schema import MemwrightSettings


_ENV_PREFIX = "MEMWRIGHT_"


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_file(store_path: Path) -> Dict[str, Any]:
    toml_file = store_path / "config.toml"
    json_file = store_path / "config.json"
    if toml_file.exists():
        return _read_toml(toml_file)
    if json_file.exists():
        return _read_json(json_file)
    return {}


def _coerce_env_value(raw: str) -> Any:
    """Best-effort coerce env string to int/float/bool/JSON; fall back to string."""
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _env_overrides() -> Dict[str, Any]:
    """Read MEMWRIGHT_* env vars into a flat dict (lowercased keys)."""
    overrides: Dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        name = key[len(_ENV_PREFIX):].lower()
        overrides[name] = _coerce_env_value(val)
    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    store_path: Path,
    *,
    profile: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> MemwrightSettings:
    """Resolve config across all layers and return a validated MemwrightSettings.

    Layer order (last wins):
        1. Pydantic defaults
        2. File (config.toml or config.json)
        3. Selected profile section
        4. MEMWRIGHT_* env vars
        5. cli_overrides
    """
    file_data = _load_file(store_path)

    profiles = file_data.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown profile {profile!r}; available: {sorted(profiles)}")
        file_data = _deep_merge(file_data, profiles[profile])

    env_data = _env_overrides()
    merged = _deep_merge(file_data, env_data)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return MemwrightSettings(**merged)
