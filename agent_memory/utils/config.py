"""Configuration loading and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_CONFIG = {
    "default_token_budget": 16000,
    "min_results": 3,
}


_DEFAULT_BACKENDS = ["sqlite", "chroma", "networkx"]


@dataclass
class MemoryConfig:
    default_token_budget: int = 16000
    min_results: int = 3
    backends: List[str] = field(default_factory=lambda: list(_DEFAULT_BACKENDS))
    backend_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Retrieval tuning
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    fusion_mode: str = "rrf"  # "rrf" or "graph_blend"
    confidence_gate: float = 0.0
    confidence_decay_rate: float = 0.001
    confidence_boost_rate: float = 0.03

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        backends = data.get("backends", list(_DEFAULT_BACKENDS))
        backend_configs: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if key not in known_fields and isinstance(value, dict):
                backend_configs[key] = value
        filtered = {k: v for k, v in data.items() if k in known_fields}
        filtered["backends"] = backends
        filtered["backend_configs"] = {**backend_configs, **filtered.get("backend_configs", {})}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "default_token_budget": self.default_token_budget,
            "min_results": self.min_results,
            "backends": self.backends,
            "enable_mmr": self.enable_mmr,
            "mmr_lambda": self.mmr_lambda,
            "fusion_mode": self.fusion_mode,
            "confidence_gate": self.confidence_gate,
            "confidence_decay_rate": self.confidence_decay_rate,
            "confidence_boost_rate": self.confidence_boost_rate,
        }
        for name, cfg in self.backend_configs.items():
            result[name] = cfg
        return result


def load_config(path: Path) -> MemoryConfig:
    """Load config via the new layered loader, returning a legacy MemoryConfig.

    Kept for backward compat; new code should use agent_memory.config.load_settings.
    """
    from agent_memory.config.loader import load_settings

    settings = load_settings(path)
    # Use exclude_unset so user-supplied values that happen to match schema
    # defaults (e.g., arangodb.mode == "local") still round-trip into
    # backend_configs. exclude_defaults would strip them, breaking legacy
    # save_config/load_config round-trip behavior.
    data = settings.model_dump(exclude_unset=True)
    data.pop("profiles", None)
    return MemoryConfig.from_dict(data)


def _format_toml_value(value: Any) -> str:
    """Serialize a Python scalar/list to a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # Use JSON encoding for safe escaping; TOML basic strings share syntax.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    if value is None:
        # TOML has no null; emit empty string to avoid losing the key silently.
        return '""'
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _emit_toml(data: Mapping[str, Any]) -> str:
    """Minimal TOML emitter covering MemoryConfig.to_dict()'s shape.

    Scalars/lists first, then nested tables (one per backend_configs entry).
    Not a general-purpose emitter; avoids adding a tomli_w dependency just
    for round-trip save/load.
    """
    scalars: List[str] = []
    tables: List[str] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines = [f"[{key}]"]
            for sub_key, sub_value in value.items():
                lines.append(f"{sub_key} = {_format_toml_value(sub_value)}")
            tables.append("\n".join(lines))
        else:
            scalars.append(f"{key} = {_format_toml_value(value)}")
    sections = []
    if scalars:
        sections.append("\n".join(scalars))
    sections.extend(tables)
    return "\n\n".join(sections) + "\n"


def save_config(path: Path, config: MemoryConfig) -> None:
    """Persist config. If config.toml exists, write TOML; else write JSON.

    This keeps the user's chosen format authoritative -- avoids a silent
    divergence where save writes JSON but load_config prefers TOML, which
    would cause in-memory mutations to be dropped on the next load.
    """
    data = config.to_dict()
    toml_file = path / "config.toml"
    if toml_file.exists():
        toml_file.write_text(_emit_toml(data))
        return
    config_file = path / "config.json"
    with open(config_file, "w") as f:
        json.dump(data, f, indent=2)
