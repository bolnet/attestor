"""Configuration loading and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


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
    """Load config from a JSON file, falling back to defaults."""
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
        return MemoryConfig.from_dict(data)
    return MemoryConfig()


def save_config(path: Path, config: MemoryConfig) -> None:
    """Save config to a JSON file."""
    config_file = path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
