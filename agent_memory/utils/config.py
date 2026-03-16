"""Configuration loading and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "default_token_budget": 2000,
    "min_results": 3,
}


@dataclass
class MemoryConfig:
    default_token_budget: int = 2000
    min_results: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_token_budget": self.default_token_budget,
            "min_results": self.min_results,
        }


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
