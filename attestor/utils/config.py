"""Configuration loading and defaults."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib as _toml_reader
else:  # pragma: no cover - 3.10 fallback
    import tomli as _toml_reader  # type: ignore[no-redef]


DEFAULT_CONFIG = {
    "default_token_budget": 16000,
    "min_results": 3,
}


_DEFAULT_BACKENDS = ["postgres", "neo4j"]


@dataclass
class MemoryConfig:
    default_token_budget: int = 16000
    min_results: int = 3
    backends: list[str] = field(default_factory=lambda: list(_DEFAULT_BACKENDS))
    backend_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Retrieval tuning
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    fusion_mode: str = "rrf"  # "rrf" or "graph_blend"
    confidence_gate: float = 0.0
    confidence_decay_rate: float = 0.001
    confidence_boost_rate: float = 0.03

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryConfig:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        backends = data.get("backends", list(_DEFAULT_BACKENDS))
        backend_configs: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if key not in known_fields and isinstance(value, dict):
                backend_configs[key] = value
        filtered = {k: v for k, v in data.items() if k in known_fields}
        filtered["backends"] = backends
        filtered["backend_configs"] = {**backend_configs, **filtered.get("backend_configs", {})}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
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
    """Load config from `config.toml` (preferred) or legacy `config.json`."""
    toml_file = path / "config.toml"
    if toml_file.exists():
        with open(toml_file, "rb") as f:
            data = _toml_reader.load(f)
        return MemoryConfig.from_dict(data)

    json_file = path / "config.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        return MemoryConfig.from_dict(data)

    return MemoryConfig()


def save_config(path: Path, config: MemoryConfig) -> None:
    """Save config atomically to a JSON file.

    AgentMemory.__init__ calls this on every construction. Two processes
    constructing AgentMemory concurrently would otherwise truncate-then-
    write the same file, leaving a half-written JSON visible to any
    third reader. We write to a sibling temp file and ``os.replace`` it
    in — that rename is atomic on POSIX, so readers always see either
    the old complete file or the new complete file, never a torn write.
    """
    config_file = path / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".config.", suffix=".json.tmp", dir=str(config_file.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        os.replace(tmp_path, config_file)
    except Exception:
        # Best-effort cleanup of the temp file if rename never happened.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
