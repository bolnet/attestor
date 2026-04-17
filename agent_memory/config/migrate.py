"""Convert legacy config.json to config.toml."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _to_toml(data: dict[str, Any], indent: int = 0) -> str:
    """Tiny TOML emitter — only handles the shapes memwright produces."""
    scalars: list[str] = []
    tables: list[str] = []
    for key, val in data.items():
        if isinstance(val, dict):
            tables.append(f"\n[{key}]\n" + _to_toml(val, indent + 1))
        elif isinstance(val, str):
            scalars.append(f'{key} = "{val}"')
        elif isinstance(val, bool):
            scalars.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            scalars.append(f"{key} = {val}")
        elif isinstance(val, list):
            inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in val)
            scalars.append(f"{key} = [{inner}]")
        else:
            scalars.append(f"{key} = {json.dumps(val)}")
    return "\n".join(scalars) + "".join(tables)


def migrate_json_to_toml(src: Path) -> Path:
    """Convert config.json to config.toml in the same directory.

    The original JSON is preserved as config.json.bak.
    Returns the path to the new TOML file.
    """
    if src.suffix != ".json":
        raise ValueError(f"Expected .json file, got {src}")
    with open(src) as f:
        data = json.load(f)

    dst = src.with_suffix(".toml")
    dst.write_text(_to_toml(data) + "\n")
    shutil.copy(src, src.with_suffix(".json.bak"))
    return dst
