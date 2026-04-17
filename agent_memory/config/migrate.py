"""Convert legacy config.json to config.toml."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _escape_toml_string(s: str) -> str:
    """Escape backslashes and double-quotes for TOML basic strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _to_toml(data: dict[str, Any], prefix: str = "") -> str:
    """Tiny TOML emitter — only handles the shapes memwright produces.

    ``prefix`` tracks the dotted table path for nested sections, e.g. "a.b".
    ``None`` values are silently skipped (TOML has no null concept).
    """
    scalars: list[str] = []
    tables: list[str] = []
    for key, val in data.items():
        if val is None:
            # TOML has no null — skip silently so legacy config.json files
            # with unset optional fields migrate cleanly.
            continue
        if isinstance(val, dict):
            section = f"{prefix}.{key}" if prefix else key
            tables.append(f"\n[{section}]\n" + _to_toml(val, prefix=section))
        elif isinstance(val, str):
            scalars.append(f'{key} = "{_escape_toml_string(val)}"')
        elif isinstance(val, bool):
            scalars.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            scalars.append(f"{key} = {val}")
        elif isinstance(val, list):
            inner = ", ".join(
                f'"{_escape_toml_string(x)}"' if isinstance(x, str) else str(x)
                for x in val
            )
            scalars.append(f"{key} = [{inner}]")
        # Any other type (e.g. unexpected objects) is skipped gracefully.
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
