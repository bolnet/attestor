"""JSON-Schema bundles for the state lane.

Each ``.json`` file here is addressable by its filename stem (e.g.
``user_preferences_v1.json`` is referenced as ``schema="user_preferences_v1"``
on ``StateRepo.set``). Resolution is filesystem-based so users can ship
their own schemas in a sibling package and pass the directory to
``register_schema_directory`` -- no need to monkey-patch a registry.
"""

from __future__ import annotations

import json
from pathlib import Path

from attestor.state.models import StateValidationError

_BUILTIN_DIR = Path(__file__).resolve().parent
_USER_DIRS: list[Path] = []


def register_schema_directory(directory: str | Path) -> None:
    """Add an additional directory to the schema search path.

    Schemas in directories registered later take precedence over the
    built-ins, so callers can override the bundled definitions without
    forking the package.
    """
    p = Path(directory).resolve()
    if not p.is_dir():
        raise StateValidationError(f"schema directory does not exist: {p}")
    if p not in _USER_DIRS:
        _USER_DIRS.append(p)


def _candidate_paths(schema_id: str) -> list[Path]:
    # User-registered dirs take precedence over the built-ins.
    return [d / f"{schema_id}.json" for d in (*reversed(_USER_DIRS), _BUILTIN_DIR)]


def load_schema(schema_id: str) -> dict:
    """Load the JSON-Schema document for ``schema_id``.

    Raises:
        StateValidationError: when the schema cannot be found in any
            registered directory.
    """
    for path in _candidate_paths(schema_id):
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise StateValidationError(
                    f"schema '{schema_id}' is not valid JSON: {e}"
                ) from e
    raise StateValidationError(
        f"unknown schema '{schema_id}'. searched: "
        + ", ".join(str(p) for p in _candidate_paths(schema_id))
    )


def validate(value: object, schema_id: str) -> None:
    """Validate ``value`` against ``schema_id``.

    Raises:
        StateValidationError: when the schema is unknown, or when the
            value fails validation. The exception message names the
            schema_id and includes the validator's reason.
    """
    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - jsonschema is in deps
        raise StateValidationError(
            "the `jsonschema` package is required for state schema validation"
        ) from e

    schema = load_schema(schema_id)
    try:
        jsonschema.validate(instance=value, schema=schema)
    except jsonschema.ValidationError as e:
        raise StateValidationError(
            f"value failed schema '{schema_id}': {e.message}"
        ) from e


__all__ = ["load_schema", "register_schema_directory", "validate"]
