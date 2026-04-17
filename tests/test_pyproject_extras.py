# tests/test_pyproject_extras.py
"""Lock the contract between optional extras and backend imports."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _extras() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


def test_docker_extra_exists_and_pins_docker_sdk() -> None:
    extras = _extras()
    assert "docker" in extras, "pyproject must declare a [docker] extra"
    deps = " ".join(extras["docker"])
    assert "docker" in deps, "[docker] extra must include the docker SDK"


def test_all_extra_includes_docker_sdk() -> None:
    extras = _extras()
    deps = " ".join(extras["all"])
    assert "docker" in deps, "[all] extra must include the docker SDK"
