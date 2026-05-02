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


def test_canonical_stack_extras_present() -> None:
    """The canonical PG+Pinecone+Neo4j stack each get a backend-specific extra
    so users can install only what they need."""
    extras = _extras()
    for name in ("postgres", "pinecone", "neo4j"):
        assert name in extras, f"pyproject must declare a [{name}] extra"


def test_postgres_extra_pins_psycopg2() -> None:
    extras = _extras()
    deps = " ".join(extras["postgres"])
    assert "psycopg2" in deps, "[postgres] extra must include psycopg2"


def test_pinecone_extra_pins_pinecone_client() -> None:
    extras = _extras()
    deps = " ".join(extras["pinecone"])
    assert "pinecone" in deps, "[pinecone] extra must include the pinecone client"


def test_neo4j_extra_pins_neo4j_driver() -> None:
    extras = _extras()
    deps = " ".join(extras["neo4j"])
    assert "neo4j" in deps, "[neo4j] extra must include the neo4j driver"


def test_all_extra_covers_canonical_stack() -> None:
    """The umbrella `all` extra must install every backend in the canonical
    stack so a single `pip install attestor[all]` boots a working node."""
    extras = _extras()
    deps = " ".join(extras["all"])
    for marker in ("psycopg2", "pinecone", "neo4j"):
        assert marker in deps, f"[all] extra must include {marker}"


def test_alt_backend_extras_removed() -> None:
    """The 2026-05-02 single-stack policy deletes alt-backend extras from
    pyproject. Re-introducing one without updating CLAUDE.md / registry would
    silently bring back code paths that no longer exist."""
    extras = _extras()
    for dropped in ("arangodb", "aws", "azure", "gcp", "docker"):
        assert dropped not in extras, (
            f"pyproject still declares [{dropped}] extra — remove it; the "
            f"canonical stack is postgres + pinecone + neo4j only."
        )
