# tests/test_skill_md.py
"""Contract tests for the canonical 2026-agent-SDK SKILL.md packaging.

Both Anthropic (skills-2025-10-02) and OpenAI (Responses API, March 2026)
converged on a SKILL.md file with YAML frontmatter + markdown body as the
distribution format for reusable agent expertise. This test pins the
contract:

    1. The file exists at the canonical path ``skills/attestor-memory/SKILL.md``.
    2. The frontmatter has the fields a 2026 agent SDK will look for
       (``name``, ``description``, ``version``, ``capabilities``).
    3. The body covers the sections an installer will surface to its agent
       (when to use, how to install, API reference, examples).
    4. Every Python example uses methods that ACTUALLY exist on
       ``attestor.AgentMemory`` — preventing the SKILL.md drifting from the
       real public surface.
    5. The SKILL.md version matches ``pyproject.toml`` so wheel + skill never
       advertise different versions.
    6. The ``skills/`` tree is wired into the Poetry ``include`` block so the
       SKILL.md actually ships with the wheel — otherwise the file is dead
       weight in the repo.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = REPO_ROOT / "skills" / "attestor-memory" / "SKILL.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ---------- helpers ---------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _load_skill() -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body_str)`` from the SKILL.md file."""
    text = SKILL_PATH.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    assert match, "SKILL.md must start with a YAML frontmatter block (---\\n...\\n---)"
    fm = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    return fm, body


def _pyproject_version() -> str:
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


# ---------- existence + structure ------------------------------------------


def test_skill_md_exists() -> None:
    """Canonical layout: skills/<skill-name>/SKILL.md."""
    assert SKILL_PATH.exists(), (
        f"SKILL.md missing at {SKILL_PATH.relative_to(REPO_ROOT)}; "
        "this is the path 2026 agent SDKs auto-discover."
    )


def test_frontmatter_has_required_fields() -> None:
    """Frontmatter is the machine-readable contract — installers parse it."""
    fm, _ = _load_skill()
    required = {"name", "description", "version", "capabilities"}
    missing = required - set(fm.keys())
    assert not missing, f"SKILL.md frontmatter missing required fields: {sorted(missing)}"

    assert fm["name"] == "attestor-memory", (
        "Skill name must match the directory; auto-discovery is name-keyed."
    )
    assert isinstance(fm["description"], str) and fm["description"].strip(), (
        "description must be a non-empty string"
    )
    assert isinstance(fm["capabilities"], list) and fm["capabilities"], (
        "capabilities must be a non-empty list of capability strings"
    )


def test_capabilities_cover_core_primitives() -> None:
    """Every public memory primitive an agent reaches for must be advertised."""
    fm, _ = _load_skill()
    caps = set(fm["capabilities"])
    required_caps = {
        "memory.recall",
        "memory.add",
        "memory.timeline",
        "memory.supersede",
        "memory.forget",
    }
    missing = required_caps - caps
    assert not missing, f"capabilities must include {sorted(missing)}; got {sorted(caps)}"


def test_body_has_required_sections() -> None:
    """An installer + a human reading the file both need the same anchors."""
    _, body = _load_skill()
    required_headings = [
        "When to use",
        "How to install",
        "API reference",
        "Examples",
    ]
    for heading in required_headings:
        # Match any heading level (## / ### / ####) that contains the phrase.
        pattern = re.compile(rf"^#{{1,4}}\s+.*{re.escape(heading)}", re.MULTILINE | re.IGNORECASE)
        assert pattern.search(body), (
            f"SKILL.md body missing required section heading: {heading!r}"
        )


def test_version_matches_pyproject() -> None:
    """Wheel and SKILL.md must advertise the same version."""
    fm, _ = _load_skill()
    pyproject_version = _pyproject_version()
    assert str(fm["version"]) == pyproject_version, (
        f"SKILL.md version {fm['version']!r} drifted from "
        f"pyproject {pyproject_version!r}"
    )


# ---------- examples reference real methods --------------------------------


_PY_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
# Match `mem.<method>(`  — only on a public-looking name (no leading underscore).
_MEM_CALL_RE = re.compile(r"\bmem\.([a-z_][a-z0-9_]*)\s*\(")


def _public_agentmemory_methods() -> set[str]:
    from attestor import AgentMemory

    return {
        name
        for name in dir(AgentMemory)
        if not name.startswith("_") and callable(getattr(AgentMemory, name, None))
    }


def test_examples_reference_real_methods() -> None:
    """Every ``mem.<method>(...)`` in the SKILL.md must exist on AgentMemory.

    This is the anti-drift rail. SKILL.md is the public face of attestor for
    2026 — we cannot ship hallucinated APIs into a file that 2026 agent SDKs
    will install verbatim.
    """
    _, body = _load_skill()
    real_methods = _public_agentmemory_methods()

    seen: set[str] = set()
    for block in _PY_FENCE_RE.findall(body):
        for method in _MEM_CALL_RE.findall(block):
            seen.add(method)

    if not seen:
        pytest.fail(
            "Examples section parsed zero `mem.<method>(...)` calls — either "
            "the SKILL.md has no python examples or the convention drifted."
        )

    missing = seen - real_methods
    assert not missing, (
        f"SKILL.md examples reference methods that don't exist on AgentMemory: "
        f"{sorted(missing)}. Either fix the example or expose the method."
    )


# ---------- packaging: wheel must ship the skill ---------------------------


def test_skill_md_in_package_data() -> None:
    """Poetry ``include`` block must ship skills/**/SKILL.md with the wheel.

    A SKILL.md that doesn't ride the wheel is invisible to ``pip install
    attestor`` users, which defeats the entire 2026-agent-SDK plumbing.
    """
    pyproject = tomllib.loads(PYPROJECT.read_text())
    poetry_cfg = pyproject.get("tool", {}).get("poetry", {})
    include = poetry_cfg.get("include", [])

    # Each entry can be a string or a dict with a "path" key.
    paths = [
        (entry["path"] if isinstance(entry, dict) else entry)
        for entry in include
    ]
    assert any("skills" in p and "SKILL.md" in p for p in paths), (
        "pyproject [tool.poetry].include must contain an entry that ships "
        "skills/**/SKILL.md (e.g. 'skills/*/SKILL.md'); current include="
        f"{paths!r}"
    )
