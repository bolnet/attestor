"""Loader for externalized prompt templates.

Prompts live as ``.md`` files under ``attestor/extraction/prompts/`` and
are loaded once at module import via ``importlib.resources``. The version
suffix of each prompt name is exposed as a top-level helper so extraction
code can record it in memory metadata for traceability.

The loader is intentionally tiny — read the file, cache the bytes, raise
a clear error if the prompt is missing. No string interpolation here:
callers use ``.format(...)`` (the same contract the previous inline
templates exposed) so test_extraction_prompts.py keeps working without
edits.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Final

PROMPTS_PACKAGE: Final = "attestor.extraction.prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt template by stem (without ``.md`` extension).

    Cached for the lifetime of the process. Raises ``FileNotFoundError``
    (with the prompt name in the message) if the file is missing — never
    silently returns an empty string, since a downstream LLM call against
    an empty prompt would burn tokens for nothing.
    """
    try:
        return (
            resources.files(PROMPTS_PACKAGE)
            .joinpath(f"{name}.md")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise FileNotFoundError(
            f"prompt template {name!r} not found in {PROMPTS_PACKAGE}"
        ) from e


def prompt_version(name: str) -> str:
    """Return the version suffix of a prompt name.

    ``"user_fact_v1"`` -> ``"v1"``. Anything that doesn't match the
    ``<purpose>_v<N>`` shape returns ``"v0"`` so callers always have a
    string to record in metadata.
    """
    parts = name.rsplit("_", 1)
    if len(parts) > 1 and len(parts[1]) > 1 and parts[1].startswith("v"):
        suffix = parts[1]
        if suffix[1:].isdigit():
            return suffix
    return "v0"
