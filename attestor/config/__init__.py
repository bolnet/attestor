"""Single source of truth for Attestor configuration.

Every model name, DB URL, embedder choice, retrieval budget, registry
address and per-cloud deploy plan in this codebase comes from one
file: ``configs/attestor.yaml``. This package is the loader.

Public surface
--------------

    get_stack(*, strict=False) -> StackConfig
        Lazy-loads, parses, and caches the YAML. Subsequent calls
        return the same object. Pass ``strict=True`` to fail loudly
        when required env refs are missing (the loader otherwise
        substitutes safe placeholders so tests/CI work without a
        full secrets set).

    set_stack(stack)
        Replace the cached stack. For tests + benchmark harnesses that
        want to swap configs at runtime.

    reset_stack()
        Drop the cache so the next ``get_stack()`` re-reads the YAML.

    StackConfig (frozen dataclass)
        Resolved values. See class for fields.

Resolution order (highest to lowest priority):

    1. Explicit CLI flag passed by the user
    2. Explicit env var (e.g. POSTGRES_URL, VOYAGE_API_KEY, ANSWER_MODEL)
    3. ``configs/attestor.yaml`` value
    4. Hardcoded fallback (only when YAML is absent AND env unset)

Env override:
    ATTESTOR_CONFIG=/path/to/different.yaml
        Use a non-default config file. Useful for one-off A/B runs.

Module layout
-------------
    models.py    — frozen dataclass declarations
    resolver.py  — env-var / required-key helpers
    loader.py    — YAML parser, cache, and runtime helpers
"""

from __future__ import annotations

# Re-export the public API so the 22+ production callers continue to
# work unchanged via ``from attestor.config import …``.

from attestor.config.models import (
    LLM_PROVIDER_DEFAULTS,
    CloudTarget,
    CritiqueReviseCfg,
    EmbedderCfg,
    HydeCfg,
    ImageCfg,
    LLMCfg,
    ModelsCfg,
    MultiQueryCfg,
    Neo4jCfg,
    PineconeCfg,
    PostgresCfg,
    ProviderCfg,
    RetrievalCfg,
    SelfConsistencyCfg,
    StackConfig,
    TemporalPrefilterCfg,
)
from attestor.config.loader import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    build_backend_config,
    chat_kwargs_for_role,
    cloud_target,
    configure_embedder,
    confirm_or_exit,
    get_stack,
    load_stack,
    print_stack_banner,
    reset_stack,
    set_stack,
    verify_neo4j_reachable,
)

__all__ = [
    # Module-level constants
    "DEFAULT_CONFIG",
    "LLM_PROVIDER_DEFAULTS",
    "REPO_ROOT",
    # Dataclasses
    "CloudTarget",
    "CritiqueReviseCfg",
    "EmbedderCfg",
    "HydeCfg",
    "ImageCfg",
    "LLMCfg",
    "ModelsCfg",
    "MultiQueryCfg",
    "Neo4jCfg",
    "PineconeCfg",
    "PostgresCfg",
    "ProviderCfg",
    "RetrievalCfg",
    "SelfConsistencyCfg",
    "StackConfig",
    "TemporalPrefilterCfg",
    # Functions
    "build_backend_config",
    "chat_kwargs_for_role",
    "cloud_target",
    "configure_embedder",
    "confirm_or_exit",
    "get_stack",
    "load_stack",
    "print_stack_banner",
    "reset_stack",
    "set_stack",
    "verify_neo4j_reachable",
]
