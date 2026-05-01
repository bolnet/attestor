"""Retrieval-pipeline runtime config (frozen, slots).

Holds the score-blending knobs the recall cascade reads at request time.
Sourced from ``configs/attestor.yaml`` via
:meth:`RetrievalRuntimeConfig.from_stack`. Defaults below match the
historical hardcoded constants so existing benches reproduce when YAML
is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping


# ── Pipeline structure (not user-tunable) ─────────────────────────────
# Hard-coded pipeline shape, not a knob: BFS depth bound for the
# graph-narrow lane. Changing this requires a code change because the
# affinity-bonus map below is keyed on the same horizon.
GRAPH_MAX_DEPTH = 2

# ── Tuning defaults (sourced from configs/attestor.yaml) ──────────────
# These literals match the historical hardcoded constants so behavior
# is identical when YAML is silent. The single source of truth is
# ``configs/attestor.yaml`` under ``stack.retrieval``; these defaults
# only fire when a unit test constructs ``RetrievalRuntimeConfig()``
# without arguments.
_DEFAULT_VECTOR_TOP_K = 50
_DEFAULT_MMR_LAMBDA = 0.7
_DEFAULT_VECTOR_WEIGHT = 0.7
_DEFAULT_GRAPH_WEIGHT = 0.3
_DEFAULT_GRAPH_AFFINITY_BONUS: Mapping[int, float] = MappingProxyType(
    {0: 0.30, 1: 0.20, 2: 0.10},
)
_DEFAULT_GRAPH_UNREACHABLE_PENALTY = -0.05


def _default_graph_affinity_bonus() -> dict[int, float]:
    """Mutable factory for the dataclass default — returns a fresh dict
    every time so two ``RetrievalRuntimeConfig()`` instances don't
    accidentally share a mutable default."""
    return dict(_DEFAULT_GRAPH_AFFINITY_BONUS)


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeConfig:
    """Tuning knobs the retrieval pipeline reads at recall time.

    All values are sourced from ``configs/attestor.yaml`` via
    :func:`attestor.config.get_stack` and built with
    :meth:`from_stack`. Defaults below match the historical hardcoded
    constants in this module so existing benches reproduce when YAML is
    silent.

    Named ``RetrievalRuntimeConfig`` to avoid colliding with
    :class:`attestor.config.RetrievalCfg`, which is the YAML-loader
    dataclass (a superset that also carries ``multi_query``,
    ``temporal_prefilter``, ``hyde`` cfgs that this orchestrator wires
    onto separate attributes — those configure independent lanes,
    not the score-blending hot path).
    """

    vector_top_k: int = _DEFAULT_VECTOR_TOP_K
    mmr_lambda: float = _DEFAULT_MMR_LAMBDA
    vector_weight: float = _DEFAULT_VECTOR_WEIGHT
    graph_weight: float = _DEFAULT_GRAPH_WEIGHT
    graph_affinity_bonus: dict[int, float] = field(
        default_factory=_default_graph_affinity_bonus,
    )
    graph_unreachable_penalty: float = _DEFAULT_GRAPH_UNREACHABLE_PENALTY

    @classmethod
    def from_stack(cls, stack: Any) -> RetrievalRuntimeConfig:
        """Build a runtime config from :func:`attestor.config.get_stack`.

        Reads from ``stack.retrieval``. Missing keys fall back to the
        dataclass defaults so partial YAMLs (e.g. test fixtures) don't
        crash.
        """
        r = stack.retrieval
        return cls(
            vector_top_k=int(getattr(r, "vector_top_k", _DEFAULT_VECTOR_TOP_K)),
            mmr_lambda=float(getattr(r, "mmr_lambda", _DEFAULT_MMR_LAMBDA)),
            vector_weight=float(
                getattr(r, "vector_weight", _DEFAULT_VECTOR_WEIGHT),
            ),
            graph_weight=float(
                getattr(r, "graph_weight", _DEFAULT_GRAPH_WEIGHT),
            ),
            graph_affinity_bonus=dict(
                getattr(
                    r, "graph_affinity_bonus", _DEFAULT_GRAPH_AFFINITY_BONUS,
                )
            ),
            graph_unreachable_penalty=float(
                getattr(
                    r,
                    "graph_unreachable_penalty",
                    _DEFAULT_GRAPH_UNREACHABLE_PENALTY,
                )
            ),
        )
