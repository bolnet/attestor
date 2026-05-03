"""Frozen dataclass definitions for the Attestor stack config.

Pure declarative — no I/O, no env reads. The YAML loader in
``attestor.config.loader`` populates these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── LLM provider catalog ──────────────────────────────────────────────
#
# Known OpenAI-compatible endpoints, indexed by short name. These are
# NOT fallback defaults for stack/model/embedder/budget — those MUST
# come from `configs/attestor.yaml`. This dict only resolves the
# `base_url` and `api_key_env` for a given provider name selected by
# YAML (e.g. `stack.llm.provider: openrouter`). Adding a new provider
# means adding an entry here.
LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


# ─── Dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PostgresCfg:
    url: str
    v4: bool
    skip_schema_init: bool


@dataclass(frozen=True)
class Neo4jCfg:
    url: str
    username: str
    password: str
    database: str


@dataclass(frozen=True)
class PineconeCfg:
    """Pinecone vector backend.

    Two deployment modes (auto-detected from ``host``):
      - **Pinecone Local Docker**: ``host=http://localhost:5080`` — no
        cloud round-trip, free, no rate limit. Matches development.
      - **Pinecone Cloud**: ``host=None`` (or omit) — uses the SDK's
        default routing via ``api_key``. Pair with a serverless index.
    """
    host: str | None
    api_key_env: str
    index_name: str
    metric: str
    cloud: str
    region: str


@dataclass(frozen=True)
class EmbedderCfg:
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class ModelsCfg:
    """Per-role model assignment.

    All seven roles are required. Roles map to actual call sites:

      answerer    — final answer synthesis from retrieved context
                    (attestor.locomo.answer_question, longmemeval.run_async)
      judge       — single-judge correctness verdict on benchmark answers
                    (locomo.judge_answer, longmemeval.judge_*)
      extraction  — fact extraction during ingest
                    (extraction.llm_extractor, extraction.round_extractor)
      distill     — Mem0-style distillation during ingest
                    (longmemeval.run_async distill_model)
      verifier    — second-pass cross-check after judge
                    (mab.run_*, longmemeval verifier role)
      planner     — query rewriting / multi-step planning
                    (retrieval.planner)
      benchmark_default — generic fallback for ad-hoc bench commands
                          where no specific role applies

    ``reasoning_effort`` (gpt-5.x reasoning models only): role → effort
        level (none|minimal|low|medium|high|xhigh). Models that don't
        support this param ignore it silently (Anthropic, older OpenAI).
        Default empty dict = don't pass the param = legacy behavior.

    ``max_tokens``: role → completion-token cap. Reasoning tokens count
        against this, so high-effort roles need real headroom (3000 is
        a typical target on the answerer; 1000 on judge). Default empty
        dict = use a per-role legacy default (300 for most roles).
    """
    answerer: str
    judge: str
    extraction: str
    distill: str
    verifier: str
    planner: str
    benchmark_default: str
    reasoning_effort: dict[str, str] = field(default_factory=dict)
    max_tokens: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiQueryCfg:
    """Multi-query retrieval knobs (Phase 3 PR-C, RC1 — biggest +8%
    accuracy lever).

    When ``enabled`` is True, the orchestrator rewrites the user
    question into ``n`` paraphrases via a single LLM call, runs each
    paraphrase through the vector lane independently, and merges the
    ``n+1`` ranked lists via reciprocal rank fusion before the rest
    of the cascade runs.

    Disabled by default — flip on per bench run via this YAML, or via
    env var ``ATTESTOR_MULTI_QUERY_ENABLED=1`` for ad-hoc smokes.

    Configuration:
      enabled         — master switch
      n               — number of paraphrases (3 is the sweet spot
                        per the RCA)
      rewriter_model  — null → falls back to ``models.extraction``
      rewriter_reasoning_effort — gpt-5.x reasoning effort for the
                        rewriter call; ``low`` is fine — paraphrasing
                        is structurally simple
      merge           — ``rrf`` (consensus-weighted, recommended) or
                        ``union`` (cheaper, no rank fusion)
    """

    enabled: bool = False
    n: int = 3
    rewriter_model: str | None = None
    rewriter_reasoning_effort: str = "low"
    merge: str = "rrf"


@dataclass(frozen=True)
class SelfConsistencyCfg:
    """Answerer-side self-consistency knobs (Phase 3 PR-B, +3-6%).

    When ``enabled`` is True, the LME answerer draws K independent
    samples at non-zero temperature and elects the consensus answer
    via majority vote (default) or a judge LLM. Lives on
    ``StackConfig`` directly because this is answerer behavior, NOT
    retrieval behavior — peer of ``stack.retrieval``, not nested
    inside it.

    Disabled by default — flip per bench run via this YAML, or via
    ``ATTESTOR_SELF_CONSISTENCY_ENABLED=1`` for ad-hoc smokes if
    callers want to add an env override later.

    Cost: K × answerer cost per sample. K=5 means 5x answerer spend;
    gate carefully.

    Configuration:
      enabled       — master switch
      k             — number of samples (5 is the sweet spot per
                      Wang et al. 2022; diminishing returns past ~10)
      temperature   — per-sample temperature; 0.7 is the standard
                      value from the paper
      voter         — ``majority`` (normalized-fingerprint vote) or
                      ``judge_pick`` (LLM picks best of K)
      judge_model   — model id for ``judge_pick``; null falls back
                      to ``models.judge``
    """

    enabled: bool = False
    k: int = 5
    temperature: float = 0.7
    voter: str = "majority"
    judge_model: str | None = None


_VALID_SC_VOTERS = ("majority", "judge_pick")


@dataclass(frozen=True)
class CritiqueReviseCfg:
    """Answerer-side critique-and-revise knobs (Phase 3 PR-E, +3-5%).

    When ``enabled`` is True, the LME answerer runs a three-step
    pipeline: initial answer → critic checks against retrieved
    context → conditional revise. Lives on ``StackConfig`` directly
    because this is answerer behavior, NOT retrieval behavior — peer
    of ``stack.self_consistency``, not nested inside ``stack.retrieval``.

    Disabled by default — flip via ``configs/attestor.yaml``'s
    ``stack.critique_revise`` per bench run.

    Cost: ~3x answerer in the worst case (one critique + one revise
    on top of the initial). In the common case where the critic says
    ``pass`` it's ~2x answerer (one critique, no revise).

    Configuration:
      enabled        — master switch
      critic_model   — model id for the critique step; null falls back
                       to ``models.verifier``
      revise_model   — model id for the revise step; null falls back
                       to ``models.answerer``
      max_revisions  — hard-capped at 1 in this PR (literature shows
                       diminishing returns past one revision and rapidly-
                       rising cost). Loader rejects values > 1.
    """

    enabled: bool = False
    critic_model: str | None = None
    revise_model: str | None = None
    max_revisions: int = 1


# Hard cap enforced by the YAML loader. This PR caps critique-revise
# at one revision; future PRs can lift this when we validate the
# cost / benefit curve past one round.
_MAX_CRITIQUE_REVISIONS = 1


@dataclass(frozen=True)
class HydeCfg:
    """HyDE retrieval knobs (Phase 3 PR-D — +6-10% recall lever).

    When ``enabled`` is True, the orchestrator generates a hypothetical
    answer to the question via a small LLM, embeds it, and runs both
    the original question AND the hypothetical answer through the
    vector lane — RRF-merging the two ranked lists before the rest
    of the cascade. Sibling of multi_query (PR #94); same RRF helpers,
    different rewrite strategy.

    Mutually exclusive with multi_query in this PR — if both flags are
    on, the orchestrator prefers multi_query (logged warning).

    Configuration:
      enabled                       — master switch
      generator_model               — null → ``models.extraction``
      generator_reasoning_effort    — low | medium | high (gpt-5.x knob)
      merge                         — ``rrf`` (default, consensus-weighted)
                                      or ``union`` (cheaper, no fusion)
    """

    enabled: bool = False
    generator_model: str | None = None
    generator_reasoning_effort: str = "low"
    merge: str = "rrf"


@dataclass(frozen=True)
class TemporalPrefilterCfg:
    """Regex-only temporal pre-filter knobs (Phase 3 RC4 — +1.5% LME-S).

    When ``enabled`` is True and the question contains a relative
    time phrase ("two weeks ago", "yesterday", "last Monday"), the
    orchestrator builds a ``TimeWindow`` around the implied event date
    and passes it through the existing ``time_window`` kwarg to the
    vector + BM25 lanes — narrowing recall to memories that are
    plausibly contemporaneous with the question.

    Caller-supplied ``time_window`` always wins; this only fires when
    no explicit bound was passed. Disabled by default so legacy
    callers see no behavior change.

    Configuration:
      enabled         — master switch
      tolerance_days  — half-width of the window in days. People say
                        "last week" when they mean 8 days ago; the
                        tolerance absorbs that drift so the filter
                        doesn't false-negative.
    """

    enabled: bool = False
    tolerance_days: int = 3


@dataclass(frozen=True)
class RetrievalCfg:
    """Knobs for the 6-step recall cascade.

    The recall pipeline (vector → BM25 → RRF → graph → MMR → fit) is
    bounded by three quantities that interact:

      vector_top_k  — pgvector's ``LIMIT`` on the cosine-similarity
                      lane. More candidates = more raw material for
                      MMR + graph affinity to choose from. Default 50.

      mmr_top_n     — cap on what MMR emits; the diversity rerank
                      drops redundant memories. Default ``None``
                      preserves ``mmr_rerank``'s legacy behavior
                      (no explicit cap; it returns whatever survives
                      the diversity trim).

      budget        — token cap on the final pack sent to the
                      answerer. Lives on StackConfig directly (legacy
                      placement); keeping it there for back-compat.
                      Models support 200k-1M context; ``budget`` is
                      what we choose to actually use.

    These three knobs MUST be tuned together. Raising budget alone
    does nothing if MMR cuts to 10 first; raising vector_top_k alone
    just gives MMR more candidates to discard.

    Score-blending knobs (Phase 5 — wired from YAML 2026-05-01):

      mmr_lambda                — diversity vs relevance trade-off in
                                  the MMR rerank step. 1.0 = pure
                                  relevance, 0.0 = pure diversity.
                                  Default 0.7 matches the historical
                                  literal in
                                  ``attestor.retrieval.orchestrator``.

      vector_weight             — weight on the normalized vector
                                  similarity in ``_blend_score``.
                                  Default 0.7.

      graph_weight              — weight on the graph affinity bonus
                                  in ``_blend_score``. Default 0.3.
                                  ``vector_weight + graph_weight`` is
                                  not strictly required to sum to 1
                                  (and historically did) — they're
                                  independent multipliers, not a
                                  convex combination.

      graph_affinity_bonus      — per-hop additive bonus applied when
                                  a candidate's entity is reachable
                                  within ``GRAPH_MAX_DEPTH`` from a
                                  question entity. Default
                                  ``{0: 0.30, 1: 0.20, 2: 0.10}``.

      graph_unreachable_penalty — applied to candidates whose entity
                                  is NOT reachable from any question
                                  entity. Default -0.05.

    ``multi_query`` extends the cascade with a query-rewrite +
    RRF-merge lane in front of the vector step. Disabled by default
    so legacy callers see no behavior change.

    ``temporal_prefilter`` (RC4) detects relative time phrases in the
    question and tightens the event-time window passed to the vector
    + BM25 lanes. Disabled by default; flip per bench run.
    """

    vector_top_k: int = 50
    mmr_top_n: int | None = None
    mmr_lambda: float = 0.7
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    graph_affinity_bonus: dict[int, float] = field(
        default_factory=lambda: {0: 0.30, 1: 0.20, 2: 0.10},
    )
    graph_unreachable_penalty: float = -0.05
    multi_query: MultiQueryCfg = field(default_factory=MultiQueryCfg)
    temporal_prefilter: TemporalPrefilterCfg = field(
        default_factory=TemporalPrefilterCfg,
    )
    hyde: HydeCfg = field(default_factory=HydeCfg)


@dataclass(frozen=True)
class ProviderCfg:
    """One entry in the LLMCfg.providers map."""

    name: str
    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class LLMCfg:
    """Where chat-completion calls are routed.

    ``provider`` selects an OpenAI-compatible endpoint. Two are wired:

      openrouter  — base_url=https://openrouter.ai/api/v1; key=OPENROUTER_API_KEY
                    (default; matches the canonical benchmark stack so
                    cross-provider models like anthropic/claude-sonnet-4-6
                    work in the same call).
      openai      — base_url=https://api.openai.com/v1; key=OPENAI_API_KEY
                    (no per-call OpenRouter markup; only OpenAI models work).

    ``base_url`` and ``api_key_env`` override the per-provider defaults
    when set in YAML — useful for ad-hoc local OpenAI-compatible
    endpoints or for swapping the env var name.

    Multi-provider mode: when ``providers`` is set, it overrides the
    single-provider ``provider``/``base_url``/``api_key_env`` fields,
    and ``default_provider`` selects which one is used for unprefixed
    model names. When unset, the legacy single-provider mode applies.
    """

    provider: str
    base_url: str
    api_key_env: str
    providers: dict[str, ProviderCfg] | None = None
    default_provider: str | None = None

    @classmethod
    def for_provider(cls, provider: str) -> LLMCfg:
        """Build with the canonical defaults for a known provider."""
        defaults = LLM_PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(
                f"unknown LLM provider {provider!r}; "
                f"expected one of {sorted(LLM_PROVIDER_DEFAULTS)}"
            )
        return cls(
            provider=provider,
            base_url=defaults["base_url"],
            api_key_env=defaults["api_key_env"],
        )


@dataclass(frozen=True)
class ImageCfg:
    ref: str
    api_ref_template: str
    registries: dict[str, str] = field(default_factory=dict)

    def native_ref(self, key: str, *, version: str | None = None) -> str:
        if key not in self.registries:
            raise KeyError(f"unknown registry key {key!r}")
        base = self.registries[key]
        tag = f"api-{version}" if version else "latest"
        return f"{base}:{tag}"


@dataclass(frozen=True)
class StackConfig:
    postgres: PostgresCfg
    neo4j: Neo4jCfg
    embedder: EmbedderCfg
    models: ModelsCfg
    llm: LLMCfg
    retrieval: RetrievalCfg
    image: ImageCfg
    budget: int
    parallel: int
    clouds: dict[str, dict[str, Any]]
    self_consistency: SelfConsistencyCfg = field(default_factory=SelfConsistencyCfg)
    critique_revise: CritiqueReviseCfg = field(default_factory=CritiqueReviseCfg)
    pinecone: PineconeCfg | None = None


@dataclass(frozen=True)
class CloudTarget:
    name: str
    region: str
    compute: str
    postgres: str
    neo4j: str
    image_ref: str
