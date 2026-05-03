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
class RerankerCfg:
    """Cross-encoder reranker stage knobs (Phase 1, +60% retrieval per
    Anthropic's Sept-2024 contextual-retrieval research).

    Sits between RRF (Step 3) and graph BFS (Step 4) of the recall
    cascade. Three providers ship in this PR:

      bge     — ``BAAI/bge-reranker-v2-gemma`` via sentence-transformers
                (open weights, deterministic, lazy-loads). Default.
      cohere  — ``rerank-english-v3.0`` via the cohere SDK. Requires
                ``COHERE_API_KEY``; degrades to no-op without.
      llm     — LLM-as-reranker (Haiku 4.5 / gpt-4o-mini) via the
                YAML-configured LiteLLM provider.

    Disabled by default — Phase 1 ships off; bench cells flip on per
    ablation. All providers degrade to "return input unchanged" on
    error so the recall path never breaks.

    Configuration:
      enabled       — master switch.
      provider      — bge | cohere | llm.
      top_k         — how many candidates survive the rerank.
      top_n_input   — how many candidates the provider sees from RRF.
      bge_model     — HF model id for the BGE provider.
      cohere_model  — Cohere model id for the cohere provider.
      llm_model     — LiteLLM model id for the llm provider; ``null``
                      falls back to ``models.benchmark_default``.
    """

    enabled: bool = False
    provider: str = "bge"
    top_k: int = 10
    top_n_input: int = 50
    bge_model: str = "BAAI/bge-reranker-v2-gemma"
    cohere_model: str = "rerank-english-v3.0"
    llm_model: str | None = None


_VALID_RERANKER_PROVIDERS = ("bge", "cohere", "llm")


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
class GlobalQueryCfg:
    """Multi-hop / global-query lane knobs (Microsoft GraphRAG-style).

    The 6-step recall pipeline (vector → BM25 → RRF → reranker → graph
    BFS → MMR → fit) is great for LOCAL queries ("what did I say about
    X?") but fails on GLOBAL queries that span many memories and need
    cluster-level synthesis ("summarize my interactions with X over
    6 months", "what are my recurring concerns?", "what trips have I
    taken across 8 months?").

    When ``enabled`` is True, ``AgentMemory.recall()`` runs a cheap
    classifier over the question. If the classifier votes "global", the
    call short-circuits the 6-step pipeline and instead:

      1. Pulls candidate entities from the user's Neo4j subgraph.
      2. Runs Leiden community detection on the subgraph (via
         ``gds.leiden.stream``).
      3. For each top community, pulls the underlying memories from
         the document store.
      4. Asks a summarizer LLM to write one cluster-level summary.
      5. Returns those summaries as ``RetrievalResult`` objects with
         ``category="global_summary"`` and a ``_cluster_id`` metadata
         marker so callers can tell them apart from local hits.

    Disabled by default — flip per bench cell via
    ``configs/attestor.yaml``'s ``stack.retrieval.global_query`` block
    or ``evals/matrix.yaml``.

    Cost: 1 classifier call (~$0.0001 with Haiku-4.5) + N community
    summaries (N ≤ ``max_clusters``; ~$0.005 each with Haiku-4.5).
    Total ~$0.04 per global query at the default cap of 8 clusters.

    Configuration:
      enabled                   — master switch.
      classifier_model          — small/cheap LLM for the heuristic
                                  tiebreaker. ``null`` means heuristics
                                  only (no LLM call).
      summary_model             — LLM for community summaries.
      max_clusters              — hard cap on the number of communities
                                  the lane will summarize. The largest
                                  communities (by node count) win.
      max_entities_per_cluster  — cap on how many entities are passed
                                  to the doc-store fetch + summarizer
                                  per cluster.
      subgraph_depth            — k-hop neighborhood depth around each
                                  candidate entity. 2 hops matches the
                                  GraphRAG paper.
      cost_per_summary_usd      — naive per-summary cost estimate; the
                                  ``GlobalQueryResult.cost_estimate_usd``
                                  field rolls these up.
    """

    enabled: bool = False
    classifier_model: str | None = "anthropic/claude-haiku-4.5"
    summary_model: str = "anthropic/claude-haiku-4.5"
    max_clusters: int = 8
    max_entities_per_cluster: int = 20
    subgraph_depth: int = 2
    cost_per_summary_usd: float = 0.005


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
    # Default cap when ``recall(long_context=True)`` omits an explicit
    # ``long_context_max_tokens``. Sized for 1M-context answerers — see
    # ``attestor.retrieval.orchestrator.postprocess._long_context_pack``.
    long_context_default_max_tokens: int = 200_000
    multi_query: MultiQueryCfg = field(default_factory=MultiQueryCfg)
    temporal_prefilter: TemporalPrefilterCfg = field(
        default_factory=TemporalPrefilterCfg,
    )
    hyde: HydeCfg = field(default_factory=HydeCfg)
    reranker: RerankerCfg = field(default_factory=RerankerCfg)
    global_query: GlobalQueryCfg = field(default_factory=GlobalQueryCfg)


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
class ContextualEmbeddingCfg:
    """Anthropic-style contextual-embedding ingest knobs.

    When ``enabled`` is True, ``AgentMemory.add()`` runs an LLM
    pre-pass before the vector store add: a small model writes 1-2
    sentences situating the new memory in the last N turns of the same
    session/namespace, the prefix is bounded at ``max_prefix_chars``,
    and the *prefixed* form is what gets embedded. The raw content
    stays in the document store unchanged.

    Reference: https://www.anthropic.com/news/contextual-retrieval
    (Sept 2024 — cuts retrieval failures by ~49% on Anthropic's eval).

    Disabled by default — flip per bench cell via ``configs/attestor.yaml``
    or by overriding the YAML in ``evals/matrix.yaml``. The Phase 1
    rollout ships off so the default codepath is byte-identical to
    ``main``; LME-S ablation cells in the matrix flip it on.

    Cost: one LLM call per ingest. With ``model=claude-haiku-4.5`` and
    a ~1k-char content + 5-turn session window, expect ~$0.001 / sample
    on LME-S — gate via the ``stack.ingest.contextual_embedding.enabled``
    YAML toggle so this never fires unintentionally.

    Configuration:
      enabled         — master switch
      model           — small/cheap LLM for the prefix call
                        (haiku-4.5 is the recommended default)
      session_window  — last N memories from the same session/namespace
                        passed as document context (5 is the sweet spot;
                        longer windows blow up prompt size without
                        meaningful accuracy gain)
      max_prefix_chars — hard cap on the generated prefix
                        (anti-runaway; protects against models that
                        ignore the 1-2 sentence rule)
      cache           — when True, key by content_hash so re-ingesting
                        the same content reuses the prior prefix
                        without paying for another LLM call.
    """

    enabled: bool = False
    model: str = "anthropic/claude-haiku-4.5"
    session_window: int = 5
    max_prefix_chars: int = 200
    cache: bool = True


@dataclass(frozen=True)
class LLMExtractionCfg:
    """LLM-driven entity + relation extraction at ingest (Mem0/Zep alignment).

    When ``enabled`` is True, ``AgentMemory.add()`` runs an LLM pass
    over the content to extract proper-noun + value-bearing entities
    and explicit relationships before writing to the graph store.
    The regex-based ``attestor.graph.extractor`` returns
    ``entity_count=0`` for most LME-S content (verified via production
    traces); this path replaces that pass with an LLM call that
    produces meaningfully richer entities for natural-language
    inputs.

    Disabled by default — flip per bench cell via ``configs/attestor.yaml``
    or ``evals/matrix.yaml``. The shipping default keeps the regex
    extractor wired so the codepath is byte-identical to ``main``.

    Failure isolation: an LLM timeout / malformed response degrades
    to the regex extractor (NOT to empty), so ingest never breaks.

    Cost: ~$0.0003 / ingest with claude-haiku-4.5 on LME-S content
    (~1k input chars, ~80 output tokens of JSON). Lower than the
    contextual-embedding pre-pass because the prompt is shorter and
    the output JSON is small.

    Configuration:
      enabled    — master switch
      model      — small/cheap LLM for the extraction call
                   (haiku-4.5 is the recommended default; the prompt
                   asks for strict JSON which haiku follows reliably)
      timeout_s  — per-request deadline. Default 15s — tight enough
                   that ingest latency stays bounded, loose enough
                   that haiku-4.5 returns comfortably for 4k content.
      cache      — process-local cache by content_hash so re-ingesting
                   the same content reuses the prior result without
                   another LLM call.
    """

    enabled: bool = False
    model: str = "anthropic/claude-haiku-4.5"
    timeout_s: float = 15.0
    cache: bool = True


@dataclass(frozen=True)
class IngestCfg:
    """Ingest-time knobs (write-path transforms).

    Exposes :class:`ContextualEmbeddingCfg` (Anthropic Sept-2024
    chunk-context prefix) and :class:`LLMExtractionCfg` (Mem0/Zep
    LLM-driven entity extraction). New ingest-stage features
    (semantic chunking, redaction, etc.) will land here as sibling
    fields rather than under ``stack.retrieval`` — those are
    read-path levers and live in :class:`RetrievalCfg`.
    """

    contextual_embedding: ContextualEmbeddingCfg = field(
        default_factory=ContextualEmbeddingCfg,
    )
    llm_entity_extraction: LLMExtractionCfg = field(
        default_factory=LLMExtractionCfg,
    )


@dataclass(frozen=True)
class ConsolidationCfg:
    """Reflection / consolidation pass knobs.

    Reflection runs out-of-band (sleep-time) and distills a window of
    raw episodic memories into a small number of compact semantic
    memories. Originals are kept in the supersession chain — nothing
    is deleted. The pass is quality-first (cost OK), so this defaults
    to a strong-but-cheap target_count and a bounded source_limit
    that keeps prompt tokens predictable.

    Configuration:
      enabled         — master switch. Defaults to ``False`` so the
                        legacy install behaviour is unchanged; flip on
                        per-deploy when you want reflection wired into
                        the consolidator schedule.
      target_count    — number of distilled facts produced per pass.
                        5 is the spec-default sweet spot — small enough
                        the LLM doesn't get loose with attribution,
                        large enough to span typical user topic clusters.
      source_limit    — hard cap on source memories per pass. 50 keeps
                        prompt tokens bounded for any commercially
                        sensible context window.
      since_days      — relative lookback window (days). ``None`` means
                        "no lower bound" — the pass picks up every
                        active source. Set to e.g. 30 when running
                        nightly to keep the window monthly.
      model           — explicit LLM id for the distillation step.
                        ``None`` falls back to ``stack.models.distill``.
      dry_run         — when True, the LLM is still called for cost
                        preview but no writes happen. Useful for canary
                        deployments.
    """

    enabled: bool = False
    target_count: int = 5
    source_limit: int = 50
    since_days: int | None = None
    model: str | None = None
    dry_run: bool = False


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
    ingest: IngestCfg = field(default_factory=IngestCfg)
    consolidation: ConsolidationCfg = field(default_factory=ConsolidationCfg)
    pinecone: PineconeCfg | None = None


@dataclass(frozen=True)
class CloudTarget:
    name: str
    region: str
    compute: str
    postgres: str
    neo4j: str
    image_ref: str
