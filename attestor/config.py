"""Single source of truth for Attestor configuration.

Every model name, DB URL, embedder choice, retrieval budget, registry
address and per-cloud deploy plan in this codebase comes from one
file: ``configs/attestor.yaml``. This module is the loader.

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
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "attestor.yaml"


# ─── Hardcoded fallbacks (USED ONLY when YAML is missing) ──────────────
#
# These are the canonical defaults captured at the time the YAML was
# written. They exist so tests / CI can run in a stripped checkout
# without `configs/attestor.yaml`. Production deployments must have
# the YAML; the runtime emits a warning when these fallbacks fire.

_FALLBACK_POSTGRES_URL = "postgresql://postgres:attestor@localhost:5432/attestor_v4_test"
_FALLBACK_NEO4J_URL = "bolt://localhost:7687"
_FALLBACK_NEO4J_DB = "neo4j"
_FALLBACK_NEO4J_USER = "neo4j"
_FALLBACK_EMBEDDER_PROVIDER = "voyage"
_FALLBACK_EMBEDDER_MODEL = "voyage-4"
_FALLBACK_EMBEDDER_DIM = 1024
_FALLBACK_ANSWERER = "openai/gpt-5.4-mini"
_FALLBACK_JUDGE = "openai/gpt-5.5"
# Note: OpenRouter has no gpt-5.5-mini SKU; the -mini roles use 5.4-mini
# (cheapest current 5.x mini) until a 5.5-mini variant ships.
_FALLBACK_EXTRACTION = "openai/gpt-5.4-mini"
_FALLBACK_DISTILL = "openai/gpt-5.4-mini"
_FALLBACK_VERIFIER = "anthropic/claude-sonnet-4-6"
_FALLBACK_PLANNER = "anthropic/claude-opus-4.7"
_FALLBACK_BENCHMARK = "openai/gpt-5.4-mini"
_FALLBACK_BUDGET = 4000
_FALLBACK_PARALLEL = 2

# LLM client routing — which OpenAI-compatible endpoint we send chat
# completion calls to. OpenRouter is the default (matches the canonical
# benchmark stack); "openai" hits api.openai.com directly which avoids
# OpenRouter's per-call markup when the model is an OpenAI one and you
# already have an OpenAI key.
_FALLBACK_LLM_PROVIDER = "openrouter"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": _OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": _OPENAI_BASE_URL,
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
    reasoning_effort: Dict[str, str] = field(default_factory=dict)
    max_tokens: Dict[str, int] = field(default_factory=dict)


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
    rewriter_model: Optional[str] = None
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
    judge_model: Optional[str] = None


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
    critic_model: Optional[str] = None
    revise_model: Optional[str] = None
    max_revisions: int = 1


# Hard cap enforced by the YAML loader. This PR caps critique-revise
# at one revision; future PRs can lift this when we validate the
# cost / benefit curve past one round.
_MAX_CRITIQUE_REVISIONS = 1


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

    ``multi_query`` extends the cascade with a query-rewrite +
    RRF-merge lane in front of the vector step. Disabled by default
    so legacy callers see no behavior change.

    ``temporal_prefilter`` (RC4) detects relative time phrases in the
    question and tightens the event-time window passed to the vector
    + BM25 lanes. Disabled by default; flip per bench run.
    """

    vector_top_k: int = 50
    mmr_top_n: Optional[int] = None
    multi_query: MultiQueryCfg = field(default_factory=MultiQueryCfg)
    temporal_prefilter: TemporalPrefilterCfg = field(
        default_factory=TemporalPrefilterCfg,
    )


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
    when set in YAML — useful for local Ollama (set base_url to
    http://localhost:11434/v1) or for swapping the env var name.
    """

    provider: str
    base_url: str
    api_key_env: str

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
    registries: Dict[str, str] = field(default_factory=dict)

    def native_ref(self, key: str, *, version: Optional[str] = None) -> str:
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
    clouds: Dict[str, Dict[str, Any]]
    self_consistency: SelfConsistencyCfg = field(default_factory=SelfConsistencyCfg)
    critique_revise: CritiqueReviseCfg = field(default_factory=CritiqueReviseCfg)


# ─── YAML helpers ─────────────────────────────────────────────────────

def _resolve_env_password(node: Any, *, strict: bool) -> str:
    """Resolve a password from ``password`` (literal) or ``password_env``."""
    if isinstance(node, dict):
        if node.get("password"):
            return str(node["password"])
        env_name = node.get("password_env")
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
            if strict:
                raise SystemExit(
                    f"[attestor.config] required env {env_name!r} not set"
                )
            return ""  # placeholder for non-strict (tests / CI without secrets)
    if strict:
        raise SystemExit(
            f"[attestor.config] could not resolve password from {node!r}"
        )
    return ""


def _build_fallback_stack() -> StackConfig:
    """Hardcoded stack used when ``configs/attestor.yaml`` is absent."""
    return StackConfig(
        postgres=PostgresCfg(
            url=os.environ.get("POSTGRES_URL", _FALLBACK_POSTGRES_URL),
            v4=True,
            skip_schema_init=True,
        ),
        neo4j=Neo4jCfg(
            url=os.environ.get("NEO4J_URI", _FALLBACK_NEO4J_URL),
            username=os.environ.get("NEO4J_USERNAME", _FALLBACK_NEO4J_USER),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            database=os.environ.get("NEO4J_DATABASE", _FALLBACK_NEO4J_DB),
        ),
        embedder=EmbedderCfg(
            provider=_FALLBACK_EMBEDDER_PROVIDER,
            model=_FALLBACK_EMBEDDER_MODEL,
            dimensions=_FALLBACK_EMBEDDER_DIM,
        ),
        models=ModelsCfg(
            answerer=_FALLBACK_ANSWERER,
            judge=_FALLBACK_JUDGE,
            extraction=_FALLBACK_EXTRACTION,
            distill=_FALLBACK_DISTILL,
            verifier=_FALLBACK_VERIFIER,
            planner=_FALLBACK_PLANNER,
            benchmark_default=_FALLBACK_BENCHMARK,
        ),
        llm=LLMCfg.for_provider(_FALLBACK_LLM_PROVIDER),
        retrieval=RetrievalCfg(),
        image=ImageCfg(ref="", api_ref_template="", registries={}),
        budget=_FALLBACK_BUDGET,
        parallel=_FALLBACK_PARALLEL,
        clouds={},
        self_consistency=SelfConsistencyCfg(),
        critique_revise=CritiqueReviseCfg(),
    )


def _parse_yaml(cfg_path: Path, *, strict: bool) -> StackConfig:
    raw = yaml.safe_load(cfg_path.read_text())
    stack_blk = raw.get("stack") or {}
    image_blk = raw.get("image") or {}
    clouds_blk = raw.get("clouds") or {}

    pg = stack_blk.get("postgres") or {}
    neo = stack_blk.get("neo4j") or {}
    emb = stack_blk.get("embedder") or {}
    models = stack_blk.get("models") or {}
    llm_blk = stack_blk.get("llm") or {}
    retrieval_blk = stack_blk.get("retrieval") or {}

    mmr_top_raw = retrieval_blk.get("mmr_top_n")
    mq_blk = retrieval_blk.get("multi_query") or {}
    mq_cfg = MultiQueryCfg(
        enabled=bool(mq_blk.get("enabled", False)),
        n=int(mq_blk.get("n", 3)),
        rewriter_model=mq_blk.get("rewriter_model"),
        rewriter_reasoning_effort=str(
            mq_blk.get("rewriter_reasoning_effort", "low"),
        ),
        merge=str(mq_blk.get("merge", "rrf")),
    )
    tp_blk = retrieval_blk.get("temporal_prefilter") or {}
    tp_cfg = TemporalPrefilterCfg(
        enabled=bool(tp_blk.get("enabled", False)),
        tolerance_days=int(tp_blk.get("tolerance_days", 3)),
    )
    retrieval_cfg = RetrievalCfg(
        vector_top_k=int(retrieval_blk.get("vector_top_k", 50)),
        mmr_top_n=(int(mmr_top_raw) if mmr_top_raw is not None else None),
        multi_query=mq_cfg,
        temporal_prefilter=tp_cfg,
    )

    sc_blk = stack_blk.get("self_consistency") or {}
    sc_voter = str(sc_blk.get("voter", "majority"))
    if sc_voter not in _VALID_SC_VOTERS:
        raise SystemExit(
            f"[attestor.config] unknown self_consistency voter "
            f"{sc_voter!r}; expected one of {list(_VALID_SC_VOTERS)}"
        )
    sc_cfg = SelfConsistencyCfg(
        enabled=bool(sc_blk.get("enabled", False)),
        k=int(sc_blk.get("k", 5)),
        temperature=float(sc_blk.get("temperature", 0.7)),
        voter=sc_voter,
        judge_model=sc_blk.get("judge_model"),
    )

    cr_blk = stack_blk.get("critique_revise") or {}
    cr_max_revisions = int(cr_blk.get("max_revisions", 1))
    if cr_max_revisions > _MAX_CRITIQUE_REVISIONS:
        raise SystemExit(
            f"[attestor.config] critique_revise.max_revisions="
            f"{cr_max_revisions} exceeds the hard cap of "
            f"{_MAX_CRITIQUE_REVISIONS}; this PR enforces a single "
            f"revision (literature shows diminishing returns past one)"
        )
    if cr_max_revisions < 0:
        raise SystemExit(
            f"[attestor.config] critique_revise.max_revisions="
            f"{cr_max_revisions} must be >= 0"
        )
    cr_cfg = CritiqueReviseCfg(
        enabled=bool(cr_blk.get("enabled", False)),
        critic_model=cr_blk.get("critic_model"),
        revise_model=cr_blk.get("revise_model"),
        max_revisions=cr_max_revisions,
    )

    llm_provider = str(llm_blk.get("provider", _FALLBACK_LLM_PROVIDER))
    if llm_provider not in LLM_PROVIDER_DEFAULTS:
        raise SystemExit(
            f"[attestor.config] unknown LLM provider {llm_provider!r}; "
            f"expected one of {sorted(LLM_PROVIDER_DEFAULTS)}"
        )
    llm_defaults = LLM_PROVIDER_DEFAULTS[llm_provider]
    llm_cfg = LLMCfg(
        provider=llm_provider,
        base_url=str(llm_blk.get("base_url", llm_defaults["base_url"])),
        api_key_env=str(llm_blk.get("api_key_env", llm_defaults["api_key_env"])),
    )

    return StackConfig(
        postgres=PostgresCfg(
            url=pg.get("url", _FALLBACK_POSTGRES_URL),
            v4=bool(pg.get("v4", True)),
            skip_schema_init=bool(pg.get("skip_schema_init", True)),
        ),
        neo4j=Neo4jCfg(
            url=neo.get("url", _FALLBACK_NEO4J_URL),
            username=(neo.get("auth") or {}).get("username", _FALLBACK_NEO4J_USER),
            password=_resolve_env_password(neo.get("auth") or {}, strict=strict),
            database=neo.get("database", _FALLBACK_NEO4J_DB),
        ),
        embedder=EmbedderCfg(
            provider=emb.get("provider", _FALLBACK_EMBEDDER_PROVIDER),
            model=emb.get("model", _FALLBACK_EMBEDDER_MODEL),
            dimensions=int(emb.get("dimensions", _FALLBACK_EMBEDDER_DIM)),
        ),
        models=ModelsCfg(
            answerer=models.get("answerer", _FALLBACK_ANSWERER),
            judge=models.get("judge", _FALLBACK_JUDGE),
            extraction=models.get("extraction", _FALLBACK_EXTRACTION),
            distill=models.get("distill", _FALLBACK_DISTILL),
            verifier=models.get("verifier", _FALLBACK_VERIFIER),
            planner=models.get("planner", _FALLBACK_PLANNER),
            benchmark_default=models.get("benchmark_default", _FALLBACK_BENCHMARK),
            reasoning_effort=dict(models.get("reasoning_effort") or {}),
            max_tokens={k: int(v) for k, v in (models.get("max_tokens") or {}).items()},
        ),
        llm=llm_cfg,
        retrieval=retrieval_cfg,
        image=ImageCfg(
            ref=image_blk.get("ref", ""),
            api_ref_template=image_blk.get("api_ref_template", ""),
            registries=dict(image_blk.get("registries") or {}),
        ),
        budget=int(stack_blk.get("budget", _FALLBACK_BUDGET)),
        parallel=int(stack_blk.get("parallel", _FALLBACK_PARALLEL)),
        clouds=dict(clouds_blk),
        self_consistency=sc_cfg,
        critique_revise=cr_cfg,
    )


# ─── Public surface ────────────────────────────────────────────────────

_cached_stack: Optional[StackConfig] = None


def load_stack(path: Path | str | None = None, *, strict: bool = False) -> StackConfig:
    """Read ``configs/attestor.yaml`` (or override path) and resolve env refs.

    Bypasses the cache. Most callers should use ``get_stack()`` instead.
    """
    if path is None:
        path = os.environ.get("ATTESTOR_CONFIG") or DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        if strict:
            raise SystemExit(f"[attestor.config] config not found: {cfg_path}")
        return _build_fallback_stack()
    return _parse_yaml(cfg_path, strict=strict)


_CACHE_LOCK = threading.Lock()


def get_stack(*, strict: bool = False) -> StackConfig:
    """Return the cached stack, loading on first call.

    Most production code should call this. The cache is per-process and
    safe to call repeatedly. Use ``set_stack()`` for test injection or
    ``reset_stack()`` to force a re-read.

    Thread-safety: double-checked locking around the cache. Without the
    lock, two parallel threads on cold cache would each call
    ``load_stack()`` (wasted YAML reads but correct result). With it,
    only the first thread loads.
    """
    global _cached_stack
    if _cached_stack is None:
        with _CACHE_LOCK:
            if _cached_stack is None:
                _cached_stack = load_stack(strict=strict)
    return _cached_stack


def set_stack(stack: StackConfig) -> None:
    """Override the cached stack. For tests + benchmark harnesses."""
    global _cached_stack
    with _CACHE_LOCK:
        _cached_stack = stack


def reset_stack() -> None:
    """Drop the cache. Next ``get_stack()`` re-reads the YAML."""
    global _cached_stack
    with _CACHE_LOCK:
        _cached_stack = None


# ─── Helpers used by scripts ─────────────────────────────────────────

def configure_embedder(stack: StackConfig) -> None:
    """Pin embedder selection via env vars so the auto-detect chain in
    ``attestor.store.embeddings.get_embedding_provider()`` returns the
    provider this stack asked for. Always disables Ollama auto-probe."""
    os.environ["ATTESTOR_DISABLE_LOCAL_EMBED"] = "1"
    if stack.embedder.provider == "voyage":
        if not os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit(
                "[attestor.config] embedder=voyage but VOYAGE_API_KEY not set"
            )
        os.environ["VOYAGE_EMBEDDING_MODEL"] = stack.embedder.model
        os.environ["VOYAGE_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
        os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
    elif stack.embedder.provider == "openai":
        os.environ.pop("VOYAGE_API_KEY", None)
        os.environ["OPENAI_EMBEDDING_MODEL"] = stack.embedder.model
        os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
    else:
        raise SystemExit(
            f"[attestor.config] unknown embedder provider: {stack.embedder.provider!r}"
        )


def build_backend_config(
    stack: StackConfig, *, no_graph: bool = False
) -> Dict[str, Any]:
    """``AgentMemory`` ``backend_configs`` payload for the canonical
    PG+Neo4j stack. When ``no_graph`` is set, drops Neo4j (graph role)
    so a Postgres-only run is possible."""
    from urllib.parse import urlparse

    parsed = urlparse(stack.postgres.url)
    db = (parsed.path or "/").lstrip("/") or "attestor_v4_test"

    backend_configs: Dict[str, Dict[str, Any]] = {
        "postgres": {
            "url": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 5432}",
            "database": db,
            "auth": {
                "username": parsed.username or "postgres",
                "password": parsed.password or "attestor",
            },
            "v4": stack.postgres.v4,
            "skip_schema_init": stack.postgres.skip_schema_init,
        },
    }
    backends = ["postgres"]
    if not no_graph:
        backend_configs["neo4j"] = {
            "url": stack.neo4j.url,
            "database": stack.neo4j.database,
            "auth": {
                "username": stack.neo4j.username,
                "password": stack.neo4j.password,
            },
        }
        backends.append("neo4j")
    return {
        "mode": "solo",
        "backends": backends,
        "backend_configs": backend_configs,
    }


def verify_neo4j_reachable(stack: StackConfig) -> None:
    """Connect to Neo4j up-front so a benchmark that *expects* the
    graph role doesn't fall through silently."""
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise SystemExit(
            f"[attestor.config] neo4j driver not installed: {e}"
        )
    try:
        with GraphDatabase.driver(
            stack.neo4j.url, auth=(stack.neo4j.username, stack.neo4j.password)
        ) as drv:
            drv.verify_connectivity()
    except Exception as e:
        raise SystemExit(
            f"[attestor.config] Neo4j unreachable at {stack.neo4j.url}: {e}\n"
            "        The default stack requires Neo4j (graph role)."
        )


def chat_kwargs_for_role(
    role: str,
    *,
    fallback_max_tokens: int = 300,
) -> Dict[str, Any]:
    """Build the kwargs dict to pass to OpenAI chat.completions.create()
    for a given role, sourced from the resolved stack.

    Returns at minimum ``{"max_tokens": int}``; adds ``reasoning_effort``
    when the YAML configures one for this role. Models that don't support
    reasoning_effort (Anthropic, older OpenAI) silently ignore it via
    OpenRouter's API surface — no client-side filtering needed.

    Stack load failures (no YAML, missing keys) fall back to safe
    defaults: ``max_tokens=fallback_max_tokens``, no reasoning_effort.
    Callers don't crash on stripped-checkout / test scenarios.
    """
    out: Dict[str, Any] = {"max_tokens": fallback_max_tokens}
    try:
        m = get_stack(strict=False).models
    except Exception:
        return out

    if role in m.max_tokens:
        out["max_tokens"] = m.max_tokens[role]
    if role in m.reasoning_effort:
        out["reasoning_effort"] = m.reasoning_effort[role]
    return out


def print_stack_banner(stack: StackConfig, *, run_label: str) -> None:
    """Print the resolved stack so users sanity-check before any
    expensive call."""
    print("=" * 72)
    print(f"[{run_label}] resolved Attestor stack (configs/attestor.yaml):")
    print(f"  document/vector  postgres @ {stack.postgres.url}")
    print(f"  graph            neo4j    @ {stack.neo4j.url} ({stack.neo4j.database})")
    print(f"  embedder         {stack.embedder.provider}:{stack.embedder.model}"
          f" @ {stack.embedder.dimensions}-D")
    print(f"  llm              {stack.llm.provider} → {stack.llm.base_url}"
          f" (key from ${stack.llm.api_key_env})")
    print(f"  models")
    print(f"    answerer            {stack.models.answerer}")
    print(f"    judge               {stack.models.judge}")
    print(f"    extraction          {stack.models.extraction}")
    print(f"    distill             {stack.models.distill}")
    print(f"    verifier            {stack.models.verifier}")
    print(f"    planner             {stack.models.planner}")
    print(f"    benchmark_default   {stack.models.benchmark_default}")
    if stack.models.reasoning_effort:
        ef = ", ".join(f"{r}={v}" for r, v in stack.models.reasoning_effort.items())
        print(f"  reasoning_effort {ef}")
    if stack.models.max_tokens:
        mt = ", ".join(f"{r}={v}" for r, v in stack.models.max_tokens.items())
        print(f"  max_tokens       {mt}")
    mmr_cap = stack.retrieval.mmr_top_n if stack.retrieval.mmr_top_n is not None else "uncapped"
    print(f"  retrieval        vector_top_k={stack.retrieval.vector_top_k}"
          f" · mmr_top_n={mmr_cap}")
    print(f"  budget           {stack.budget} tokens · parallel = {stack.parallel}")
    print("=" * 72)


def confirm_or_exit(stack: StackConfig, *, run_label: str, yes: bool) -> None:
    """Print banner + interactive confirm (or ``--yes`` for non-interactive)."""
    print_stack_banner(stack, run_label=run_label)
    if yes:
        print(f"[{run_label}] --yes supplied; proceeding without prompt")
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            f"[{run_label}] non-interactive shell and --yes not supplied; aborting."
        )
    answer = input(f"[{run_label}] Proceed with this stack? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit(f"[{run_label}] aborted by user")


@dataclass(frozen=True)
class CloudTarget:
    name: str
    region: str
    compute: str
    postgres: str
    neo4j: str
    image_ref: str


def cloud_target(
    stack: StackConfig, name: str, *, version: Optional[str] = None
) -> CloudTarget:
    """Resolve the deploy plan for ``gcp`` / ``azure`` / ``aws`` with the
    native-registry image ref already substituted."""
    if name not in stack.clouds:
        available = ", ".join(sorted(stack.clouds))
        raise SystemExit(
            f"[attestor.config] unknown cloud {name!r} (available: {available})"
        )
    cloud = stack.clouds[name]
    image_ref = stack.image.native_ref(cloud["image_ref_key"], version=version)
    return CloudTarget(
        name=name,
        region=cloud["region"],
        compute=cloud["compute"],
        postgres=cloud["postgres"],
        neo4j=cloud["neo4j"],
        image_ref=image_ref,
    )
