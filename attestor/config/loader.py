"""YAML loader, cache, and runtime helpers for the Attestor stack config."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

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
    _MAX_CRITIQUE_REVISIONS,
    _VALID_SC_VOTERS,
)
from attestor.config.resolver import _require, _resolve_env_password

# Project root = ``attestor/`` package's parent. This module lives at
# ``attestor/config/loader.py``; three ``parent`` hops to reach the
# repo root preserves the historical semantics of the pre-split
# ``attestor/config.py`` (which used two hops).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "attestor.yaml"


def _parse_yaml(cfg_path: Path, *, strict: bool) -> StackConfig:
    raw = yaml.safe_load(cfg_path.read_text())
    stack_blk = raw.get("stack") or {}
    image_blk = raw.get("image") or {}
    clouds_blk = raw.get("clouds") or {}

    pg = stack_blk.get("postgres") or {}
    neo = stack_blk.get("neo4j") or {}
    pcn = stack_blk.get("pinecone")  # None when unset → vector role falls back to pgvector
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
    hyde_blk = retrieval_blk.get("hyde") or {}
    hyde_cfg = HydeCfg(
        enabled=bool(hyde_blk.get("enabled", False)),
        generator_model=hyde_blk.get("generator_model"),
        generator_reasoning_effort=str(
            hyde_blk.get("generator_reasoning_effort", "low"),
        ),
        merge=str(hyde_blk.get("merge", "rrf")),
    )
    # Score-blending knobs — the recall hot path reads these every call.
    # Coerce types defensively so YAML int/float duck-typing doesn't
    # surface surprises downstream (e.g. ``0.3`` parsed as int).
    affinity_blk = retrieval_blk.get("graph_affinity_bonus")
    if affinity_blk is None:
        affinity_map: dict[int, float] = {0: 0.30, 1: 0.20, 2: 0.10}
    else:
        affinity_map = {int(k): float(v) for k, v in affinity_blk.items()}

    retrieval_cfg = RetrievalCfg(
        vector_top_k=int(retrieval_blk.get("vector_top_k", 50)),
        mmr_top_n=(int(mmr_top_raw) if mmr_top_raw is not None else None),
        mmr_lambda=float(retrieval_blk.get("mmr_lambda", 0.7)),
        vector_weight=float(retrieval_blk.get("vector_weight", 0.7)),
        graph_weight=float(retrieval_blk.get("graph_weight", 0.3)),
        graph_affinity_bonus=affinity_map,
        graph_unreachable_penalty=float(
            retrieval_blk.get("graph_unreachable_penalty", -0.05),
        ),
        multi_query=mq_cfg,
        temporal_prefilter=tp_cfg,
        hyde=hyde_cfg,
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

    llm_provider = str(_require(llm_blk, "provider", "stack.llm.provider"))
    if llm_provider not in LLM_PROVIDER_DEFAULTS:
        raise SystemExit(
            f"[attestor.config] unknown LLM provider {llm_provider!r}; "
            f"expected one of {sorted(LLM_PROVIDER_DEFAULTS)}"
        )
    llm_defaults = LLM_PROVIDER_DEFAULTS[llm_provider]

    providers_blk = llm_blk.get("providers") or {}
    providers_map: dict[str, ProviderCfg] | None
    default_provider: str | None
    if providers_blk:
        built: dict[str, ProviderCfg] = {}
        for name, entry in providers_blk.items():
            entry = entry or {}
            base_url = entry.get("base_url")
            api_key_env = entry.get("api_key_env")
            if not base_url or not api_key_env:
                raise ValueError(
                    f"providers.{name} requires base_url and api_key_env"
                )
            built[str(name)] = ProviderCfg(
                name=str(name),
                base_url=str(base_url),
                api_key_env=str(api_key_env),
            )
        providers_map = built
        default_provider_raw = llm_blk.get("default_provider")
        if default_provider_raw is not None:
            default_provider = str(default_provider_raw)
            if default_provider not in providers_map:
                raise ValueError(
                    f"default_provider {default_provider!r} not in "
                    f"providers {sorted(providers_map)}"
                )
        else:
            default_provider = None
    else:
        providers_map = None
        default_provider = None

    llm_cfg = LLMCfg(
        provider=llm_provider,
        base_url=str(llm_blk.get("base_url", llm_defaults["base_url"])),
        api_key_env=str(llm_blk.get("api_key_env", llm_defaults["api_key_env"])),
        providers=providers_map,
        default_provider=default_provider,
    )

    neo_auth_blk = neo.get("auth") or {}

    return StackConfig(
        postgres=PostgresCfg(
            url=str(_require(pg, "url", "stack.postgres.url")),
            v4=bool(pg.get("v4", True)),
            skip_schema_init=bool(pg.get("skip_schema_init", True)),
        ),
        neo4j=Neo4jCfg(
            url=str(_require(neo, "url", "stack.neo4j.url")),
            username=str(_require(
                neo_auth_blk, "username", "stack.neo4j.auth.username"
            )),
            password=_resolve_env_password(neo_auth_blk, strict=strict),
            database=str(_require(neo, "database", "stack.neo4j.database")),
        ),
        embedder=EmbedderCfg(
            provider=str(_require(emb, "provider", "stack.embedder.provider")),
            model=str(_require(emb, "model", "stack.embedder.model")),
            dimensions=int(_require(emb, "dimensions", "stack.embedder.dimensions")),
        ),
        models=ModelsCfg(
            answerer=str(_require(models, "answerer", "stack.models.answerer")),
            judge=str(_require(models, "judge", "stack.models.judge")),
            extraction=str(_require(models, "extraction", "stack.models.extraction")),
            distill=str(_require(models, "distill", "stack.models.distill")),
            verifier=str(_require(models, "verifier", "stack.models.verifier")),
            planner=str(_require(models, "planner", "stack.models.planner")),
            benchmark_default=str(_require(
                models, "benchmark_default", "stack.models.benchmark_default"
            )),
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
        budget=int(_require(stack_blk, "budget", "stack.budget")),
        parallel=int(_require(stack_blk, "parallel", "stack.parallel")),
        clouds=dict(clouds_blk),
        self_consistency=sc_cfg,
        critique_revise=cr_cfg,
        pinecone=(
            PineconeCfg(
                host=pcn.get("host"),
                api_key_env=str(pcn.get("api_key_env", "PINECONE_API_KEY")),
                index_name=str(pcn.get("index_name", "attestor")),
                metric=str(pcn.get("metric", "cosine")),
                cloud=str(pcn.get("cloud", "aws")),
                region=str(pcn.get("region", "us-east-1")),
            )
            if pcn is not None else None
        ),
    )


# ─── Public surface ────────────────────────────────────────────────────

_cached_stack: StackConfig | None = None


def load_stack(path: Path | str | None = None, *, strict: bool = False) -> StackConfig:
    """Read ``configs/attestor.yaml`` (or override path) and resolve env refs.

    Bypasses the cache. Most callers should use ``get_stack()`` instead.

    The YAML is the single source of truth for stack/model/embedder/
    budget/parallel choices. If it's missing, this function raises —
    there is no Python-level fallback any more.
    """
    if path is None:
        path = os.environ.get("ATTESTOR_CONFIG") or DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(
            f"[attestor.config] config not found: {cfg_path}\n"
            "        configs/attestor.yaml is required — fallback "
            "constants were removed; YAML is the only source of truth."
        )
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
    provider this stack asked for."""
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
    elif stack.embedder.provider == "pinecone":
        # Pinecone Inference (cloud-only). Drop Voyage so its _try_voyage
        # probe doesn't beat the Pinecone preference, then pin model/dim
        # via env so PineconeEmbeddingProvider picks them up.
        os.environ.pop("VOYAGE_API_KEY", None)
        if not os.environ.get("PINECONE_API_KEY"):
            raise SystemExit(
                "[attestor.config] embedder=pinecone but PINECONE_API_KEY "
                "not set (cloud key from app.pinecone.io required — Local "
                "Docker doesn't serve the Inference API)"
            )
        os.environ["PINECONE_EMBEDDING_MODEL"] = stack.embedder.model
        os.environ["PINECONE_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
    else:
        raise SystemExit(
            f"[attestor.config] unknown embedder provider: {stack.embedder.provider!r}"
        )


def build_backend_config(
    stack: StackConfig, *, no_graph: bool = False
) -> dict[str, Any]:
    """``AgentMemory`` ``backend_configs`` payload for the canonical
    PG+Neo4j stack. When ``no_graph`` is set, drops Neo4j (graph role)
    so a Postgres-only run is possible."""
    from urllib.parse import urlparse

    parsed = urlparse(stack.postgres.url)
    db = (parsed.path or "/").lstrip("/") or "attestor_v4_test"

    pg_cfg = {
        "url": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 5432}",
        "database": db,
        "auth": {
            "username": parsed.username or "postgres",
            "password": parsed.password or "attestor",
        },
        "v4": stack.postgres.v4,
        "skip_schema_init": stack.postgres.skip_schema_init,
    }

    backend_configs: dict[str, dict[str, Any]] = {}
    backends: list[str] = []

    # Document role: postgres always; if pinecone is also configured the
    # postgres backend stays document-only (vector role goes to pinecone).
    # When pinecone is absent we use the bundled "pgvector" registry entry
    # which claims both document and vector.
    if stack.pinecone is not None:
        backend_configs["postgres"] = pg_cfg
        backends.append("postgres")
        pcn_cfg: dict[str, Any] = {
            "index_name": stack.pinecone.index_name,
            "metric": stack.pinecone.metric,
            "cloud": stack.pinecone.cloud,
            "region": stack.pinecone.region,
            "dimension": stack.embedder.dimensions,
        }
        if stack.pinecone.host:
            pcn_cfg["host"] = stack.pinecone.host
        api_key = os.environ.get(stack.pinecone.api_key_env)
        if api_key:
            pcn_cfg["api_key"] = api_key
        backend_configs["pinecone"] = pcn_cfg
        backends.append("pinecone")
    else:
        # Legacy bundle — postgres holds doc + pgvector.
        backend_configs["pgvector"] = pg_cfg
        backends.append("pgvector")

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
) -> dict[str, Any]:
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
    out: dict[str, Any] = {"max_tokens": fallback_max_tokens}
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
    print("  models")
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


def cloud_target(
    stack: StackConfig, name: str, *, version: str | None = None
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
