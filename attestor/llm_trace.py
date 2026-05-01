"""Per-call LLM tracing — captures the OpenAI/OpenRouter ``response.usage``
block + per-call latency on every chat completion the codebase makes.

Why this exists:
    The pipeline trace (attestor.trace) covers ingest + recall pipeline
    stages, but every LLM call's `response.usage` block was being
    discarded by the various `_chat()` wrappers throughout the
    codebase. With reasoning_effort=high on gpt-5.x, that meant the
    *invisible* reasoning tokens (which dominate cost) were entirely
    untracked. This module restores end-to-end LLM-call observability
    without touching the model client itself.

Usage:
    from attestor.llm_trace import traced_create

    response = traced_create(
        client,
        role="answerer",          # for per-role aggregation
        model="openai/gpt-5.5",
        max_tokens=3000,
        reasoning_effort="high",
        messages=[...],
    )

    # response is the unmodified OpenAI ChatCompletion object.
    # A `chat.completion` trace event was emitted as a side effect.

Trace event shape (one per call):
    chat.completion role=<role> model=<requested>
                    actual_model=<served>          # OpenRouter aliasing
                    request_id=<gen-...>           # for /v1/generation lookup
                    prompt_tokens=N completion_tokens=N
                    reasoning_tokens=N             # gpt-5.x reasoning param
                    cached_tokens=N                # prompt-cache hit savings
                    total_tokens=N latency_ms=ms

Off when ATTESTOR_TRACE is unset — the event helper short-circuits to
a no-op so this module is zero-overhead in production.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# Patterns that flag a model as requiring `max_completion_tokens` instead of
# the legacy `max_tokens` field. Compiled once at import.
#
#   - ``^gpt-5\.5``       — gpt-5.5 family (the cut where OpenAI flipped the
#                           param name on the chat-completions surface).
#   - ``^gpt-([6-9]|\d{2,})`` — future-proofing for gpt-6/7/8/9 and gpt-1X+.
#   - ``^o[1-9]``         — o-series reasoning models (o1, o3, o4, o1-mini…).
#
# Any model NOT matching these (gpt-4o, gpt-4.1, claude-*, embedders, etc.)
# falls through to the legacy ``max_tokens`` path.
_MAX_COMPLETION_TOKENS_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^gpt-5\.5"),
    re.compile(r"^gpt-([6-9]|\d{2,})"),
    re.compile(r"^o[1-9]"),
)


def make_client(
    *,
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 0,
):
    """Construct a default-configured OpenAI/OpenRouter client.

    Centralizes the timeout + retry policy so a hung LLM call cannot
    block the pipeline indefinitely (caught 2026-04-30 — a 133q LME-S
    smoke hung for 5+ hours on a single distill call with no timeout;
    process slept on socket recv with no deadline). 60s is well above
    any healthy reasoning-effort=high call (observed median 2-9s).

    ``max_retries=0`` is the safe default: SDK-level retries stack on
    top of the per-attempt timeout AND share a connection pool, so a
    transient flake can pin every connection in pool-acquisition wait
    until the wall clock exceeds even a generous timeout (caught
    2026-04-30 v2 run — 17-min hang after 212 successful distills with
    timeout=60, max_retries=2). If a role needs retries, add them at
    the call site with an overall deadline rather than relying on the
    SDK's per-attempt accounting.
    """
    from openai import OpenAI
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def make_async_client(
    *,
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 0,
):
    """Async sibling of ``make_client`` — used by the P1/P2 async
    HyDE + multi-query rewriter paths."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


@dataclass(frozen=True)
class LLMProviderStrategy:
    """Runtime descriptor for a single LLM provider endpoint.

    Mirrors ``attestor.config.ProviderCfg`` but is intentionally
    decoupled — this module stays usable when the project's config
    layer isn't loaded (e.g. in unit tests or standalone scripts that
    pass strategies directly to ``LLMClientPool``).
    """

    name: str
    base_url: str
    api_key_env: str


class LLMClientPool:
    """Thread-safe lazy cache of OpenAI-compatible clients keyed by
    provider name.

    Each provider's client is constructed at most once, on first
    request, using the standard ``make_client(...)`` factory. The pool
    holds an ``RLock`` so concurrent first-touches from worker threads
    cannot double-construct or race the cache. Subsequent calls return
    the cached client by reference.
    """

    def __init__(
        self,
        strategies: Dict[str, LLMProviderStrategy],
        default: str,
    ) -> None:
        if not strategies:
            raise ValueError("LLMClientPool requires at least one strategy")
        if default not in strategies:
            raise ValueError(
                f"default provider {default!r} not in strategies "
                f"{sorted(strategies)}"
            )
        self._strategies: Dict[str, LLMProviderStrategy] = dict(strategies)
        self._default: str = default
        self._clients: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def client_for(self, name: str) -> Any:
        """Return the cached client for ``name``, building it on first
        request. Raises ``KeyError`` if the provider is unknown and
        ``RuntimeError`` if its API-key env var is unset.
        """
        with self._lock:
            cached = self._clients.get(name)
            if cached is not None:
                return cached
            try:
                strategy = self._strategies[name]
            except KeyError:
                raise KeyError(
                    f"unknown LLM provider {name!r}; expected one of "
                    f"{sorted(self._strategies)}"
                )
            api_key = os.environ.get(strategy.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{strategy.api_key_env} not set — required to build "
                    f"the {strategy.name!r} LLM client (base_url="
                    f"{strategy.base_url})"
                )
            client = make_client(base_url=strategy.base_url, api_key=api_key)
            self._clients[name] = client
            return client

    def default_strategy(self) -> LLMProviderStrategy:
        """Return the strategy used when a model has no ``provider/``
        prefix."""
        return self._strategies[self._default]

    @property
    def providers(self) -> Tuple[str, ...]:
        """Sorted tuple of known provider names — handy for error
        messages and tests."""
        return tuple(sorted(self._strategies))


# --- Module-level singleton plumbing -----------------------------------
# The pool is built lazily from ``attestor.config`` on first use so this
# module stays importable in stripped-checkout / no-YAML environments
# (the legacy single-provider path is the fallback). ``_reset_client_pool``
# exists for tests that need to reload config between cases.

_pool_singleton: Optional[LLMClientPool] = None
_singleton_lock = threading.RLock()


def _build_pool_from_config() -> LLMClientPool:
    """Construct an ``LLMClientPool`` from ``attestor.config``'s current
    ``LLMCfg``.

    Imports lazily to avoid circular-import pain (``attestor.config``
    can pull this module indirectly via the trace stack). When the
    config exposes a non-empty ``providers`` map (multi-provider mode),
    every entry becomes a strategy and ``default_provider`` (or the
    first key) becomes the default. Otherwise we fall back to a single-
    provider pool built from the legacy ``provider/base_url/api_key_env``
    triple.
    """
    # Lazy import — same loader path used by attestor/longmemeval.py:_get_client.
    from attestor.config import get_stack  # noqa: WPS433 — intentional lazy

    cfg = get_stack().llm

    providers_map = getattr(cfg, "providers", None)
    if providers_map:
        strategies: Dict[str, LLMProviderStrategy] = {
            name: LLMProviderStrategy(
                name=name,
                base_url=p.base_url,
                api_key_env=p.api_key_env,
            )
            for name, p in providers_map.items()
        }
        default = getattr(cfg, "default_provider", None) or next(iter(strategies))
        return LLMClientPool(strategies, default)

    # Legacy single-provider mode.
    strategy = LLMProviderStrategy(
        name=cfg.provider,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
    )
    return LLMClientPool({cfg.provider: strategy}, cfg.provider)


def _get_pool() -> LLMClientPool:
    """Return the process-wide pool singleton, building it on first call."""
    global _pool_singleton
    with _singleton_lock:
        if _pool_singleton is None:
            _pool_singleton = _build_pool_from_config()
        return _pool_singleton


def _reset_client_pool() -> None:
    """Drop the pool singleton so the next call rebuilds it from config.

    For tests / config-reload scenarios; not part of the public API.
    """
    global _pool_singleton
    with _singleton_lock:
        _pool_singleton = None


def get_client_for_model(model: str) -> Tuple[Any, str]:
    """Resolve ``model`` to an ``(OpenAI client, clean_model)`` pair.

    Splits ``model`` on the first ``/``. If the prefix matches a known
    provider in the pool, the call is routed there and ``clean_model``
    is the suffix (e.g. ``"openai/gpt-5.5"`` → openai client +
    ``"gpt-5.5"``). If there's no slash — or the prefix isn't a known
    provider name AND there's no slash at all — the pool's default
    provider handles the call and ``clean_model`` is the original
    string. A non-empty prefix that isn't a known provider raises
    ``ValueError`` so misconfigured callers fail loudly.
    """
    pool = _get_pool()
    head, sep, tail = model.partition("/")
    if sep:
        if head in pool.providers:
            return pool.client_for(head), tail
        raise ValueError(
            f"unknown LLM provider {head!r} in model {model!r}; "
            f"expected one of {list(pool.providers)}"
        )
    # No slash — route via default.
    default = pool.default_strategy()
    return pool.client_for(default.name), model


def _needs_max_completion_tokens(model: str) -> bool:
    """Return True if ``model`` requires the ``max_completion_tokens``
    field instead of the legacy ``max_tokens``.

    Matches gpt-5.5+, gpt-6+ (future-proof), and o-series reasoning
    models. The caller should strip any ``provider/`` prefix before
    calling — the patterns key on the bare model id.
    """
    return any(p.match(model) for p in _MAX_COMPLETION_TOKENS_PATTERNS)


def traced_create(
    client: Any,
    *,
    role: str,
    **create_kwargs: Any,
) -> Any:
    """Wrap ``client.chat.completions.create()``: emits a trace event
    with the response's usage block + measured latency, then returns
    the unmodified response.

    YAML override behavior: when ``configs/attestor.yaml`` configures
    ``models.max_tokens[role]`` or ``models.reasoning_effort[role]``,
    those values are applied to the create call (YAML wins over caller-
    provided defaults). Lets callers keep their pre-existing hardcoded
    max_tokens (200, 400, 300, etc.) as fallbacks while a single YAML
    edit lifts every site at once. Caller-provided values are used as
    fallbacks when the YAML doesn't have an entry for the role.

    Wraps the API call in a try/except so a tracing / config failure
    doesn't bring down the actual chat call — telemetry must never
    break the user-facing path.
    """
    # Apply YAML overrides for max_tokens and reasoning_effort.
    # Caller's value becomes the fallback used when YAML has no entry
    # for this role.
    try:
        from attestor.config import chat_kwargs_for_role
        fallback = create_kwargs.get("max_tokens", 300)
        yaml_kwargs = chat_kwargs_for_role(role, fallback_max_tokens=fallback)
        # YAML wins: override caller-provided max_tokens
        create_kwargs["max_tokens"] = yaml_kwargs["max_tokens"]
        # Add reasoning_effort if YAML configures it AND caller didn't pass one
        if "reasoning_effort" in yaml_kwargs and "reasoning_effort" not in create_kwargs:
            create_kwargs["reasoning_effort"] = yaml_kwargs["reasoning_effort"]
    except Exception:  # noqa: BLE001 — config lookup is best-effort
        pass

    # Translate max_tokens → max_completion_tokens for model families
    # that require the new field (gpt-5.5+, gpt-6+, o-series). The model
    # may carry a ``provider/`` prefix at this layer; strip it for the
    # discrimination check without mutating the kwarg the SDK sees.
    try:
        raw_model = create_kwargs.get("model", "") or ""
        _, sep, tail = raw_model.partition("/")
        bare_model = tail if sep else raw_model
        if (
            _needs_max_completion_tokens(bare_model)
            and "max_tokens" in create_kwargs
        ):
            create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
    except Exception:  # noqa: BLE001 — translation is best-effort
        pass

    t0 = time.monotonic()
    response = client.chat.completions.create(**create_kwargs)
    latency_ms = round((time.monotonic() - t0) * 1000, 2)

    try:
        emit_chat_trace(
            response,
            role=role,
            requested_model=create_kwargs.get("model", "?"),
            latency_ms=latency_ms,
        )
    except Exception:  # noqa: BLE001 — telemetry is best-effort
        pass

    return response


def emit_chat_trace(
    response: Any,
    *,
    role: str,
    requested_model: str,
    latency_ms: float,
) -> None:
    """Pull the usage block off a ChatCompletion response and emit a
    ``chat.completion`` trace event.

    Defensive against missing fields — older models / non-OpenAI
    backends may not populate every detail block.
    """
    from attestor import trace as _tr
    if not _tr.is_enabled():
        return

    usage = getattr(response, "usage", None) or {}
    # The SDK returns either a typed object or a dict depending on
    # version — handle both.
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "__dict__"):
        usage = dict(usage.__dict__)
    elif isinstance(usage, dict):
        pass
    else:
        usage = {}

    completion_details = usage.get("completion_tokens_details") or {}
    if hasattr(completion_details, "model_dump"):
        completion_details = completion_details.model_dump()
    prompt_details = usage.get("prompt_tokens_details") or {}
    if hasattr(prompt_details, "model_dump"):
        prompt_details = prompt_details.model_dump()

    actual_model = getattr(response, "model", None) or "?"
    request_id = getattr(response, "id", None) or "?"

    _tr.event(
        "chat.completion",
        role=role,
        requested_model=requested_model,
        actual_model=actual_model,
        request_id=request_id,
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        reasoning_tokens=(completion_details or {}).get("reasoning_tokens", 0) or 0,
        cached_tokens=(prompt_details or {}).get("cached_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
        latency_ms=latency_ms,
    )
