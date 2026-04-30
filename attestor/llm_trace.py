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

import time
from typing import Any, Optional


def make_client(
    *,
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 2,
):
    """Construct a default-configured OpenAI/OpenRouter client.

    Centralizes the timeout + retry policy so a hung LLM call cannot
    block the pipeline indefinitely (caught 2026-04-30 — a 133q LME-S
    smoke hung for 5+ hours on a single distill call with no timeout;
    process slept on socket recv with no deadline). 60s is well above
    any healthy reasoning-effort=high call (observed median 2-9s);
    2 retries cover transient transport errors. Callers can override
    per-call via ``traced_create(client, ..., timeout=N)`` when a
    longer-running role needs it.
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
    max_retries: int = 2,
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
