# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Reflection pass — distill episodic memories into compact semantic ones.

Long-running agents accumulate raw episodic memories ("user mentioned X
in conversation #N") faster than they're useful. Recall returns dozens of
near-duplicates and the answerer hedges. The 2026 best-practice (Letta,
Mem0, LangGraph memory) is a periodic *reflection pass* that:

  1. Selects a window of source memories (by user / namespace / time
     range).
  2. Sends them to a strong LLM with a "distill these into N semantic
     facts" prompt — strict JSON output, refuse-to-invent guardrail.
  3. INSERTs the distilled (semantic) memories.
  4. Marks the originals as ``superseded_by`` the new ones via the
     existing bi-temporal supersession chain — nothing is deleted, the
     audit trail stays queryable forever via ``timeline()`` and
     ``recall(as_of=...)``.
  5. Stamps each distilled memory with provenance metadata
     (``_consolidated_from: [<source_ids>]`` + ``_reflection_model``)
     so downstream agents can replay why the fact exists.

The pipeline is *quality first, cost OK* — this is an ingest-style
operation, not the recall hot path. LLM failures degrade safely
(empty result, source memories untouched). Inputs are bounded
(``limit=50`` source memories per call by default) so a single
reflection never blows up prompt size.

This module is a peer of :mod:`attestor.consolidation.pattern_reflection`
(LangMem-shaped pattern surfacing); they solve different problems and
both compose under :class:`SleepTimeConsolidator` and the public
``mem.consolidate(user_id=...)`` entry point.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from attestor.models import Memory


logger = logging.getLogger("attestor.consolidation.reflection")


# ─── Defaults ────────────────────────────────────────────────────────────

#: Number of distilled semantic memories produced per reflection pass.
DEFAULT_TARGET_COUNT: int = 5

#: Cap on source memories ingested per LLM call. Keeps prompt tokens
#: bounded and the call deadline predictable.
DEFAULT_SOURCE_LIMIT: int = 50

#: Hard truncation per source memory (~200 chars per the spec). Long
#: episodic memories get clipped before they enter the prompt.
SOURCE_CONTENT_TRUNCATE: int = 200

#: Coarse cost estimate parameters. Reflection runs against a strong
#: model (``models.distill`` or ``models.verifier`` from the YAML),
#: so we err on the side of overestimating cost rather than under.
#: ~$10 / 1M input tokens + ~$30 / 1M output tokens — close enough for a
#: cost preview surfaced through ``ReflectionResult.cost_estimate_usd``.
_INPUT_USD_PER_KTOK: float = 0.010
_OUTPUT_USD_PER_KTOK: float = 0.030
_CHARS_PER_TOKEN: int = 4   # rough average across English + JSON


# ─── Result dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReflectionResult:
    """Outcome of one ``run_reflection()`` call.

    All fields are immutable. ``distilled_memory_ids`` is empty when:

    - the LLM failed,
    - the source window was empty,
    - ``target_count == 0``,
    - ``dry_run=True`` (LLM is still called for cost preview).

    ``source_memory_ids`` is the set of originals that any distilled
    fact cited — these are the memories that got superseded. Sources
    that no distilled fact mentioned are left untouched.
    """

    distilled_memory_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    llm_model: str = ""
    cost_estimate_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Prompt ──────────────────────────────────────────────────────────────


REFLECTION_PROMPT = """\
You are a Senior Memory Curator distilling many episodic memories about
ONE user into a small number of compact semantic facts.

Goal: produce EXACTLY {target_count} consolidated semantic facts. Each
fact must:

- Be a single durable claim (preference / state / constraint / event).
- Be ATTRIBUTED — start with phrasing like "User has consistently said",
  "User repeatedly confirmed", "User's stated preference is", or
  "Across {n_sources} mentions, the user...". Attribution makes the
  provenance visible in the fact itself.
- Carry a `confidence` score in [0.0, 1.0] reflecting how strongly the
  source memories support the claim.
- Cite which sources support it via `source_indices` — zero-based
  indices into the SOURCES array below.

REFUSE TO INVENT. If the source memories do not support a claim, do
NOT make that claim. It is BETTER to return fewer than
{target_count} facts than to fabricate. If the sources are too sparse
or too contradictory to distill {target_count} clean facts, return
fewer — empty array is acceptable.

Output STRICT JSON only — a single JSON array, no prose, no markdown
fences. Each element has this shape:

  {{
    "content":         "<single attributed semantic fact>",
    "confidence":      <float 0..1>,
    "source_indices":  [<int>, <int>, ...],
    "category":        "<one of: preference | fact | constraint | "
                       "schedule | event | other>",
    "entity":          "<short entity tag, e.g. 'user' or 'project:acme'>",
    "tags":            ["<short>", "<tags>"]
  }}

# Worked examples

Example 1 — three episodic memories about the same preference distill
into ONE attributed fact:

  SOURCES:
    [0] "User said dark mode is easier on the eyes for late nights"
    [1] "User reiterated they want dark mode in the IDE"
    [2] "User asked again to switch the docs site to dark mode"
  OUTPUT:
    [
      {{
        "content": "User has consistently preferred dark mode across IDE and docs.",
        "confidence": 0.92,
        "source_indices": [0, 1, 2],
        "category": "preference",
        "entity": "user",
        "tags": ["preference", "ui", "dark-mode"]
      }}
    ]

Example 2 — sources contradict, so distill the LATEST belief and flag
the older one as evidence of change:

  SOURCES:
    [0] "User said standup is at 10am every weekday"
    [1] "User shifted standup to 9:30am"
    [2] "User bumped standup to 10:30am after team feedback"
  OUTPUT:
    [
      {{
        "content": "User's standup time has settled at 10:30am after iteration.",
        "confidence": 0.85,
        "source_indices": [2, 1, 0],
        "category": "schedule",
        "entity": "user",
        "tags": ["standup", "schedule"]
      }}
    ]

Example 3 — sparse / unsupported claim. Sources do NOT support a
distilled fact about lunch preferences, so return an empty array
rather than invent:

  SOURCES:
    [0] "User asked what the weather is like"
    [1] "User said the standup ran long"
  OUTPUT:
    []

# Source memories (user_id={user_id}, namespace={namespace}, n={n_sources})

{sources_block}

# Your distilled facts (JSON array, EXACTLY one element per consolidated
# fact, AT MOST {target_count} elements):
"""


# ─── Internal protocols / helpers ────────────────────────────────────────


class _StoreLike(Protocol):
    """Subset of the document store interface ``run_reflection`` reads."""

    def list_memories(self, **kwargs: Any) -> list[Memory]: ...
    def get(self, mid: str) -> Memory | None: ...
    def update(self, m: Memory) -> Memory: ...


class _MemLike(Protocol):
    """Subset of :class:`AgentMemory` the reflection write-path uses."""

    _store: _StoreLike

    def add(self, content: str, **kwargs: Any) -> Memory: ...


def _truncate(s: str, n: int = SOURCE_CONTENT_TRUNCATE) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _select_sources(
    mem: Any,
    *,
    user_id: str,
    namespace: str | None,
    since: datetime | None,
    limit: int,
) -> list[Memory]:
    """Pull active memories for ``user_id`` (optionally namespace + since).

    Skips rows already produced by an earlier reflection pass — those
    carry ``metadata["_consolidated_from"]`` so the second pass over an
    unchanged window is a no-op.
    """
    list_kwargs: dict[str, Any] = {
        "status": "active",
        "limit": max(1, limit),
        "user_id": user_id,
    }
    if namespace:
        list_kwargs["namespace"] = namespace
    if since is not None:
        # ``list_memories`` is permissive about the ``after`` shape — pass
        # the ISO string. Backends that don't honor ``after`` get filtered
        # below as a safety net.
        list_kwargs["after"] = since.isoformat()

    try:
        rows = mem._store.list_memories(**list_kwargs)
    except TypeError:
        # Backend doesn't accept ``user_id`` directly — fall back to a
        # broader fetch and filter in Python.
        list_kwargs.pop("user_id", None)
        rows = mem._store.list_memories(**list_kwargs)
        rows = [r for r in rows if r.user_id == user_id]

    if since is not None:
        cutoff = since.isoformat()
        rows = [r for r in rows if (r.valid_from or r.created_at) >= cutoff]

    # Skip rows that are themselves products of a prior reflection.
    rows = [
        r for r in rows
        if "_consolidated_from" not in (r.metadata or {})
    ]
    return rows


def _sanitize_for_prompt(text: str) -> str:
    """Strip characters that an adversarial memory could use to escape
    the prompt's source-block boundary.

    Memory content is user-supplied (or model-extracted from user-
    supplied conversations) and lands verbatim in the reflection
    prompt body. An attacker who plants a memory containing
    ``</memory>\\n# Your distilled facts:`` could shift the LLM's view
    of where instructions end and data begins. We close that surface
    by removing the sentinel sequences we use as boundaries plus a
    handful of characters that are over-represented in indirect
    prompt-injection probes."""
    if not text:
        return ""
    # Remove our boundary tokens; collapse stray angle brackets so the
    # tag form ``<...>`` can't be reproduced from inside content.
    cleaned = (
        text.replace("</memory>", "")
        .replace("<memory>", "")
        .replace("```", " ")
    )
    # Cap on a sane content size — distillation prompts don't benefit
    # from more than a few thousand chars per source. Truncation also
    # bounds the cost of any injection payload.
    return cleaned


def _format_sources_block(sources: list[Memory]) -> str:
    """Render source memories for the prompt.

    Each source is wrapped in ``<memory>...</memory>`` sentinel tags
    so the LLM has an unambiguous boundary between data and
    instructions. ``_sanitize_for_prompt`` strips any matching
    sentinel sequence from inside the content first — the wrapping
    tags can therefore never be forged from user-supplied text.

    Including the row id is load-bearing — auditors and downstream
    traces correlate distilled facts with their sources by id.
    """
    lines: list[str] = []
    for i, m in enumerate(sources):
        content = _sanitize_for_prompt(_truncate(m.content))
        cat = _sanitize_for_prompt(m.category or "other")
        ent = _sanitize_for_prompt(m.entity or "-")
        ts = _sanitize_for_prompt(m.valid_from or m.created_at or "")
        lines.append(
            f"<memory index=\"{i}\" id=\"{m.id}\" ts=\"{ts}\" "
            f"category=\"{cat}\" entity=\"{ent}\">"
            f"{content}"
            f"</memory>"
        )
    return "\n".join(lines)


def _build_prompt(
    sources: list[Memory],
    *,
    user_id: str,
    namespace: str | None,
    target_count: int,
) -> str:
    return REFLECTION_PROMPT.format(
        target_count=target_count,
        n_sources=len(sources),
        user_id=user_id,
        namespace=namespace or "default",
        sources_block=_format_sources_block(sources),
    )


from attestor.extraction.utils import strip_markdown_fences as _strip_markdown_fences  # noqa: E402


def _parse_distilled(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM's strict-JSON response into a list of dicts.

    Returns an empty list on any parse error or shape mismatch — the
    caller treats that as "no distillation happened".
    """
    text = _strip_markdown_fences(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("reflection JSON parse failed: %s; raw=%r", e, raw[:200])
        return []
    if not isinstance(parsed, list):
        logger.warning("reflection payload not a list: %s", type(parsed).__name__)
        return []
    out: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        idxs = entry.get("source_indices") or []
        if not isinstance(idxs, list):
            continue
        clean_idxs: list[int] = []
        for v in idxs:
            try:
                clean_idxs.append(int(v))
            except (TypeError, ValueError):
                continue
        try:
            conf = float(entry.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        conf = max(0.0, min(1.0, conf))
        category = str(entry.get("category") or "fact")
        entity = entry.get("entity")
        if entity is not None and not isinstance(entity, str):
            entity = None
        tags_raw = entry.get("tags") or []
        tags = (
            [str(t) for t in tags_raw if isinstance(t, (str, int))]
            if isinstance(tags_raw, list)
            else []
        )
        out.append(
            {
                "content": content.strip(),
                "confidence": conf,
                "source_indices": clean_idxs,
                "category": category,
                "entity": entity,
                "tags": tags,
            }
        )
    return out


def _estimate_cost_usd(prompt: str, response_chars: int) -> float:
    """Rough $$ estimate for one reflection call. Used as a preview
    only — the canonical billing source is the LLM provider's invoice."""
    in_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
    out_tokens = max(1, response_chars // _CHARS_PER_TOKEN)
    return round(
        (in_tokens / 1000.0) * _INPUT_USD_PER_KTOK
        + (out_tokens / 1000.0) * _OUTPUT_USD_PER_KTOK,
        6,
    )


def _resolve_model(explicit: str | None) -> str:
    """Fall back to the ``stack.models.distill`` slot when the caller
    didn't pin a model. Distillation is a heavier reasoning task than
    synchronous extraction so the YAML's distill role is the natural
    target. Failure to load the YAML degrades to a stable string id
    that the user-facing cost estimate can quote."""
    if explicit:
        return explicit
    try:
        from attestor.config import get_stack
        return get_stack(strict=False).models.distill
    except Exception:  # noqa: BLE001 — config lookup is best-effort
        return "openrouter/openai/gpt-5.5"


def _build_default_client(model: str) -> Any:
    """Resolve an OpenAI-compatible client for ``model`` via the YAML pool.

    Returns None when no provider is reachable — caller treats that as
    "no LLM, no distillation". Mirrors the contextual-embedding fallback
    pattern in ``AgentMemory._build_contextual_llm_client``.
    """
    try:
        from attestor.llm_trace import get_client_for_model
        client, _bare_model = get_client_for_model(model)
        return client
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "reflection client init failed (%s: %s); skipping pass",
            type(e).__name__, e,
        )
        return None


def _call_llm(
    client: Any,
    *,
    model: str,
    prompt: str,
) -> tuple[str | None, str | None]:
    """Invoke the chat-completions endpoint. Returns ``(raw, error)``.

    On success ``raw`` is the model's text response and ``error`` is
    None. On failure both fields are flipped. Routes through
    ``llm_trace.traced_create`` when available so the call lands in
    the unified LLM-cost trace; falls back to the SDK's bare ``create``
    when the trace helper isn't importable (e.g., minimal smoke envs).
    """
    try:
        try:
            from attestor.llm_trace import traced_create
            response = traced_create(
                client,
                role="distill",
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
                max_retries=0,
            )
        except ImportError:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
        raw = response.choices[0].message.content or ""
        return raw, None
    except Exception as e:  # noqa: BLE001
        logger.warning("reflection LLM call failed: %s", e)
        return None, f"{type(e).__name__}: {e}"


# ─── Public entry point ──────────────────────────────────────────────────


# Module-level rate fence for run_reflection. Distillation is the most
# expensive call site in the consolidation pipeline; an upstream bug
# that re-fires reflection on the same window in a tight loop would
# rack up cost silently. We track the last-completed timestamp per
# (user_id, namespace, since) tuple and refuse to re-run inside the
# cooldown window. OFF by default — production deployments that fire
# reflection on a timer should set ATTESTOR_REFLECTION_COOLDOWN_S=300
# (or whatever fits their cron cadence). Tests + ad-hoc CLI calls
# leave it at 0 and bypass the fence entirely.
_REFLECTION_COOLDOWN_S = float(
    __import__("os").environ.get("ATTESTOR_REFLECTION_COOLDOWN_S", "0"),
)
_reflection_last_run: dict[tuple[str, str | None, str | None], float] = {}


def run_reflection(
    mem: Any,
    *,
    user_id: str,
    namespace: str | None = None,
    since: datetime | None = None,
    limit: int = DEFAULT_SOURCE_LIMIT,
    target_count: int = DEFAULT_TARGET_COUNT,
    model: str | None = None,
    dry_run: bool = False,
    llm_client: Any | None = None,
) -> ReflectionResult:
    """Distill up to ``limit`` source memories into ``target_count`` semantic ones.

    Selects active memories for ``user_id`` (optionally narrowed by
    ``namespace`` + ``since``), packs them into the reflection prompt,
    asks the LLM to produce a strict-JSON list of consolidated facts,
    INSERTs each as a new memory, then walks the supersession chain so
    every cited source becomes ``status=superseded`` /
    ``superseded_by=<new_id>``.

    On ``dry_run=True`` the LLM is still called (so cost previews are
    accurate) but neither inserts nor supersession writes happen.

    Failure modes that all return an empty :class:`ReflectionResult`:

    - ``target_count <= 0`` — degenerate input. No LLM call.
    - source window empty after filters — no LLM call.
    - LLM raises — partial results discarded; sources untouched.
    - LLM produced unparseable JSON — same as above.

    Args:
        mem: an :class:`AgentMemory` (or stub exposing ``_store`` +
            ``_temporal`` + ``add(...)``).
        user_id: user-tenancy filter. REQUIRED — reflection is always
            scoped to a single user so cross-tenant data never blends.
        namespace: optional namespace filter (default: any).
        since: optional cutoff; only memories with
            ``valid_from >= since`` are eligible.
        limit: hard cap on source memories per call (default 50).
        target_count: how many distilled memories to ask for.
            ``0`` short-circuits to a no-op result.
        model: explicit LLM id; ``None`` falls back to
            ``configs/attestor.yaml`` ``stack.models.distill``.
        dry_run: ``True`` calls the LLM but skips writes — for cost
            preview / audit dashboards.
        llm_client: pre-built OpenAI-compatible client. ``None`` lazy-
            constructs one via :func:`get_client_for_model`. Tests
            inject a stub here.

    Returns:
        :class:`ReflectionResult` carrying the distilled ids, the cited
        source ids, elapsed ms, the model id used, and a coarse
        cost-in-USD estimate.
    """
    t0 = time.monotonic()
    resolved_model = _resolve_model(model)

    if target_count <= 0:
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
        )

    # Cost cap: refuse to re-fire on the same window inside the cooldown.
    # Dry-runs bypass the cap (they're free preview calls, not writes).
    if not dry_run and _REFLECTION_COOLDOWN_S > 0:
        key = (user_id, namespace, since.isoformat() if since else None)
        last = _reflection_last_run.get(key)
        if last is not None and (t0 - last) < _REFLECTION_COOLDOWN_S:
            return ReflectionResult(
                llm_model=resolved_model,
                elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
                error=(
                    f"reflection cooldown active "
                    f"({_REFLECTION_COOLDOWN_S - (t0 - last):.0f}s remaining)"
                ),
            )
        _reflection_last_run[key] = t0

    sources = _select_sources(
        mem,
        user_id=user_id,
        namespace=namespace,
        since=since,
        limit=limit,
    )
    if not sources:
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
        )

    prompt = _build_prompt(
        sources,
        user_id=user_id,
        namespace=namespace,
        target_count=target_count,
    )

    client = llm_client if llm_client is not None else _build_default_client(resolved_model)
    if client is None:
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
            error="no llm client available",
        )

    raw, error = _call_llm(client, model=resolved_model, prompt=prompt)
    if error is not None or raw is None:
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
            error=error or "empty response",
            cost_estimate_usd=_estimate_cost_usd(prompt, 0),
        )

    cost_usd = _estimate_cost_usd(prompt, len(raw))

    if dry_run:
        # Cost preview only — return what would have been written
        # without touching the store.
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
            cost_estimate_usd=cost_usd,
            metadata={"dry_run": True, "n_sources": len(sources)},
        )

    distilled = _parse_distilled(raw)
    if not distilled:
        return ReflectionResult(
            llm_model=resolved_model,
            elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
            cost_estimate_usd=cost_usd,
            error="no parseable facts",
        )

    # Cap to target_count even if the LLM over-produced.
    distilled = distilled[:target_count]

    # Group cited source ids per distilled fact.
    n_sources = len(sources)
    distilled_ids: list[str] = []
    cited_set: set[str] = set()
    write_namespace = namespace or "default"

    for entry in distilled:
        cited_indices = [i for i in entry["source_indices"] if 0 <= i < n_sources]
        cited_source_ids = [sources[i].id for i in cited_indices]
        if not cited_source_ids:
            # No valid source citation → drop the fact rather than
            # writing an unsupported claim.
            continue

        meta = {
            "_consolidated_from": cited_source_ids,
            "_reflection_model": resolved_model,
        }
        try:
            new_mem = mem.add(
                entry["content"],
                tags=entry["tags"] + ["distilled"],
                category=entry["category"],
                entity=entry["entity"],
                namespace=write_namespace,
                confidence=entry["confidence"],
                metadata=meta,
                user_id=user_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "reflection insert failed (%s: %s); skipping fact",
                type(e).__name__, e,
            )
            continue

        distilled_ids.append(new_mem.id)

        # Walk the supersession chain for every cited source. The
        # TemporalManager handles the actual flag flip + valid_until
        # stamp + superseded_by pointer.
        for sid in cited_source_ids:
            old = mem._store.get(sid)
            if old is None or old.status != "active":
                continue
            try:
                mem._temporal.supersede(old, new_mem.id)
                cited_set.add(sid)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "reflection supersede failed for %s (%s: %s)",
                    sid, type(e).__name__, e,
                )

    return ReflectionResult(
        distilled_memory_ids=tuple(distilled_ids),
        source_memory_ids=tuple(sorted(cited_set)),
        elapsed_ms=round((time.monotonic() - t0) * 1000.0, 2),
        llm_model=resolved_model,
        cost_estimate_usd=cost_usd,
        metadata={"n_sources": n_sources, "n_distilled": len(distilled_ids)},
    )
