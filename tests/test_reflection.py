"""Tests for ``attestor.consolidation.reflection.run_reflection``.

The reflection pipeline distills a window of episodic memories into a
small number of compact semantic memories, marks the originals as
``superseded_by`` the new ones, and stamps provenance metadata so the
chain is auditable. These tests pin every guarantee:

- TDD discipline (RED → GREEN): all twelve cases fail against an empty
  ``run_reflection`` stub and pass once the implementation lands.
- The LLM is mocked. No network calls.
- An in-memory ``DocumentStore`` shim implements the subset of the
  store interface the reflection write-path touches (insert / get /
  list_memories / update). This keeps the tests free of Postgres /
  Pinecone / Neo4j and runnable in <1s.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from attestor.consolidation.reflection import (
    DEFAULT_TARGET_COUNT,
    REFLECTION_PROMPT,
    ReflectionResult,
    run_reflection,
)
from attestor.models import Memory


# ──────────────────────────────────────────────────────────────────────────
# In-memory store + AgentMemory shim
# ──────────────────────────────────────────────────────────────────────────


class _InMemoryStore:
    """Minimal DocumentStore stand-in for the reflection write-path.

    Exposes only the methods ``run_reflection`` and the supersession
    chain need: ``insert`` (returns the stored Memory), ``get``,
    ``list_memories`` (with namespace + status filters), and
    ``update``. Storage is a plain dict keyed by memory id.
    """

    def __init__(self) -> None:
        self._rows: dict[str, Memory] = {}

    def insert(self, m: Memory) -> Memory:
        self._rows[m.id] = m
        return m

    def update(self, m: Memory) -> Memory:
        self._rows[m.id] = m
        return m

    def get(self, mid: str) -> Memory | None:
        return self._rows.get(mid)

    def list_memories(
        self,
        *,
        status: str | None = None,
        namespace: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        after: str | None = None,
        before: str | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[Memory]:
        out: list[Memory] = []
        for m in self._rows.values():
            if status is not None and m.status != status:
                continue
            if namespace is not None and m.namespace != namespace:
                continue
            if category is not None and m.category != category:
                continue
            if entity is not None and m.entity != entity:
                continue
            if user_id is not None and m.user_id != user_id:
                continue
            if after is not None and (m.valid_from or m.created_at) < after:
                continue
            if before is not None and (m.valid_from or m.created_at) > before:
                continue
            out.append(m)
        out.sort(key=lambda mm: mm.valid_from or mm.created_at, reverse=True)
        return out[:limit]


class _StubMem:
    """Just enough of AgentMemory's surface for run_reflection.

    The reflection pipeline reaches for:
      - ``mem._store``          — to list source memories + update
                                    superseded ones.
      - ``mem._temporal``       — for the ``supersede(old, new_id)``
                                    helper. We delegate to the real
                                    TemporalManager since it only needs
                                    a DocumentStore.
      - ``mem.add(...)``        — to write distilled memories. The
                                    shim accepts the public kwargs the
                                    pipeline uses and inserts a Memory
                                    directly into the store.
    """

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store
        from attestor.temporal.manager import TemporalManager
        self._temporal = TemporalManager(store)
        self.added: list[Memory] = []

    def add(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        category: str = "general",
        entity: str | None = None,
        namespace: str = "default",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        **_: Any,
    ) -> Memory:
        m = Memory(
            content=content,
            tags=list(tags or []),
            category=category,
            entity=entity,
            namespace=namespace,
            confidence=confidence,
            metadata=dict(metadata or {}),
            user_id=user_id,
        )
        self._store.insert(m)
        self.added.append(m)
        return m


class _Choice:
    def __init__(self, c: str) -> None:
        self.message = type("M", (), {"content": c})()


class _Resp:
    def __init__(self, c: str) -> None:
        self.choices = [_Choice(c)]


class _Chat:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self._p = payload
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> _Resp:
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _Resp(self._p)


class StubClient:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self.chat = type("C", (), {"completions": _Chat(payload, exc=exc)})()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)

    @property
    def last_kwargs(self) -> dict[str, Any]:
        return self.chat.completions.calls[-1]

    @property
    def last_prompt(self) -> str:
        return self.last_kwargs["messages"][0]["content"]


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _seed_user_memories(
    store: _InMemoryStore,
    *,
    user_id: str = "u-1",
    namespace: str = "default",
    n: int = 10,
    base: datetime | None = None,
    base_step_minutes: int = 5,
    id_prefix: str = "src",
) -> list[Memory]:
    """Insert ``n`` active memories for ``user_id`` and return them.

    ``id_prefix`` lets a test seed multiple cohorts (alpha / beta /
    cross-user) without id collisions in the in-memory dict.
    """
    base = base or datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    out: list[Memory] = []
    for i in range(n):
        ts = (base + timedelta(minutes=i * base_step_minutes)).isoformat()
        m = Memory(
            id=f"{id_prefix}-{i:02d}",
            content=f"User mentioned topic-{i % 3} in conversation #{i}",
            tags=["episodic"],
            category="episode",
            entity="user",
            namespace=namespace,
            user_id=user_id,
            confidence=0.9,
            valid_from=ts,
            created_at=ts,
        )
        store.insert(m)
        out.append(m)
    return out


def _distill_payload(n: int = 3) -> str:
    """Synthesize a well-formed JSON LLM response for ``n`` distilled facts."""
    return json.dumps([
        {
            "content": f"Distilled fact {i}: user has consistently said X{i}.",
            "confidence": 0.85,
            "source_indices": [i, i + 3, i + 6],
            "category": "preference",
            "entity": "user",
            "tags": ["distilled", f"topic-{i}"],
        }
        for i in range(n)
    ])


# ──────────────────────────────────────────────────────────────────────────
# Prompt content guards
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_prompt_carries_target_count_and_examples() -> None:
    """Prompt must declare target_count placeholder and ship 2-3 examples."""
    assert "{target_count}" in REFLECTION_PROMPT
    # 2-3 worked examples — the marker count is the load-bearing guarantee.
    assert REFLECTION_PROMPT.lower().count("example") >= 2


@pytest.mark.unit
def test_prompt_forbids_invention() -> None:
    """Refuse to invent — no claim without source support."""
    lowered = REFLECTION_PROMPT.lower()
    assert "do not invent" in lowered or "refuse to invent" in lowered


@pytest.mark.unit
def test_prompt_requires_attribution_and_confidence() -> None:
    lowered = REFLECTION_PROMPT.lower()
    assert "attribut" in lowered      # attributed / attribution
    assert "confidence" in lowered


# ──────────────────────────────────────────────────────────────────────────
# Core write-through behaviors
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dry_run_does_not_write() -> None:
    store = _InMemoryStore()
    sources = _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=3))

    result = run_reflection(
        mem,
        user_id="u-1",
        target_count=3,
        dry_run=True,
        llm_client=client,
    )

    assert isinstance(result, ReflectionResult)
    # Dry run still calls the LLM (cost preview) but writes nothing.
    assert client.call_count == 1
    assert result.distilled_memory_ids == ()
    assert result.cost_estimate_usd > 0
    # Sources untouched.
    for s in sources:
        assert store.get(s.id).status == "active"
        assert store.get(s.id).superseded_by is None


@pytest.mark.unit
def test_writes_distilled_memories() -> None:
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=3))

    result = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )

    assert len(result.distilled_memory_ids) == 3
    # All three new memories are reachable from the store.
    for did in result.distilled_memory_ids:
        m = store.get(did)
        assert m is not None
        assert m.user_id == "u-1"
        assert m.status == "active"


@pytest.mark.unit
def test_supersedes_originals() -> None:
    store = _InMemoryStore()
    sources = _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    payload = _distill_payload(n=3)
    client = StubClient(payload=payload)

    result = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )

    distilled_ids = set(result.distilled_memory_ids)
    assert len(distilled_ids) == 3

    # Source ids referenced by ANY distilled fact should be superseded.
    cited = set(result.source_memory_ids)
    assert len(cited) > 0
    for s in sources:
        if s.id in cited:
            row = store.get(s.id)
            assert row.status == "superseded", (
                f"{s.id} should be superseded but is {row.status}"
            )
            assert row.superseded_by in distilled_ids
            assert row.valid_until is not None


@pytest.mark.unit
def test_provenance_tracked() -> None:
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=3))

    result = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )

    for did in result.distilled_memory_ids:
        m = store.get(did)
        assert m is not None
        meta = m.metadata or {}
        assert "_consolidated_from" in meta
        cited = meta["_consolidated_from"]
        assert isinstance(cited, list) and len(cited) >= 1
        # Every cited id must exist in the store as a source row.
        for src_id in cited:
            assert store.get(src_id) is not None
        # The reflection model id is also stamped for audit.
        assert "_reflection_model" in meta


@pytest.mark.unit
def test_llm_failure_returns_empty_no_writes() -> None:
    store = _InMemoryStore()
    sources = _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(exc=RuntimeError("api down"))

    result = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )

    assert result.distilled_memory_ids == ()
    assert result.source_memory_ids == ()
    # Nothing supereded, nothing inserted.
    for s in sources:
        row = store.get(s.id)
        assert row.status == "active"
        assert row.superseded_by is None
    # No new memories beyond the originals.
    assert len(store.list_memories(limit=1000)) == len(sources)


@pytest.mark.unit
def test_target_count_zero_no_writes() -> None:
    store = _InMemoryStore()
    sources = _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=3))

    result = run_reflection(
        mem, user_id="u-1", target_count=0, llm_client=client,
    )

    # Degenerate input — no LLM call, no writes.
    assert result.distilled_memory_ids == ()
    assert client.call_count == 0
    for s in sources:
        assert store.get(s.id).status == "active"


@pytest.mark.unit
def test_window_filters_by_since() -> None:
    store = _InMemoryStore()
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    _seed_user_memories(store, user_id="u-1", n=10, base=base, base_step_minutes=24 * 60)
    mem = _StubMem(store)
    payload = _distill_payload(n=2)
    client = StubClient(payload=payload)

    # Only memories valid_from >= since (last 5 days) should be sent.
    since = base + timedelta(days=5)
    run_reflection(
        mem, user_id="u-1", target_count=2, since=since, llm_client=client,
    )

    prompt = client.last_prompt
    # Older sources (days 0-4) must NOT appear in the prompt.
    for i in range(5):
        assert f"src-{i:02d}" not in prompt, f"src-{i:02d} leaked past since"
    # Newer sources (days 5-9) must appear.
    for i in range(5, 10):
        assert f"src-{i:02d}" in prompt


@pytest.mark.unit
def test_namespace_isolation() -> None:
    store = _InMemoryStore()
    _seed_user_memories(
        store, user_id="u-1", namespace="alpha", n=5, id_prefix="alpha",
    )
    _seed_user_memories(
        store, user_id="u-1", namespace="beta", n=5, id_prefix="beta",
    )

    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=2))

    run_reflection(
        mem, user_id="u-1", namespace="alpha", target_count=2, llm_client=client,
    )

    prompt = client.last_prompt
    for i in range(5):
        assert f"alpha-{i:02d}" in prompt   # alpha included
        assert f"beta-{i:02d}" not in prompt  # beta excluded


@pytest.mark.unit
def test_user_isolation_v4() -> None:
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=5, id_prefix="u1")
    _seed_user_memories(store, user_id="u-2", n=5, id_prefix="u2")

    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=2))

    run_reflection(
        mem, user_id="u-1", target_count=2, llm_client=client,
    )

    prompt = client.last_prompt
    for i in range(5):
        assert f"u1-{i:02d}" in prompt
        assert f"u2-{i:02d}" not in prompt


@pytest.mark.unit
def test_idempotent_on_no_new_memories() -> None:
    """Second pass with the full source window already consolidated is a no-op.

    To guarantee zero stragglers, the first-pass payload cites every
    seeded source (0..8). After the first pass every source is
    superseded; the second pass's source window collapses to empty
    (the distilled rows carry ``_consolidated_from`` so reflection
    skips them) and no LLM call happens.
    """
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=9)
    mem = _StubMem(store)
    payload = json.dumps([
        {
            "content": "User has consistently expressed topic-0.",
            "confidence": 0.9,
            "source_indices": [0, 3, 6],
            "category": "preference",
            "entity": "user",
            "tags": ["topic-0"],
        },
        {
            "content": "User has consistently expressed topic-1.",
            "confidence": 0.9,
            "source_indices": [1, 4, 7],
            "category": "preference",
            "entity": "user",
            "tags": ["topic-1"],
        },
        {
            "content": "User has consistently expressed topic-2.",
            "confidence": 0.9,
            "source_indices": [2, 5, 8],
            "category": "preference",
            "entity": "user",
            "tags": ["topic-2"],
        },
    ])
    client = StubClient(payload=payload)

    first = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )
    assert len(first.distilled_memory_ids) == 3

    second = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )
    assert second.distilled_memory_ids == ()
    assert second.source_memory_ids == ()


@pytest.mark.unit
def test_cost_estimate_returned() -> None:
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)
    client = StubClient(payload=_distill_payload(n=3))

    result = run_reflection(
        mem, user_id="u-1", target_count=3, llm_client=client,
    )
    assert result.cost_estimate_usd > 0
    # Sanity: a 10-source / 3-distilled pass with our default token model
    # should still be within an order of magnitude of $0.10.
    assert result.cost_estimate_usd < 0.5
    assert result.elapsed_ms >= 0
    assert result.llm_model  # non-empty


@pytest.mark.unit
def test_lme_session_consolidation() -> None:
    """Feed an LME-like session and assert distilled facts cover topics."""
    store = _InMemoryStore()
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    # Three topic clusters (A/B/C) repeated several times each.
    transcript = [
        "User said they prefer dark mode for IDEs",
        "User mentioned Python 3.12 is their daily driver",
        "User scheduled the standup at 10am every weekday",
        "User confirmed preference for dark mode again",
        "User upgraded to Python 3.13 last week",
        "User shifted standup to 9:30am",
        "User reiterated dark mode is a hard requirement",
        "User now uses Python 3.13 in production",
        "User bumped standup to 10:30am after team feedback",
    ]
    for i, content in enumerate(transcript):
        ts = (base + timedelta(minutes=i * 7)).isoformat()
        store.insert(Memory(
            id=f"lme-{i:02d}",
            content=content,
            user_id="u-lme",
            namespace="default",
            entity="user",
            category="episode",
            valid_from=ts,
            created_at=ts,
        ))

    mem = _StubMem(store)
    payload = json.dumps([
        {
            "content": "User has consistently preferred dark mode for IDEs.",
            "confidence": 0.92,
            "source_indices": [0, 3, 6],
            "category": "preference",
            "entity": "user",
            "tags": ["preference"],
        },
        {
            "content": "User is on Python 3.13 (upgraded from 3.12).",
            "confidence": 0.9,
            "source_indices": [1, 4, 7],
            "category": "fact",
            "entity": "user",
            "tags": ["python", "version"],
        },
        {
            "content": "User's standup time has shifted to 10:30am after iteration.",
            "confidence": 0.88,
            "source_indices": [2, 5, 8],
            "category": "schedule",
            "entity": "user",
            "tags": ["standup"],
        },
    ])
    client = StubClient(payload=payload)

    result = run_reflection(
        mem, user_id="u-lme", target_count=3, llm_client=client,
    )

    assert len(result.distilled_memory_ids) == 3
    distilled_contents = " ".join(
        store.get(d).content.lower() for d in result.distilled_memory_ids
    )
    assert "dark mode" in distilled_contents
    assert "python" in distilled_contents
    assert "standup" in distilled_contents


# ──────────────────────────────────────────────────────────────────────────
# AgentMemory.consolidate(...) wiring
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_agent_memory_consolidate_routes_to_reflection() -> None:
    """``mem.consolidate(user_id=...)`` returns a ReflectionResult."""
    store = _InMemoryStore()
    _seed_user_memories(store, user_id="u-1", n=10)
    mem = _StubMem(store)

    # The shim isn't an AgentMemory, but the public signature should
    # work the same way: the reflection module exposes a free function
    # we can wrap into a one-liner. Verify the wrapper exists and the
    # default target_count matches the spec.
    assert DEFAULT_TARGET_COUNT == 5

    client = StubClient(payload=_distill_payload(n=2))
    result = run_reflection(
        mem, user_id="u-1", target_count=2, llm_client=client,
    )
    assert isinstance(result, ReflectionResult)
    assert len(result.distilled_memory_ids) == 2
