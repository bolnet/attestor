"""Unit tests for the contextual-embedding ingest module.

Anthropic's Sept 2024 "Contextual Retrieval" research showed that
prefixing each chunk with 1-2 sentences of LLM-generated document-level
context BEFORE embedding cuts retrieval failures by ~49%. The module
``attestor.ingest.contextual`` implements that pre-pass at write time.

These tests cover the contract:

  - disabled_by_default       → raw content embedded, LLM never called.
  - enabled_generates_prefix  → LLM called once, prefix attached to
                                metadata["_context_prefix"], embedded
                                payload is the prefixed form.
  - prefix_is_short           → generated prefix bounded by
                                ``max_prefix_chars``.
  - session_window_provides_context
                              → last N session memories pass through
                                to the LLM prompt.
  - llm_failure_falls_back_gracefully
                              → LLM raises → degrade to raw embedding,
                                emit a warning, never break add().
  - prefix_cached_by_content_hash
                              → re-ingesting the same content reuses
                                the prefix (one LLM call).
  - metadata_round_trips_prefix
                              → end-to-end: prefix survives the
                                document store round-trip.
  - recall_uses_contextualized_embedding
                              → integration sanity: the embedded payload
                                handed to the vector store is the
                                contextualized one.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Module-level helpers under test
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_imports() -> None:
    """The new module + dataclass + entrypoint should import cleanly."""
    from attestor.ingest.contextual import (  # noqa: F401
        ContextPrefix,
        ContextualEmbedder,
        build_context_prefix,
    )


@pytest.mark.unit
def test_config_dataclass_defaults() -> None:
    """Defaults: disabled, haiku, window=5, cap=200, cache=on."""
    from attestor.config import ContextualEmbeddingCfg

    cfg = ContextualEmbeddingCfg()
    assert cfg.enabled is False
    assert cfg.model == "anthropic/claude-haiku-4.5"
    assert cfg.session_window == 5
    assert cfg.max_prefix_chars == 200
    assert cfg.cache is True


@pytest.mark.unit
def test_yaml_loader_parses_ingest_block(tmp_path) -> None:
    """The YAML loader exposes ``stack.ingest.contextual_embedding``."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text("""
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
  ingest:
    contextual_embedding:
      enabled: true
      model: anthropic/claude-haiku-4.5
      session_window: 3
      max_prefix_chars: 150
      cache: false
""")
    stack = load_stack(yaml_path)
    ce = stack.ingest.contextual_embedding
    assert ce.enabled is True
    assert ce.model == "anthropic/claude-haiku-4.5"
    assert ce.session_window == 3
    assert ce.max_prefix_chars == 150
    assert ce.cache is False


@pytest.mark.unit
def test_yaml_loader_defaults_when_block_omitted(tmp_path) -> None:
    """Missing ``stack.ingest`` → default ContextualEmbeddingCfg."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text("""
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
""")
    stack = load_stack(yaml_path)
    assert stack.ingest.contextual_embedding.enabled is False
    assert stack.ingest.contextual_embedding.session_window == 5


# ─────────────────────────────────────────────────────────────────────────
# build_context_prefix — the unit doing the LLM call
# ─────────────────────────────────────────────────────────────────────────


def _fake_llm(text: str) -> Any:
    """Build a mock OpenAI-style client that returns ``text``."""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=text))]
    fake_response.id = "test-id"
    fake_response.model = "test/m"
    fake_response.usage = {
        "prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


@pytest.mark.unit
def test_build_context_prefix_calls_llm_with_session_history() -> None:
    """The LLM prompt should embed the prior memories as document context."""
    from attestor.ingest.contextual import build_context_prefix

    fake_client = _fake_llm("This snippet is about the user's Tokyo trip.")
    captured: dict[str, Any] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_client.chat.completions.create.return_value

    fake_client.chat.completions.create.side_effect = capture

    prior = [
        "Yesterday I flew to Tokyo for work.",
        "Visited the Shibuya crossing this morning.",
        "Lunch at a tiny ramen shop near the station.",
    ]
    result = build_context_prefix(
        content="Tonight I'm going out to see the cherry blossoms.",
        session_history=prior,
        client=fake_client,
        model="test/m",
        max_prefix_chars=200,
    )

    assert "Tokyo" in result.prefix
    # All three prior turns must reach the prompt — that's the whole point.
    user_msg = captured["messages"][-1]["content"]
    for line in prior:
        assert line in user_msg


@pytest.mark.unit
def test_build_context_prefix_truncates_to_max_chars() -> None:
    """Anti-runaway: prefix MUST be capped at ``max_prefix_chars``."""
    from attestor.ingest.contextual import build_context_prefix

    long_text = "A" * 500
    fake_client = _fake_llm(long_text)
    result = build_context_prefix(
        content="X",
        session_history=[],
        client=fake_client,
        model="test/m",
        max_prefix_chars=200,
    )
    assert len(result.prefix) <= 200


@pytest.mark.unit
def test_build_context_prefix_returns_empty_on_llm_failure(caplog) -> None:
    """LLM raises → return empty prefix, log warning, never escape."""
    from attestor.ingest.contextual import build_context_prefix

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("net down")

    with caplog.at_level(logging.WARNING):
        result = build_context_prefix(
            content="X",
            session_history=[],
            client=fake_client,
            model="test/m",
            max_prefix_chars=200,
        )

    assert result.prefix == ""
    assert result.degraded is True
    assert any("contextual" in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_build_context_prefix_strips_label_echo() -> None:
    """Models sometimes echo 'Context:' label — strip it."""
    from attestor.ingest.contextual import build_context_prefix

    fake_client = _fake_llm("Context: This is about the Tokyo trip.")
    result = build_context_prefix(
        content="X",
        session_history=[],
        client=fake_client,
        model="test/m",
        max_prefix_chars=200,
    )
    assert not result.prefix.lower().startswith("context:")
    assert "Tokyo" in result.prefix


# ─────────────────────────────────────────────────────────────────────────
# ContextualEmbedder — the orchestration object used by add()
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_contextual_embedder_disabled_passes_through() -> None:
    """``enabled=False`` → prepare() returns raw content, no metadata,
    LLM client never touched."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    fake_client = MagicMock()
    cfg = ContextualEmbeddingCfg(enabled=False)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    payload, prefix_meta = embedder.prepare(
        content="User likes Python.",
        session_history=[],
    )
    assert payload == "User likes Python."
    assert prefix_meta is None
    fake_client.chat.completions.create.assert_not_called()


@pytest.mark.unit
def test_contextual_embedder_enabled_calls_llm_once_and_prefixes() -> None:
    """``enabled=True`` → LLM called once; payload is the prefixed form;
    metadata carries the prefix."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    fake_client = _fake_llm("This is part of a Python ramp-up thread.")
    cfg = ContextualEmbeddingCfg(enabled=True)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    payload, prefix_meta = embedder.prepare(
        content="User likes Python.",
        session_history=["Just started learning programming yesterday."],
    )
    assert prefix_meta is not None
    assert "Python ramp-up" in prefix_meta
    assert payload.startswith("[CTX]")
    assert "User likes Python." in payload
    assert fake_client.chat.completions.create.call_count == 1


@pytest.mark.unit
def test_contextual_embedder_caches_by_content_hash() -> None:
    """Same content twice → one LLM call (cache hit on second)."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    fake_client = _fake_llm("This is about the user's Python journey.")
    cfg = ContextualEmbeddingCfg(enabled=True, cache=True)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    embedder.prepare(content="User likes Python.", session_history=[])
    embedder.prepare(content="User likes Python.", session_history=[])

    assert fake_client.chat.completions.create.call_count == 1


@pytest.mark.unit
def test_contextual_embedder_cache_disabled_calls_each_time() -> None:
    """``cache=False`` disables the dedup."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    fake_client = _fake_llm("This is about Python.")
    cfg = ContextualEmbeddingCfg(enabled=True, cache=False)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    embedder.prepare(content="User likes Python.", session_history=[])
    embedder.prepare(content="User likes Python.", session_history=[])

    assert fake_client.chat.completions.create.call_count == 2


@pytest.mark.unit
def test_contextual_embedder_falls_back_when_llm_fails() -> None:
    """LLM raises → payload = raw content, prefix_meta = None, no crash."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("net down")

    cfg = ContextualEmbeddingCfg(enabled=True)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    payload, prefix_meta = embedder.prepare(
        content="User likes Python.",
        session_history=[],
    )
    assert payload == "User likes Python."
    assert prefix_meta is None


@pytest.mark.unit
def test_contextual_embedder_session_window_truncates() -> None:
    """``session_window=3`` → only the last 3 history items pass to LLM."""
    from attestor.config import ContextualEmbeddingCfg
    from attestor.ingest.contextual import ContextualEmbedder

    captured: dict[str, Any] = {}
    fake_client = _fake_llm("It's about the latest items.")

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_client.chat.completions.create.return_value

    fake_client.chat.completions.create.side_effect = capture

    cfg = ContextualEmbeddingCfg(enabled=True, session_window=3)
    embedder = ContextualEmbedder(cfg=cfg, client=fake_client)

    history = [f"older-item-{i}" for i in range(10)]
    embedder.prepare(content="latest", session_history=history)

    user_msg = captured["messages"][-1]["content"]
    # Only the last 3 should be present
    assert "older-item-9" in user_msg
    assert "older-item-8" in user_msg
    assert "older-item-7" in user_msg
    # Older ones must be excluded
    assert "older-item-0" not in user_msg
    assert "older-item-3" not in user_msg


# ─────────────────────────────────────────────────────────────────────────
# AgentMemory.add() integration — end-to-end via FakeStore + FakeVector
# ─────────────────────────────────────────────────────────────────────────


class _FakeDocumentStore:
    """Minimal DocumentStore with the two methods AgentMemory.add() actually
    calls (insert, get_by_hash)."""

    ROLES = {"document"}

    def __init__(self) -> None:
        self.memories: dict[str, Any] = {}
        self.by_hash: dict[str, Any] = {}

    def insert(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory
        if memory.content_hash:
            self.by_hash[memory.content_hash] = memory
        return memory

    def get(self, memory_id):  # noqa: ANN001
        return self.memories.get(memory_id)

    def get_by_hash(self, content_hash, namespace="default"):  # noqa: ANN001
        return self.by_hash.get(content_hash)

    def update(self, memory):  # noqa: ANN001
        self.memories[memory.id] = memory

    def list_memories(self, **kwargs):
        return list(self.memories.values())

    def stats(self):
        return {"total_memories": len(self.memories)}

    def close(self):
        pass

    def execute(self, sql, params=None):
        return []

    def increment_access(self, ids):
        pass


class _FakeVectorStore:
    """Records every (memory_id, content) pair passed to add()."""

    ROLES = {"vector"}

    def __init__(self) -> None:
        self.adds: list[tuple[str, str, str]] = []  # (id, payload, namespace)

    def add(self, memory_id, content, namespace="default"):  # noqa: ANN001
        self.adds.append((memory_id, content, namespace))

    def search(self, query, limit=20, namespace=None):  # noqa: ANN001
        return []

    def count(self):
        return len(self.adds)

    def close(self):
        pass


def _build_mem_with_fakes(
    *, ce_enabled: bool, llm_text: str = "This is about Python learning.",
    fail_llm: bool = False,
):
    """Build a real AgentMemory wired to FakeStore + FakeVector +
    a mock LLM client. Bypasses the registry/embedder probe."""
    import tempfile
    from unittest.mock import patch

    from attestor.config import ContextualEmbeddingCfg, IngestCfg
    from attestor.core.agent_memory import AgentMemory

    fake_store = _FakeDocumentStore()
    fake_vec = _FakeVectorStore()

    fake_client = MagicMock()
    if fail_llm:
        fake_client.chat.completions.create.side_effect = RuntimeError("LLM down")
    else:
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=llm_text))]
        resp.id = "x"
        resp.model = "y"
        resp.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        fake_client.chat.completions.create.return_value = resp

    cfg = ContextualEmbeddingCfg(enabled=ce_enabled)
    ingest_cfg = IngestCfg(contextual_embedding=cfg)

    tmp = tempfile.mkdtemp()
    # Skip backend init by patching `_registry` lookup chain
    with patch(
        "attestor.core.agent_memory._registry"
    ) as _reg:
        _reg.return_value.DEFAULT_BACKENDS = ["postgres"]
        _reg.return_value.resolve_backends.return_value = {
            "document": "postgres",
            "vector": "vector",
        }

        def _instantiate(name, path, bcfg):
            if name == "postgres":
                return fake_store
            return fake_vec

        _reg.return_value.instantiate_backend.side_effect = _instantiate

        # The dim-check probe asserts on _embedding_dim — our fake doesn't
        # have one, so the guard short-circuits cleanly.
        with patch(
            "attestor.store.embedder_dim_check.assert_embedder_dim_matches_schema",
            lambda *a, **k: None,
        ):
            mem = AgentMemory(tmp)

    # Inject the contextual embedder + LLM client.
    from attestor.ingest.contextual import ContextualEmbedder

    mem._contextual_embedder = ContextualEmbedder(cfg=cfg, client=fake_client)
    mem._ingest_cfg = ingest_cfg

    return mem, fake_store, fake_vec, fake_client


@pytest.mark.unit
def test_disabled_by_default_passes_through() -> None:
    """Config off → vector store sees raw content; no LLM call."""
    mem, _store, vec, fake_client = _build_mem_with_fakes(ce_enabled=False)
    try:
        m = mem.add("User likes Python.", namespace="ns")
    finally:
        mem.close()

    fake_client.chat.completions.create.assert_not_called()
    assert vec.adds, "vector add should fire"
    _id, payload, _ns = vec.adds[-1]
    assert payload == "User likes Python."
    # Metadata should not carry a prefix when disabled.
    assert "_context_prefix" not in (m.metadata or {})


@pytest.mark.unit
def test_enabled_generates_prefix() -> None:
    """Config on → vector store sees prefixed payload; metadata carries prefix."""
    mem, _store, vec, fake_client = _build_mem_with_fakes(
        ce_enabled=True,
        llm_text="This is part of a Python ramp-up thread.",
    )
    try:
        m = mem.add("User likes Python.", namespace="ns")
    finally:
        mem.close()

    assert fake_client.chat.completions.create.call_count == 1
    assert vec.adds, "vector add should fire"
    _id, payload, _ns = vec.adds[-1]
    assert payload.startswith("[CTX]")
    assert "User likes Python." in payload
    assert "Python ramp-up" in payload
    assert m.metadata.get("_context_prefix") == "This is part of a Python ramp-up thread."


@pytest.mark.unit
def test_prefix_is_short() -> None:
    """Generated prefix is bounded by max_prefix_chars (default 200)."""
    long = "A" * 500
    mem, _store, _vec, _ = _build_mem_with_fakes(
        ce_enabled=True, llm_text=long,
    )
    try:
        m = mem.add("User likes Python.", namespace="ns")
    finally:
        mem.close()
    assert len(m.metadata["_context_prefix"]) <= 200


@pytest.mark.unit
def test_session_window_provides_context() -> None:
    """Three prior session memories should appear in the LLM prompt."""
    captured: dict[str, Any] = {}

    mem, _store, _vec, fake_client = _build_mem_with_fakes(
        ce_enabled=True, llm_text="ctx",
    )

    def capture(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ctx"))]
        resp.id = "x"
        resp.model = "y"
        resp.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return resp

    fake_client.chat.completions.create.side_effect = capture
    try:
        mem.add("Just learned about Python decorators.", namespace="lme_a")
        mem.add("Today I tried Flask for the first time.", namespace="lme_a")
        mem.add("Set up SQLAlchemy this afternoon.", namespace="lme_a")
        mem.add(
            "Now I'm wiring up the routes.",
            namespace="lme_a",
        )
    finally:
        mem.close()

    user_msg = captured["messages"][-1]["content"]
    assert "Python decorators" in user_msg
    assert "Flask" in user_msg
    assert "SQLAlchemy" in user_msg


@pytest.mark.unit
def test_llm_failure_falls_back_gracefully(caplog) -> None:
    """LLM raises → raw content embedded, warning logged, ingest still ok."""
    mem, _store, vec, _ = _build_mem_with_fakes(
        ce_enabled=True, fail_llm=True,
    )
    try:
        with caplog.at_level(logging.WARNING):
            m = mem.add("User likes Python.", namespace="ns")
    finally:
        mem.close()

    assert m is not None
    assert vec.adds, "vector add should still fire on raw payload"
    _id, payload, _ns = vec.adds[-1]
    assert payload == "User likes Python."
    assert "_context_prefix" not in (m.metadata or {})


@pytest.mark.unit
def test_prefix_cached_by_content_hash() -> None:
    """Same content twice → LLM call exactly once."""
    mem, _store, _vec, fake_client = _build_mem_with_fakes(
        ce_enabled=True, llm_text="ctx",
    )
    try:
        mem.add("User likes Python.", namespace="ns_a")
        # Distinct namespace dedup miss but same content_hash → cache hit.
        mem.add("User likes Python.", namespace="ns_b")
    finally:
        mem.close()
    assert fake_client.chat.completions.create.call_count == 1


@pytest.mark.unit
def test_metadata_round_trips_prefix() -> None:
    """End-to-end: prefix survives the document store round-trip via get()."""
    mem, _store, _vec, _ = _build_mem_with_fakes(
        ce_enabled=True, llm_text="round-trip context",
    )
    try:
        m = mem.add("User likes Python.", namespace="ns")
        round_tripped = mem.get(m.id)
    finally:
        mem.close()
    assert round_tripped is not None
    assert round_tripped.metadata.get("_context_prefix") == "round-trip context"


@pytest.mark.unit
def test_recall_uses_contextualized_embedding() -> None:
    """The exact bytes handed to the vector store must be the prefixed form
    when contextual embedding is on."""
    mem, _store, vec, _ = _build_mem_with_fakes(
        ce_enabled=True, llm_text="this is about Python.",
    )
    try:
        mem.add("User likes Python.", namespace="ns")
    finally:
        mem.close()

    assert vec.adds, "vector add should fire"
    _id, payload, _ns = vec.adds[-1]
    assert payload.startswith("[CTX]")
    assert "this is about Python." in payload
    assert "User likes Python." in payload
