"""Cross-encoder reranker stage — unit tests.

The reranker sits between RRF (Step 3) and graph BFS (Step 4) of the
6-step recall cascade. It consumes the top-N RRF candidates, asks a
cross-encoder (BGE / Cohere / LLM) to re-score each (query, candidate)
pair, then keeps the top-K.

Three providers ship in this PR:

  - ``bge``    — BAAI/bge-reranker-v2-gemma via sentence-transformers.
                 Open weights, deterministic, lazy-loads on first call.
                 No-ops when torch / transformers / sentence-transformers
                 aren't installed.
  - ``cohere`` — Cohere ``rerank-english-v3.0`` via the cohere SDK.
                 Requires ``COHERE_API_KEY``; degrades to no-op without.
  - ``llm``    — LLM-as-reranker (Haiku 4.5 / gpt-4o-mini / etc.) via
                 the YAML-configured LiteLLM provider. Sends the question
                 + top-N candidates, parses an ordered top-K back.

All providers degrade to "return input unchanged" on error. The whole
stage is opt-in via ``configs/attestor.yaml`` (default OFF).

Tests follow the TDD discipline mandated by the user's coding-style
rules — write failing tests first, then minimal implementation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.reranker import (
    BgeReranker,
    CohereReranker,
    LlmReranker,
    NoOpReranker,
    RerankerCfg,
    build_reranker,
    rerank,
)

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_result(content: str, score: float = 0.5) -> RetrievalResult:
    """Build a RetrievalResult around a Memory whose `content` is ``content``."""
    mem = Memory(content=content)
    return RetrievalResult(memory=mem, score=score, match_source="vector")


@pytest.fixture
def candidates() -> list[RetrievalResult]:
    """Five candidates of varying topical similarity to "python coding"."""
    return [
        _make_result("User loves cats and dogs", 0.40),
        _make_result("User prefers Python over Java for backend work", 0.55),
        _make_result("The weather is sunny today", 0.30),
        _make_result("User writes Python scripts for data analysis", 0.60),
        _make_result("User attended a wedding last summer", 0.20),
    ]


# ── RerankerCfg dataclass ─────────────────────────────────────────────


@pytest.mark.unit
def test_reranker_cfg_default_is_disabled() -> None:
    """RerankerCfg defaults match YAML defaults: disabled, provider=bge."""
    cfg = RerankerCfg()
    assert cfg.enabled is False
    assert cfg.provider == "bge"
    assert cfg.top_k == 10
    assert cfg.top_n_input == 50


# ── No-op when disabled ───────────────────────────────────────────────


@pytest.mark.unit
def test_reranker_noop_when_disabled(candidates: list[RetrievalResult]) -> None:
    """``rerank`` with cfg.enabled=False returns the input unchanged."""
    cfg = RerankerCfg(enabled=False)
    out = rerank("python coding", candidates, cfg=cfg)
    assert out == candidates  # same order, same identities


@pytest.mark.unit
def test_reranker_handles_empty_candidates() -> None:
    """Empty input → empty output, no crash."""
    cfg = RerankerCfg(enabled=True, provider="bge")
    out = rerank("anything", [], cfg=cfg)
    assert out == []


# ── Reorder by relevance using a deterministic stub ───────────────────


class _StubReranker:
    """Deterministic reranker: scores by lowercased word-overlap fraction.

    Returns the fraction of query words that appear (as substrings) in
    the candidate's content. ``1.0`` = every word matched, ``0.0`` =
    none matched. Used to verify reorder logic without loading a real
    cross-encoder.
    """

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        words = [w.lower() for w in query.split() if w.strip()]
        if not words:
            return [0.0] * len(candidates)
        out: list[float] = []
        for c in candidates:
            text = c.memory.content.lower()
            matched = sum(1 for w in words if w in text)
            out.append(matched / len(words))
        return out


@pytest.mark.unit
def test_reranker_reorders_by_relevance(
    candidates: list[RetrievalResult],
) -> None:
    """A relevant candidate ('Python ... data analysis') climbs to rank 0."""
    cfg = RerankerCfg(enabled=True, provider="bge", top_k=5)
    out = rerank(
        "python coding", candidates, cfg=cfg, provider=_StubReranker(),
    )
    # Both Python entries should outrank weather/wedding/cats.
    assert len(out) == 5
    assert "Python" in out[0].memory.content
    assert "Python" in out[1].memory.content
    # Tail entries should be the irrelevant ones.
    tail_contents = " ".join(r.memory.content for r in out[2:])
    assert "Python" not in tail_contents


@pytest.mark.unit
def test_reranker_respects_top_k(
    candidates: list[RetrievalResult],
) -> None:
    """``top_k=3`` returns exactly 3 results."""
    cfg = RerankerCfg(enabled=True, provider="bge", top_k=3)
    out = rerank(
        "python coding", candidates, cfg=cfg, provider=_StubReranker(),
    )
    assert len(out) == 3


@pytest.mark.unit
def test_reranker_top_k_larger_than_input_returns_all(
    candidates: list[RetrievalResult],
) -> None:
    """When top_k exceeds candidate count, return all sorted."""
    cfg = RerankerCfg(enabled=True, provider="bge", top_k=99)
    out = rerank(
        "python coding", candidates, cfg=cfg, provider=_StubReranker(),
    )
    assert len(out) == len(candidates)


@pytest.mark.unit
def test_reranker_truncates_to_top_n_input() -> None:
    """``top_n_input`` caps the candidate window the provider sees."""

    class _CountingStub:
        def __init__(self) -> None:
            self.last_n = 0

        def score(
            self, query: str, candidates: list[RetrievalResult],
        ) -> list[float]:
            self.last_n = len(candidates)
            return [float(i) for i in range(len(candidates))]

    stub = _CountingStub()
    cfg = RerankerCfg(enabled=True, provider="bge", top_k=5, top_n_input=3)
    bigger = [_make_result(f"row {i}", 0.5) for i in range(10)]
    rerank("q", bigger, cfg=cfg, provider=stub)
    assert stub.last_n == 3


# ── Failure → no-op fallback ──────────────────────────────────────────


class _ExplodingStub:
    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        raise RuntimeError("provider down")


@pytest.mark.unit
def test_reranker_falls_back_on_error(
    candidates: list[RetrievalResult], caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider raising must NOT abort recall — return input unchanged."""
    import logging
    caplog.set_level(logging.WARNING, logger="attestor.retrieval.reranker")

    cfg = RerankerCfg(enabled=True, provider="bge", top_k=5)
    out = rerank("q", candidates, cfg=cfg, provider=_ExplodingStub())
    assert out == candidates  # unchanged, original order preserved
    assert any(
        "reranker" in rec.message.lower() for rec in caplog.records
    ), "Expected a warning log when provider raises"


# ── BGE provider — lazy load ──────────────────────────────────────────


@pytest.mark.unit
def test_bge_provider_lazy_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call loads the model; second call reuses the cached instance."""
    load_calls = {"n": 0}

    class _FakeST:
        def __init__(self, model_name: str) -> None:
            load_calls["n"] += 1
            self.model_name = model_name

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(i) for i in range(len(pairs))]

    fake_st_module = MagicMock()
    fake_st_module.CrossEncoder = _FakeST
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_st_module)

    rr = BgeReranker(model_name="BAAI/bge-reranker-v2-gemma")
    assert load_calls["n"] == 0  # not loaded at construction time
    rr.score("q", [_make_result("a")])
    assert load_calls["n"] == 1  # loaded on first call
    rr.score("q", [_make_result("b")])
    assert load_calls["n"] == 1  # cached on second call


@pytest.mark.unit
def test_bge_provider_returns_zero_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sentence-transformers → score() raises ImportError so the
    rerank() wrapper degrades to no-op."""
    monkeypatch.setitem(
        __import__("sys").modules, "sentence_transformers", None,
    )
    rr = BgeReranker(model_name="BAAI/bge-reranker-v2-gemma")
    with pytest.raises(ImportError):
        rr.score("q", [_make_result("a")])


# ── Cohere provider ───────────────────────────────────────────────────


@pytest.mark.unit
def test_cohere_provider_uses_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cohere SDK is constructed with the key from COHERE_API_KEY."""
    monkeypatch.setenv("COHERE_API_KEY", "sk-test-cohere")

    fake_client = MagicMock()
    fake_client.rerank.return_value = MagicMock(
        results=[
            MagicMock(index=1, relevance_score=0.95),
            MagicMock(index=0, relevance_score=0.10),
        ],
    )
    fake_cohere_module = MagicMock()
    fake_cohere_module.Client.return_value = fake_client
    monkeypatch.setitem(
        __import__("sys").modules, "cohere", fake_cohere_module,
    )

    rr = CohereReranker(model="rerank-english-v3.0")
    cands = [_make_result("first"), _make_result("second")]
    scores = rr.score("python", cands)

    fake_cohere_module.Client.assert_called_once_with(api_key="sk-test-cohere")
    fake_client.rerank.assert_called_once()
    call_kwargs = fake_client.rerank.call_args.kwargs
    assert call_kwargs["model"] == "rerank-english-v3.0"
    assert call_kwargs["query"] == "python"
    assert call_kwargs["documents"] == ["first", "second"]
    # index 1 → 0.95; index 0 → 0.10.
    assert scores[0] == pytest.approx(0.10)
    assert scores[1] == pytest.approx(0.95)


@pytest.mark.unit
def test_cohere_provider_no_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No COHERE_API_KEY → score() raises so rerank() falls back to no-op."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    rr = CohereReranker(model="rerank-english-v3.0")
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        rr.score("q", [_make_result("a")])


# ── LLM provider ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_llm_provider_calls_litellm(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[RetrievalResult],
) -> None:
    """LLM provider sends prompt with ALL candidates and parses scores."""

    fake_response = MagicMock()
    # JSON array — index → score, ordered by descending relevance.
    fake_response.choices = [
        MagicMock(message=MagicMock(
            content='[{"index": 1, "score": 0.95}, '
                    '{"index": 3, "score": 0.92}, '
                    '{"index": 0, "score": 0.30}, '
                    '{"index": 2, "score": 0.10}, '
                    '{"index": 4, "score": 0.05}]',
        )),
    ]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    def fake_get_client_for_model(model: str) -> tuple[Any, str]:
        return fake_client, model

    monkeypatch.setattr(
        "attestor.llm_trace.get_client_for_model",
        fake_get_client_for_model,
        raising=False,
    )

    rr = LlmReranker(model="anthropic/claude-haiku-4.5")
    scores = rr.score("python coding", candidates)

    # All five candidates must have appeared in the prompt.
    fake_client.chat.completions.create.assert_called_once()
    sent = fake_client.chat.completions.create.call_args.kwargs
    user_prompt = sent["messages"][-1]["content"]
    for c in candidates:
        assert c.memory.content in user_prompt

    # Score for index 1 (Python over Java) must be highest.
    assert scores[1] == pytest.approx(0.95)
    assert scores[3] == pytest.approx(0.92)
    assert scores[2] == pytest.approx(0.10)


@pytest.mark.unit
def test_llm_provider_handles_partial_response(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[RetrievalResult],
) -> None:
    """LLM returning only some indices → missing ones default to 0.0."""

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(
            content='[{"index": 1, "score": 0.9}]',
        )),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    monkeypatch.setattr(
        "attestor.llm_trace.get_client_for_model",
        lambda model: (fake_client, model),
        raising=False,
    )

    rr = LlmReranker(model="anthropic/claude-haiku-4.5")
    scores = rr.score("q", candidates)
    assert scores[1] == pytest.approx(0.9)
    # Other indices default to 0.0
    for i in (0, 2, 3, 4):
        assert scores[i] == 0.0


@pytest.mark.unit
def test_llm_provider_handles_malformed_response_raises(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[RetrievalResult],
) -> None:
    """Non-JSON response → score() raises so the wrapper falls back."""

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content="this is not json")),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    monkeypatch.setattr(
        "attestor.llm_trace.get_client_for_model",
        lambda model: (fake_client, model),
        raising=False,
    )

    rr = LlmReranker(model="anthropic/claude-haiku-4.5")
    with pytest.raises(ValueError, match="non-JSON"):
        rr.score("q", candidates)


# ── build_reranker factory ────────────────────────────────────────────


@pytest.mark.unit
def test_build_reranker_returns_noop_when_disabled() -> None:
    cfg = RerankerCfg(enabled=False, provider="bge")
    rr = build_reranker(cfg)
    assert isinstance(rr, NoOpReranker)


@pytest.mark.unit
def test_build_reranker_unknown_provider_raises() -> None:
    cfg = RerankerCfg(enabled=True, provider="not-a-real-provider")
    with pytest.raises(ValueError, match="unknown reranker provider"):
        build_reranker(cfg)


@pytest.mark.unit
def test_build_reranker_bge() -> None:
    cfg = RerankerCfg(enabled=True, provider="bge")
    rr = build_reranker(cfg)
    assert isinstance(rr, BgeReranker)


@pytest.mark.unit
def test_build_reranker_cohere() -> None:
    cfg = RerankerCfg(enabled=True, provider="cohere")
    rr = build_reranker(cfg)
    assert isinstance(rr, CohereReranker)


@pytest.mark.unit
def test_build_reranker_llm() -> None:
    cfg = RerankerCfg(enabled=True, provider="llm")
    rr = build_reranker(cfg)
    assert isinstance(rr, LlmReranker)


# ── Orchestrator integration ──────────────────────────────────────────


@pytest.mark.unit
def test_orchestrator_calls_reranker_when_enabled() -> None:
    """Smoke: orchestrator with reranker_cfg.enabled=True calls into rerank
    and re-orders the candidate stream before MMR."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    class _StubStore:
        def __init__(self) -> None:
            self._mems = {
                "m1": Memory(id="m1", content="User loves cats", tags=[]),
                "m2": Memory(
                    id="m2",
                    content="User writes Python for backend",
                    tags=[],
                ),
                "m3": Memory(
                    id="m3", content="User goes hiking weekly", tags=[],
                ),
            }

        def get(self, mid: str) -> Memory | None:
            return self._mems.get(mid)

    class _StubVec:
        def search(self, q: str, *, limit: int = 50, namespace: str | None = None,
                   as_of: Any = None, time_window: Any = None) -> list[dict]:
            # Return in a deliberately bad order; reranker should fix it.
            return [
                {"memory_id": "m1", "distance": 0.20},
                {"memory_id": "m3", "distance": 0.30},
                {"memory_id": "m2", "distance": 0.40},
            ]

    orch = RetrievalOrchestrator(store=_StubStore(), vector_store=_StubVec())
    orch.reranker_cfg = RerankerCfg(
        enabled=True, provider="bge", top_k=3, top_n_input=10,
    )
    orch.reranker = _StubReranker()  # injected by the test path

    out = orch.recall("python coding", token_budget=2000)
    contents = [r.memory.content for r in out if r.match_source == "vector"]
    # Reranker must surface the Python row first among vector hits.
    assert contents
    assert "Python" in contents[0]


@pytest.mark.unit
def test_orchestrator_skips_reranker_when_disabled() -> None:
    """Default ``reranker_cfg=None`` → behaves exactly like before."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    class _StubStore:
        def get(self, mid: str) -> Memory | None:
            return Memory(id=mid, content=f"row {mid}", tags=[])

    class _StubVec:
        def search(self, q: str, *, limit: int = 50, namespace: str | None = None,
                   as_of: Any = None, time_window: Any = None) -> list[dict]:
            return [
                {"memory_id": "m1", "distance": 0.20},
                {"memory_id": "m2", "distance": 0.40},
            ]

    orch = RetrievalOrchestrator(store=_StubStore(), vector_store=_StubVec())
    # No reranker_cfg attribute → recall path is unchanged.
    out = orch.recall("anything", token_budget=2000)
    assert len(out) >= 1


# ── YAML loader integration ───────────────────────────────────────────


def _yaml_with_reranker(tmp_path, monkeypatch, **reranker) -> None:
    """Write a minimal config with reranker and reset the stack cache."""
    import yaml

    cfg = {
        "stack": {
            "postgres": {"url": "postgresql://postgres@localhost/x"},
            "neo4j": {
                "url": "bolt://localhost:7687",
                "auth": {"username": "neo4j", "password": "test"},
                "database": "neo4j",
            },
            "embedder": {
                "provider": "voyage", "model": "voyage-4", "dimensions": 1024,
            },
            "llm": {"provider": "openrouter"},
            "models": {
                "answerer": "openai/gpt-5.4-mini",
                "judge": "openai/gpt-5.5",
                "extraction": "openai/gpt-5.4-mini",
                "distill": "openai/gpt-5.4-mini",
                "verifier": "anthropic/claude-sonnet-4-6",
                "planner": "anthropic/claude-opus-4.7",
                "benchmark_default": "openai/gpt-5.4-mini",
            },
            "budget": 4000,
            "parallel": 2,
        },
    }
    if reranker:
        cfg["stack"]["retrieval"] = {"reranker": reranker}

    p = tmp_path / "attestor.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ATTESTOR_CONFIG", str(p))
    from attestor import config as _c
    _c.reset_stack()


@pytest.mark.unit
def test_yaml_default_reranker_is_disabled(tmp_path, monkeypatch) -> None:
    """No `reranker:` block → cfg.enabled defaults to False."""
    _yaml_with_reranker(tmp_path, monkeypatch)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    rr = s.retrieval.reranker
    assert rr.enabled is False
    assert rr.provider == "bge"
    assert rr.top_k == 10
    assert rr.top_n_input == 50


@pytest.mark.unit
def test_yaml_loads_reranker_provider(tmp_path, monkeypatch) -> None:
    _yaml_with_reranker(
        tmp_path, monkeypatch,
        enabled=True, provider="cohere", top_k=20, top_n_input=100,
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    rr = s.retrieval.reranker
    assert rr.enabled is True
    assert rr.provider == "cohere"
    assert rr.top_k == 20
    assert rr.top_n_input == 100


@pytest.mark.unit
def test_yaml_unknown_provider_rejected(tmp_path, monkeypatch) -> None:
    """Unknown provider crashes loud at YAML load time, not later."""
    _yaml_with_reranker(
        tmp_path, monkeypatch,
        enabled=True, provider="not-a-provider",
    )
    from attestor.config import get_stack
    with pytest.raises(SystemExit, match="reranker"):
        get_stack(strict=False)
