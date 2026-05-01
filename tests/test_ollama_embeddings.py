"""Local Ollama bge-m3 embedding provider tests.

Most tests use a stub `requests` module so they run without Ollama
running. One smoke test (marked live) hits a real local Ollama if it's
reachable — skipped otherwise.
"""

from __future__ import annotations

import os

import pytest

from attestor.store.embeddings import (
    OllamaEmbeddingProvider,
    _try_ollama,
    clear_embedding_cache,
    get_embedding_provider,
)


# ──────────────────────────────────────────────────────────────────────────
# Stub requests session
# ──────────────────────────────────────────────────────────────────────────


class _StubResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> dict:
        return self._payload


class _StubRequests:
    """Mocks the per-call functions of the `requests` module."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self._responses: list[_StubResponse] = []

    def queue(self, *responses: _StubResponse) -> None:
        self._responses.extend(responses)

    def post(self, url: str, *, json: dict, timeout: float) -> _StubResponse:
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            return _StubResponse({"embedding": [0.0] * 1024})
        return self._responses.pop(0)


def _provider_with_stub(monkeypatch, stub: _StubRequests) -> OllamaEmbeddingProvider:
    """Construct OllamaEmbeddingProvider whose internal `requests` is a stub."""
    p = OllamaEmbeddingProvider.__new__(OllamaEmbeddingProvider)
    p._requests = stub
    p._host = "http://localhost:11434"
    p._model = "bge-m3"
    p._timeout = 5.0
    p._dimension = 1024
    return p


# ──────────────────────────────────────────────────────────────────────────
# Defaults / config
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_default_model_is_bge_m3() -> None:
    assert OllamaEmbeddingProvider.DEFAULT_MODEL == "bge-m3"


@pytest.mark.unit
def test_default_host_is_localhost() -> None:
    assert OllamaEmbeddingProvider.DEFAULT_HOST == "http://localhost:11434"


# ──────────────────────────────────────────────────────────────────────────
# embed() / embed_batch() with stub HTTP
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_embed_posts_to_correct_endpoint(monkeypatch) -> None:
    stub = _StubRequests()
    stub.queue(_StubResponse({"embedding": [0.1] * 1024}))
    p = _provider_with_stub(monkeypatch, stub)
    v = p.embed("hi")
    assert len(v) == 1024
    assert stub.posts[0]["url"] == "http://localhost:11434/api/embeddings"
    assert stub.posts[0]["json"] == {"model": "bge-m3", "prompt": "hi"}


@pytest.mark.unit
def test_embed_raises_when_response_missing_embedding(monkeypatch) -> None:
    stub = _StubRequests()
    stub.queue(_StubResponse({"error": "model not found"}))
    p = _provider_with_stub(monkeypatch, stub)
    with pytest.raises(RuntimeError, match="missing 'embedding'"):
        p.embed("hi")


@pytest.mark.unit
def test_embed_batch_uses_batch_endpoint_first(monkeypatch) -> None:
    stub = _StubRequests()
    stub.queue(_StubResponse({"embeddings": [[0.1] * 1024, [0.2] * 1024]}))
    p = _provider_with_stub(monkeypatch, stub)
    out = p.embed_batch(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == 1024 for v in out)
    # First (and only) call hit /api/embed (the batch endpoint)
    assert stub.posts[0]["url"] == "http://localhost:11434/api/embed"


@pytest.mark.unit
def test_embed_batch_falls_back_to_per_text(monkeypatch) -> None:
    """If /api/embed errors, per-text /api/embeddings calls cover it."""
    stub = _StubRequests()
    stub.queue(
        _StubResponse({"error": "no batch"}, status=500),       # /api/embed fails
        _StubResponse({"embedding": [0.1] * 1024}),             # /api/embeddings #1
        _StubResponse({"embedding": [0.2] * 1024}),             # /api/embeddings #2
    )
    p = _provider_with_stub(monkeypatch, stub)
    out = p.embed_batch(["a", "b"])
    assert len(out) == 2
    assert out[0][0] == 0.1
    assert out[1][0] == 0.2


@pytest.mark.unit
def test_provider_name_includes_model() -> None:
    stub = _StubRequests()
    p = _provider_with_stub(None, stub)
    p._model = "nomic-embed-text"
    assert p.provider_name == "ollama:nomic-embed-text"


# ──────────────────────────────────────────────────────────────────────────
# Env var overrides
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_attestor_embedding_model_env_override(monkeypatch) -> None:
    """ATTESTOR_EMBEDDING_MODEL takes precedence over DEFAULT_MODEL."""
    monkeypatch.setenv("ATTESTOR_EMBEDDING_MODEL", "mxbai-embed-large")

    # Build a minimal provider that resolves model from env without
    # actually probing — bypass __init__'s probe.
    p = OllamaEmbeddingProvider.__new__(OllamaEmbeddingProvider)
    p._model = (
        os.environ.get("ATTESTOR_EMBEDDING_MODEL")
        or OllamaEmbeddingProvider.DEFAULT_MODEL
    )
    assert p._model == "mxbai-embed-large"


@pytest.mark.unit
def test_disable_local_embed_env_skips_probe(monkeypatch) -> None:
    monkeypatch.setenv("ATTESTOR_DISABLE_LOCAL_EMBED", "1")
    assert _try_ollama() is None


# ──────────────────────────────────────────────────────────────────────────
# Live (only if Ollama daemon up + bge-m3 pulled)
# ──────────────────────────────────────────────────────────────────────────


def _ollama_reachable() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_live_bge_m3_returns_1024d() -> None:
    """Smoke test against a real local Ollama daemon."""
    clear_embedding_cache()
    p = OllamaEmbeddingProvider()
    assert p.dimension == 1024
    v = p.embed("attestor uses local embeddings")
    assert len(v) == 1024
    assert all(isinstance(x, float) for x in v[:5])


@pytest.mark.live
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_live_get_embedding_provider_picks_ollama_when_preferred() -> None:
    """Strict dispatch — preferred='ollama' must land on the Ollama
    provider when the daemon is up. The legacy auto-detect chain (no
    `preferred=`) was removed in favor of YAML-driven selection."""
    clear_embedding_cache()
    p = get_embedding_provider(preferred="ollama")
    assert p.provider_name.startswith("ollama:")
