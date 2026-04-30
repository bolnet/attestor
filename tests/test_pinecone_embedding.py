"""Unit tests for PineconeEmbeddingProvider.

The provider talks to Pinecone Inference (cloud-only). Tests mock the
SDK so they run in CI without a real API key + without network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_embed_response(vectors: list[list[float]]) -> SimpleNamespace:
    """Shape: result.data[i].values is the embedding for input[i]."""
    return SimpleNamespace(
        data=[SimpleNamespace(values=v) for v in vectors],
    )


def _fake_pinecone_client(probe_dim: int = 1024) -> MagicMock:
    """Stub Pinecone client whose inference.embed always returns one
    vector of the requested probe dim. Override per-call via .return_value."""
    client = MagicMock()
    client.inference.embed.return_value = _fake_embed_response(
        [[0.0] * probe_dim],
    )
    return client


# ── Initialization ────────────────────────────────────────────────────


@pytest.mark.unit
def test_provider_requires_api_key(monkeypatch) -> None:
    """No PINECONE_API_KEY → RuntimeError. Don't fail silently."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    from attestor.store.embeddings import PineconeEmbeddingProvider
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        PineconeEmbeddingProvider()


@pytest.mark.unit
def test_provider_rejects_pclocal_key(monkeypatch) -> None:
    """The Pinecone-Local stub key 'pclocal' must be rejected — Local
    doesn't serve Inference, so accepting it would 404 at runtime."""
    monkeypatch.setenv("PINECONE_API_KEY", "pclocal")
    from attestor.store.embeddings import PineconeEmbeddingProvider
    with pytest.raises(RuntimeError, match="pclocal"):
        PineconeEmbeddingProvider()


@pytest.mark.unit
def test_provider_probes_dimension_at_init(monkeypatch) -> None:
    """First Pinecone call is a probe with input='dim'; provider stores
    the returned vector's length so the orchestrator's dim check sees
    the actual model dim, not the requested dim."""
    monkeypatch.setenv("PINECONE_API_KEY", "real-cloud-key")
    fake = _fake_pinecone_client(probe_dim=1024)

    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()

    assert p.dimension == 1024
    # Probe must have been issued
    fake.inference.embed.assert_called_once()
    call_kwargs = fake.inference.embed.call_args.kwargs
    assert call_kwargs["model"] == "llama-text-embed-v2"
    assert call_kwargs["inputs"] == ["dim"]
    assert call_kwargs["parameters"]["dimension"] == 1024


# ── Model + dim configuration ─────────────────────────────────────────


@pytest.mark.unit
def test_provider_honors_model_constructor_arg(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider(model="multilingual-e5-large")
    assert p.provider_name.startswith("pinecone:multilingual-e5-large")


@pytest.mark.unit
def test_provider_honors_env_model(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    monkeypatch.setenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()
    assert "llama-text-embed-v2" in p.provider_name


@pytest.mark.unit
def test_provider_supports_alternate_dimensions(monkeypatch) -> None:
    """llama-text-embed-v2 supports {384, 512, 768, 1024, 2048}."""
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=512)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider(dimensions=512)
    assert p.dimension == 512
    # Probe parameters include the requested dimension
    call_kwargs = fake.inference.embed.call_args.kwargs
    assert call_kwargs["parameters"]["dimension"] == 512


@pytest.mark.unit
def test_provider_honors_env_dimensions(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    monkeypatch.setenv("PINECONE_EMBEDDING_DIMENSIONS", "2048")
    fake = _fake_pinecone_client(probe_dim=2048)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()
    assert p.dimension == 2048


# ── embed / embed_batch ───────────────────────────────────────────────


@pytest.mark.unit
def test_embed_returns_single_vector(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()

        # Override for the actual embed call
        fake.inference.embed.return_value = _fake_embed_response(
            [[0.5] * 1024],
        )
        v = p.embed("the cto is bob")

    assert isinstance(v, list)
    assert len(v) == 1024
    assert v[0] == 0.5


@pytest.mark.unit
def test_embed_batch_preserves_order(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()

        # Three distinct vectors — order must round-trip
        fake.inference.embed.return_value = _fake_embed_response(
            [[1.0] * 1024, [2.0] * 1024, [3.0] * 1024],
        )
        vecs = p.embed_batch(["a", "b", "c"])

    assert len(vecs) == 3
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_embed_uses_passage_input_type_by_default(monkeypatch) -> None:
    """Default ``input_type='passage'`` matches Pinecone's convention
    for document-side embedding (the write path)."""
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()

        fake.inference.embed.reset_mock()
        fake.inference.embed.return_value = _fake_embed_response([[0.1] * 1024])
        p.embed("x")

    call_kwargs = fake.inference.embed.call_args.kwargs
    assert call_kwargs["parameters"]["input_type"] == "passage"


@pytest.mark.unit
def test_embed_supports_query_input_type(monkeypatch) -> None:
    """Query-side embedding uses ``input_type='query'`` per Pinecone
    docs — different vector for the same text."""
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider(input_type="query")

        fake.inference.embed.reset_mock()
        fake.inference.embed.return_value = _fake_embed_response([[0.1] * 1024])
        p.embed("who is the cto?")

    call_kwargs = fake.inference.embed.call_args.kwargs
    assert call_kwargs["parameters"]["input_type"] == "query"


# ── Provider name ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_provider_name_includes_model_and_dim(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "real")
    fake = _fake_pinecone_client(probe_dim=1024)
    with patch("pinecone.Pinecone", return_value=fake):
        from attestor.store.embeddings import PineconeEmbeddingProvider
        p = PineconeEmbeddingProvider()
    assert p.provider_name == "pinecone:llama-text-embed-v2@1024d"
