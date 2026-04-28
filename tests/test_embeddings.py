"""Tests for the shared embedding-provider layer (pure unit tests, no API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from attestor.store.embeddings import (
    AzureOpenAIEmbeddingProvider,
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    VertexAIEmbeddingProvider,
    clear_embedding_cache,
    get_embedding_provider,
)


class TestEmbeddingProviderProtocol:
    def test_protocol_requires_dimension(self):
        """Providers without `dimension` do not satisfy the protocol."""

        class BadProvider:
            @property
            def provider_name(self) -> str:  # missing .dimension
                return "bad"

            def embed(self, text):
                return []

            def embed_batch(self, texts):
                return []

        assert not isinstance(BadProvider(), EmbeddingProvider)


class TestGetEmbeddingProvider:
    def setup_method(self):
        clear_embedding_cache()

    def teardown_method(self):
        # Clear the module-level cached provider so MagicMock embedders
        # used in these tests don't leak into subsequent suites that
        # boot a real PostgresBackend (where the embedder/schema dim
        # guard would otherwise fire against a leaked stub).
        clear_embedding_cache()

    def test_raises_without_any_key(self):
        """No local ollama, no cloud creds, no OpenAI key → explicit error."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("attestor.store.embeddings._try_ollama", return_value=None):
                with patch("attestor.store.embeddings._try_openai", return_value=None):
                    with pytest.raises(RuntimeError, match="No embedding provider"):
                        get_embedding_provider()

    def test_openai_tried_when_key_set(self):
        """When OPENAI_API_KEY is set and Ollama unavailable, OpenAI is used."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.dimension = 1536

        with patch("attestor.store.embeddings._try_ollama", return_value=None):
            with patch("attestor.store.embeddings._try_openai", return_value=mock_provider):
                provider = get_embedding_provider()
                assert provider.provider_name == "openai"

    def test_preferred_bedrock_tried_first(self):
        """When preferred='bedrock', should try bedrock before openai."""
        mock_bedrock = MagicMock()
        mock_bedrock.provider_name = "bedrock"
        mock_bedrock.dimension = 1024

        with patch("attestor.store.embeddings._CLOUD_PROVIDERS", {"bedrock": lambda: mock_bedrock}):
            provider = get_embedding_provider(preferred="bedrock")
            assert provider.provider_name == "bedrock"

    def test_preferred_unavailable_falls_through_to_openai(self):
        """Preferred cloud provider failing falls through to local→OpenAI path."""
        mock_openai = MagicMock()
        mock_openai.provider_name = "openai"
        mock_openai.dimension = 1536

        with patch("attestor.store.embeddings._CLOUD_PROVIDERS", {"bedrock": lambda: None}):
            with patch("attestor.store.embeddings._try_ollama", return_value=None):
                with patch("attestor.store.embeddings._try_openai", return_value=mock_openai):
                    provider = get_embedding_provider(preferred="bedrock")
                    assert provider.provider_name == "openai"


class TestOpenAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        assert hasattr(OpenAIEmbeddingProvider, "embed")
        assert hasattr(OpenAIEmbeddingProvider, "embed_batch")
        assert hasattr(OpenAIEmbeddingProvider, "dimension")
        assert hasattr(OpenAIEmbeddingProvider, "provider_name")


class TestBedrockEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        assert hasattr(BedrockEmbeddingProvider, "embed")
        assert hasattr(BedrockEmbeddingProvider, "embed_batch")
        assert hasattr(BedrockEmbeddingProvider, "dimension")
        assert hasattr(BedrockEmbeddingProvider, "provider_name")


class TestAzureOpenAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        assert hasattr(AzureOpenAIEmbeddingProvider, "embed")
        assert hasattr(AzureOpenAIEmbeddingProvider, "embed_batch")
        assert hasattr(AzureOpenAIEmbeddingProvider, "dimension")
        assert hasattr(AzureOpenAIEmbeddingProvider, "provider_name")


class TestVertexAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        assert hasattr(VertexAIEmbeddingProvider, "embed")
        assert hasattr(VertexAIEmbeddingProvider, "embed_batch")
        assert hasattr(VertexAIEmbeddingProvider, "dimension")
        assert hasattr(VertexAIEmbeddingProvider, "provider_name")
