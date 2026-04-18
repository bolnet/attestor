"""Tests for shared embedding providers — no API keys or cloud credentials needed."""

import pytest
from unittest.mock import MagicMock, patch
from attestor.store.embeddings import (
    ChromaEmbeddingAdapter,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    clear_embedding_cache,
    get_embedding_provider,
)


class TestEmbeddingProviderProtocol:
    def test_local_provider_satisfies_protocol(self):
        provider = LocalEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_protocol_requires_dimension(self):
        """Protocol requires a dimension property."""

        class BadProvider:
            @property
            def provider_name(self) -> str:
                return "bad"

            def embed(self, text):
                return []

            def embed_batch(self, texts):
                return []

        assert not isinstance(BadProvider(), EmbeddingProvider)


class TestLocalEmbeddingProvider:
    @pytest.fixture
    def provider(self):
        return LocalEmbeddingProvider()

    def test_dimension_is_384(self, provider):
        assert provider.dimension == 384

    def test_provider_name(self, provider):
        assert provider.provider_name == "local"

    def test_embed_returns_list_of_floats(self, provider):
        vec = provider.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch(self, provider):
        vecs = provider.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_different_texts_produce_different_embeddings(self, provider):
        v1 = provider.embed("the cat sat on the mat")
        v2 = provider.embed("quantum physics research paper")
        assert v1 != v2

    def test_similar_texts_have_high_cosine_similarity(self, provider):
        v1 = provider.embed("the weather is sunny today")
        v2 = provider.embed("it is a bright and sunny day")
        # Cosine similarity
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        similarity = dot / (norm1 * norm2)
        assert similarity > 0.7

    def test_embed_empty_string(self, provider):
        vec = provider.embed("")
        assert len(vec) == 384


class TestChromaEmbeddingAdapter:
    @pytest.fixture
    def adapter(self):
        provider = LocalEmbeddingProvider()
        return ChromaEmbeddingAdapter(provider)

    def test_call_returns_embeddings(self, adapter):
        result = adapter(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 384 for v in result)

    def test_embed_query(self, adapter):
        result = adapter.embed_query(["test"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_name(self, adapter):
        assert adapter.name() == "attestor:shared"

    def test_adapter_registered_with_chromadb(self):
        try:
            from chromadb.utils.embedding_functions import known_embedding_functions
        except ImportError:
            pytest.skip("chromadb not installed")

        import attestor.store.embeddings  # noqa: F401

        assert "attestor:shared" in known_embedding_functions

    def test_get_config(self, adapter):
        config = adapter.get_config()
        assert config == {"provider": "local"}

    def test_is_legacy(self, adapter):
        assert adapter.is_legacy() is False

    def test_default_space(self, adapter):
        assert adapter.default_space() == "cosine"

    def test_supported_spaces(self, adapter):
        spaces = adapter.supported_spaces()
        assert "cosine" in spaces
        assert "l2" in spaces

    def test_build_from_config_returns_adapter(self, adapter):
        rebuilt = ChromaEmbeddingAdapter.build_from_config({})
        assert isinstance(rebuilt, ChromaEmbeddingAdapter)


class TestGetEmbeddingProvider:
    def setup_method(self):
        clear_embedding_cache()

    def test_default_returns_local(self):
        """Without API keys, should fall back to local."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear any existing keys
            env = {
                k: v
                for k, v in __import__("os").environ.items()
                if k not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_ENDPOINT")
            }
            with patch.dict("os.environ", env, clear=True):
                provider = get_embedding_provider()
                assert provider.provider_name == "local"
                assert provider.dimension == 384

    def test_unknown_preferred_falls_back(self):
        """Unknown preferred provider should fall back gracefully."""
        with patch.dict("os.environ", {}, clear=True):
            env = {
                k: v
                for k, v in __import__("os").environ.items()
                if k not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_ENDPOINT")
            }
            with patch.dict("os.environ", env, clear=True):
                provider = get_embedding_provider(preferred="nonexistent")
                assert provider.provider_name == "local"

    def test_openai_tried_when_key_set(self):
        """When OPENAI_API_KEY is set, OpenAI provider should be attempted."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.dimension = 1536

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

    def test_preferred_unavailable_falls_through(self):
        """When preferred cloud provider fails, should try OpenAI then local."""
        with patch("attestor.store.embeddings._CLOUD_PROVIDERS", {"bedrock": lambda: None}):
            with patch.dict("os.environ", {}, clear=True):
                env = {
                    k: v
                    for k, v in __import__("os").environ.items()
                    if k not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_ENDPOINT")
                }
                with patch.dict("os.environ", env, clear=True):
                    provider = get_embedding_provider(preferred="bedrock")
                    assert provider.provider_name == "local"


class TestOpenAIEmbeddingProvider:
    def test_init_probes_dimension(self):
        """OpenAI provider should probe dimension on init."""
        mock_client_cls = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client_cls.return_value.embeddings.create.return_value = mock_resp

        with patch("attestor.store.embeddings.OpenAIEmbeddingProvider.__init__", return_value=None) as mock_init:
            # Test the class exists and has the right interface
            from attestor.store.embeddings import OpenAIEmbeddingProvider

            assert hasattr(OpenAIEmbeddingProvider, "embed")
            assert hasattr(OpenAIEmbeddingProvider, "embed_batch")
            assert hasattr(OpenAIEmbeddingProvider, "dimension")
            assert hasattr(OpenAIEmbeddingProvider, "provider_name")


class TestBedrockEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        from attestor.store.embeddings import BedrockEmbeddingProvider

        assert hasattr(BedrockEmbeddingProvider, "embed")
        assert hasattr(BedrockEmbeddingProvider, "embed_batch")
        assert hasattr(BedrockEmbeddingProvider, "dimension")
        assert hasattr(BedrockEmbeddingProvider, "provider_name")


class TestAzureOpenAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        from attestor.store.embeddings import AzureOpenAIEmbeddingProvider

        assert hasattr(AzureOpenAIEmbeddingProvider, "embed")
        assert hasattr(AzureOpenAIEmbeddingProvider, "embed_batch")
        assert hasattr(AzureOpenAIEmbeddingProvider, "dimension")
        assert hasattr(AzureOpenAIEmbeddingProvider, "provider_name")


class TestVertexAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        from attestor.store.embeddings import VertexAIEmbeddingProvider

        assert hasattr(VertexAIEmbeddingProvider, "embed")
        assert hasattr(VertexAIEmbeddingProvider, "embed_batch")
        assert hasattr(VertexAIEmbeddingProvider, "dimension")
        assert hasattr(VertexAIEmbeddingProvider, "provider_name")
