"""Tests for the shared embedding-provider layer (pure unit tests, no API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from attestor.store.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
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

    def test_strict_preferred_openai_raises_when_unavailable(self):
        """preferred='openai' with no OPENAI_API_KEY → raises with config-pointing message.

        Strict mode: there's no auto-fallthrough. If YAML pins openai
        and the helper can't initialize, attestor surfaces the config
        problem rather than silently picking another backend.
        """
        # Patch the dispatch dict directly — it captures fn refs at module
        # load, so monkeypatching _try_openai wouldn't affect it.
        with patch.dict(
            "attestor.store.embeddings._PROVIDER_DISPATCH",
            {"openai": lambda: None},
        ):
            with pytest.raises(RuntimeError, match="failed to initialize"):
                get_embedding_provider(preferred="openai")

    def test_strict_preferred_openai_returns_when_available(self):
        """preferred='openai' returns the openai provider when its
        helper succeeds (no fallthrough involved)."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "openai"
        mock_provider.dimension = 1536

        with patch.dict(
            "attestor.store.embeddings._PROVIDER_DISPATCH",
            {"openai": lambda: mock_provider},
        ):
            provider = get_embedding_provider(preferred="openai")
            assert provider.provider_name == "openai"

    def test_preferred_voyage_routes_to_voyage(self):
        """preferred='voyage' dispatches to the voyage helper."""
        mock_voyage = MagicMock()
        mock_voyage.provider_name = "voyage"
        mock_voyage.dimension = 1024

        with patch.dict(
            "attestor.store.embeddings._PROVIDER_DISPATCH",
            {"voyage": lambda: mock_voyage},
        ):
            provider = get_embedding_provider(preferred="voyage")
            assert provider.provider_name == "voyage"

    def test_preferred_unavailable_raises_no_fallthrough(self):
        """Preferred provider failing must RAISE — no silent fallthrough.

        This pins the strict-mode contract: YAML's stack.embedder.provider
        is authoritative; failure is surfaced, not papered over.
        """
        with patch.dict(
            "attestor.store.embeddings._PROVIDER_DISPATCH",
            {"voyage": lambda: None},
        ):
            with pytest.raises(RuntimeError, match="failed to initialize"):
                get_embedding_provider(preferred="voyage")


class TestOpenAIEmbeddingProvider:
    def test_class_exists_with_correct_interface(self):
        assert hasattr(OpenAIEmbeddingProvider, "embed")
        assert hasattr(OpenAIEmbeddingProvider, "embed_batch")
        assert hasattr(OpenAIEmbeddingProvider, "dimension")
        assert hasattr(OpenAIEmbeddingProvider, "provider_name")
