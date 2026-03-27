"""Shared embedding providers — single source of truth for all backends.

Fallback chain (each level tried only if the previous is unavailable):
    1. Cloud-native (Bedrock / Azure OpenAI / Vertex AI) — if credentials present
    2. OpenAI / OpenRouter text-embedding-3-small — if API key set
    3. Local sentence-transformers all-MiniLM-L6-v2 — always available

Usage:
    provider = get_embedding_provider()          # auto-detect best available
    provider = get_embedding_provider("bedrock") # prefer specific cloud provider
    vec = provider.embed("hello world")          # -> List[float]
    vecs = provider.embed_batch(["a", "b"])      # -> List[List[float]]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("agent_memory")


# ═══════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface for embedding providers."""

    @property
    def dimension(self) -> int: ...

    @property
    def provider_name(self) -> str: ...

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete Providers
# ═══════════════════════════════════════════════════════════════════════


class OpenAIEmbeddingProvider:
    """OpenAI or OpenRouter text-embedding-3-small (1536D)."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        # Probe dimension with a test embedding
        resp = self._client.embeddings.create(input=["dim"], model=self._model)
        self._dimension = len(resp.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]


class BedrockEmbeddingProvider:
    """AWS Bedrock Titan Text Embeddings v2 (1024D)."""

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: str = "amazon.titan-embed-text-v2:0",
    ) -> None:
        import boto3
        import json as _json

        self._json = _json
        session = boto3.Session(region_name=region)
        self._client = session.client("bedrock-runtime")
        self._model_id = model_id
        # Probe dimension
        test = self._raw_embed("dim")
        self._dimension = len(test)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _raw_embed(self, text: str) -> List[float]:
        body = self._json.dumps({"inputText": text})
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = self._json.loads(resp["body"].read())
        return result["embedding"]

    def embed(self, text: str) -> List[float]:
        return self._raw_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._raw_embed(t) for t in texts]


class AzureOpenAIEmbeddingProvider:
    """Azure OpenAI text-embedding-3-small (1536D)."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: str = "text-embedding-3-small",
    ) -> None:
        from openai import AzureOpenAI

        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint required")

        kwargs: Dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": "2024-02-01",
        }

        if api_key:
            kwargs["api_key"] = api_key
        else:
            api_key_env = os.environ.get("AZURE_OPENAI_API_KEY")
            if api_key_env:
                kwargs["api_key"] = api_key_env
            else:
                # Managed Identity via azure-identity
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                kwargs["azure_ad_token_provider"] = token_provider

        self._client = AzureOpenAI(**kwargs)
        self._deployment = deployment
        # Probe dimension
        resp = self._client.embeddings.create(input=["dim"], model=self._deployment)
        self._dimension = len(resp.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self._deployment)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._deployment)
        return [d.embedding for d in resp.data]


class VertexAIEmbeddingProvider:
    """Google Vertex AI text-embedding-005 (768D)."""

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        model: str = "text-embedding-005",
    ) -> None:
        from google.cloud import aiplatform

        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        aiplatform.init(project=project, location=location)

        from vertexai.language_models import TextEmbeddingModel

        self._model = TextEmbeddingModel.from_pretrained(model)
        # Probe dimension
        test = self._model.get_embeddings(["dim"])
        self._dimension = len(test[0].values)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "vertex_ai"

    def embed(self, text: str) -> List[float]:
        result = self._model.get_embeddings([text])
        return result[0].values

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = self._model.get_embeddings(texts)
        return [r.values for r in results]


class LocalEmbeddingProvider:
    """Local sentence-transformers all-MiniLM-L6-v2 (384D)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        import os as _os

        from sentence_transformers import SentenceTransformer

        # Suppress safetensors LOAD REPORT (printed from Rust, bypasses Python)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        saved_stdout = _os.dup(1)
        saved_stderr = _os.dup(2)
        try:
            _os.dup2(devnull_fd, 1)
            _os.dup2(devnull_fd, 2)
            self._model = SentenceTransformer(model_name)
        finally:
            _os.dup2(saved_stdout, 1)
            _os.dup2(saved_stderr, 2)
            _os.close(devnull_fd)
            _os.close(saved_stdout)
            _os.close(saved_stderr)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "local"

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()


# ═══════════════════════════════════════════════════════════════════════
# ChromaDB Adapter
# ═══════════════════════════════════════════════════════════════════════


class ChromaEmbeddingAdapter:
    """Wraps an EmbeddingProvider to satisfy ChromaDB's EmbeddingFunction protocol.

    ChromaDB >= 1.5 expects __call__, name(), embed_query(), get_config(),
    build_from_config(). We implement the required subset.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._provider.embed_batch(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "memwright:shared"

    @staticmethod
    def build_from_config(config: dict) -> "ChromaEmbeddingAdapter":
        # Fallback — ChromaDB calls this on collection load
        provider = get_embedding_provider()
        return ChromaEmbeddingAdapter(provider)

    def get_config(self) -> dict:
        return {"provider": self._provider.provider_name}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]


# Register with ChromaDB so it can resolve persisted "memwright:shared" configs
try:
    from chromadb.utils.embedding_functions import register_embedding_function

    register_embedding_function(ChromaEmbeddingAdapter)
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


def _try_bedrock() -> Optional[EmbeddingProvider]:
    """Try to create Bedrock provider."""
    try:
        import boto3  # noqa: F401

        # Check if AWS credentials are available
        session = boto3.Session()
        if session.get_credentials() is None:
            return None
        return BedrockEmbeddingProvider()
    except Exception as e:
        logger.debug("Bedrock embeddings unavailable: %s", e)
        return None


def _try_azure_openai() -> Optional[EmbeddingProvider]:
    """Try to create Azure OpenAI provider."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        return None
    try:
        return AzureOpenAIEmbeddingProvider(endpoint=endpoint)
    except Exception as e:
        logger.debug("Azure OpenAI embeddings unavailable: %s", e)
        return None


def _try_vertex_ai() -> Optional[EmbeddingProvider]:
    """Try to create Vertex AI provider."""
    try:
        import google.auth  # noqa: F401

        return VertexAIEmbeddingProvider()
    except Exception as e:
        logger.debug("Vertex AI embeddings unavailable: %s", e)
        return None


def _try_openai() -> Optional[EmbeddingProvider]:
    """Try OpenAI or OpenRouter."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        try:
            provider = OpenAIEmbeddingProvider(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                model="openai/text-embedding-3-small",
            )
            logger.info("Using OpenAI text-embedding-3-small via OpenRouter (%dD)", provider.dimension)
            return provider
        except Exception as e:
            logger.warning("OpenRouter embeddings failed: %s", e)

    if openai_key:
        try:
            provider = OpenAIEmbeddingProvider(api_key=openai_key)
            logger.info("Using OpenAI text-embedding-3-small (%dD)", provider.dimension)
            return provider
        except Exception as e:
            logger.warning("OpenAI embeddings failed: %s", e)

    return None


_CLOUD_PROVIDERS = {
    "bedrock": _try_bedrock,
    "azure_openai": _try_azure_openai,
    "vertex_ai": _try_vertex_ai,
}

# Module-level cache — prevents ChromaDB build_from_config from reloading
# the SentenceTransformer model on every collection operation.
_cached_provider: Optional[EmbeddingProvider] = None


def clear_embedding_cache() -> None:
    """Clear the cached embedding provider (for testing)."""
    global _cached_provider
    _cached_provider = None


def get_embedding_provider(
    preferred: Optional[str] = None,
) -> EmbeddingProvider:
    """Get the best available embedding provider.

    Fallback chain:
        1. preferred cloud provider (if specified and available)
        2. OpenAI / OpenRouter (if API key set)
        3. Local sentence-transformers (always available)

    Args:
        preferred: Cloud provider hint — "bedrock", "azure_openai", "vertex_ai".
                   Tried first but falls through on failure.

    Returns:
        An EmbeddingProvider instance (cached after first call).
    """
    global _cached_provider

    # Return cached instance if no specific preference requested
    if _cached_provider is not None and preferred is None:
        return _cached_provider

    # 1. Try preferred cloud provider
    if preferred and preferred in _CLOUD_PROVIDERS:
        provider = _CLOUD_PROVIDERS[preferred]()
        if provider is not None:
            logger.info("Using %s embeddings (%dD)", provider.provider_name, provider.dimension)
            _cached_provider = provider
            return provider
        logger.debug("Preferred provider %r unavailable, trying fallbacks", preferred)

    # 2. Try OpenAI / OpenRouter
    provider = _try_openai()
    if provider is not None:
        _cached_provider = provider
        return provider

    # 3. No fallback — require OpenAI/OpenRouter or cloud provider
    raise RuntimeError(
        "No embedding provider available. Set OPENROUTER_API_KEY or OPENAI_API_KEY. "
        "Local sentence-transformers fallback has been removed."
    )
