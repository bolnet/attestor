"""Cross-encoder reranker stage — Step 3.5 of the recall cascade.

Sits between RRF (Step 3) and graph BFS (Step 4). Consumes the top-N RRF
candidates, asks a cross-encoder (BGE / Cohere / LLM) to re-score each
``(query, candidate)`` pair, then keeps the top-K.

Per Anthropic's Sept-2024 contextual-retrieval research, a reranker is
the highest-leverage retrieval-side improvement after vector + BM25 +
RRF — empirically cited at ~+60% in retrieval quality on heterogeneous
RAG corpora. Attestor's existing graph-BFS step handles entity affinity,
which is *complementary* to relevance reranking, so we keep both.

Three providers ship in this PR — all opt-in via ``configs/attestor.yaml``
under ``retrieval.reranker``:

  - ``bge``    — ``BAAI/bge-reranker-v2-gemma`` via sentence-transformers.
                 Open weights, deterministic, lazy-loads on first call.
                 Falls back to no-op if torch / transformers /
                 sentence-transformers aren't installed.
  - ``cohere`` — ``rerank-english-v3.0`` via the cohere SDK. Requires
                 ``COHERE_API_KEY``; degrades to no-op without.
  - ``llm``    — LLM-as-reranker (Haiku 4.5 / gpt-4o-mini / etc.) via
                 the YAML-configured LiteLLM provider. Sends the question
                 + top-N candidates, parses an ordered top-K back.

All providers degrade to "return input unchanged" on any error so the
recall path never breaks.

Default OFF. Phase 1 ships disabled; bench cells in ``evals/matrix.yaml``
flip it on per ablation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

from attestor.models import RetrievalResult

logger = logging.getLogger("attestor.retrieval.reranker")


# ── Public config ─────────────────────────────────────────────────────


VALID_PROVIDERS = ("bge", "cohere", "llm")


@dataclass(frozen=True)
class RerankerCfg:
    """Reranker stage knobs.

    Configuration:
      enabled       — master switch; default False (Phase 1 ships off).
      provider      — one of ``bge`` (open-weights, deterministic),
                      ``cohere`` (premium API), ``llm`` (Haiku 4.5 etc.).
      top_k         — how many candidates survive the rerank.
      top_n_input   — how many candidates the provider sees from RRF
                      (the slice that actually matters for ranking).
      bge_model     — HF model id when provider=bge.
      cohere_model  — Cohere model id when provider=cohere.
      llm_model     — LiteLLM model id when provider=llm. ``null`` falls
                      back to ``models.benchmark_default`` at call time.
    """

    enabled: bool = False
    provider: str = "bge"
    top_k: int = 10
    top_n_input: int = 50
    bge_model: str = "BAAI/bge-reranker-v2-gemma"
    cohere_model: str = "rerank-english-v3.0"
    llm_model: str | None = None


# ── Provider protocol ─────────────────────────────────────────────────


class Reranker(Protocol):
    """Score ``len(candidates)`` floats — higher = more relevant."""

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]: ...


# ── Provider implementations ──────────────────────────────────────────


class NoOpReranker:
    """Returns uniform 0.0 scores; the wrapper short-circuits to identity."""

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        return [0.0] * len(candidates)


class BgeReranker:
    """``BAAI/bge-reranker-v2-gemma`` via sentence-transformers.CrossEncoder.

    Lazy-loads on first call (model weights are ~2GB on disk; we don't
    want to pay that cost at process start when the reranker is disabled
    or a different provider is selected). Subsequent calls reuse the
    cached encoder.
    """

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-v2-gemma") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            import sentence_transformers  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "BGE reranker needs `sentence-transformers` — install via "
                "`pip install attestor[reranker]` or "
                "`pip install sentence-transformers torch`."
            ) from e
        if sentence_transformers is None:
            raise ImportError(
                "BGE reranker: sentence_transformers is None (likely "
                "monkeypatched or partially installed)."
            )
        cross_encoder_cls = sentence_transformers.CrossEncoder
        self._model = cross_encoder_cls(self.model_name)
        return self._model

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.memory.content) for c in candidates]
        raw = model.predict(pairs)  # type: ignore[attr-defined]
        return [float(s) for s in raw]


class CohereReranker:
    """Cohere ``rerank-english-v3.0`` via the cohere SDK.

    Requires ``COHERE_API_KEY``. Raises ``RuntimeError`` when the key is
    missing so the wrapper falls back to no-op (returns input unchanged).
    """

    def __init__(self, *, model: str = "rerank-english-v3.0") -> None:
        self.model = model

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        if not candidates:
            return []
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Cohere reranker needs COHERE_API_KEY to be set; "
                "the wrapper will fall back to no-op."
            )
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Cohere reranker needs `cohere` — install via "
                "`pip install cohere`."
            ) from e
        client = cohere.Client(api_key=api_key)
        documents = [c.memory.content for c in candidates]
        response = client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        scores = [0.0] * len(candidates)
        for hit in response.results:
            scores[hit.index] = float(hit.relevance_score)
        return scores


class LlmReranker:
    """LLM-as-reranker via the YAML-configured LiteLLM provider.

    Sends the question + top-N candidates as a single prompt, asks the
    model to return a JSON array of ``{"index": N, "score": F}`` entries,
    then converts that to a flat score list aligned with input order.
    Indices the LLM omits default to ``0.0``.
    """

    _PROMPT = (
        "You are a retrieval reranker. Given a user QUESTION and a list "
        "of CANDIDATE memories, return a JSON array sorted by descending "
        "relevance to the question. Each array entry has shape:\n"
        '  {{"index": <int>, "score": <float between 0 and 1>}}\n\n'
        "Score 1.0 = the candidate directly answers the question; "
        "0.0 = irrelevant. Output ONLY the JSON array, no prose.\n\n"
        "QUESTION:\n{query}\n\n"
        "CANDIDATES:\n{candidates}\n\n"
        "JSON array:"
    )

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            from attestor.config import get_stack
            return get_stack(strict=False).models.benchmark_default
        except Exception:
            return "openai/gpt-4o-mini"

    def score(
        self, query: str, candidates: list[RetrievalResult],
    ) -> list[float]:
        if not candidates:
            return []
        model = self._resolve_model()
        block = "\n".join(
            f"[{i}] {c.memory.content}" for i, c in enumerate(candidates)
        )
        prompt = self._PROMPT.format(query=query, candidates=block)

        from attestor.llm_trace import get_client_for_model
        client, clean_model = get_client_for_model(model)
        response = client.chat.completions.create(
            model=clean_model,
            max_tokens=2000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        # Trim common code-fence wrappers some models emit.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM reranker returned non-JSON response: {text[:200]!r}",
            ) from e

        if not isinstance(parsed, list):
            raise ValueError(
                f"LLM reranker expected a JSON array, got: {type(parsed).__name__}",
            )

        scores = [0.0] * len(candidates)
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            score = entry.get("score")
            if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
                continue
            try:
                scores[idx] = float(score)
            except (TypeError, ValueError):
                continue
        return scores


# ── Factory + entry point ─────────────────────────────────────────────


def build_reranker(cfg: RerankerCfg) -> Reranker:
    """Build a reranker instance for ``cfg``.

    Returns ``NoOpReranker`` when disabled. Raises ``ValueError`` for
    unknown providers (caught by the YAML loader as ``SystemExit``).
    """
    if not cfg.enabled:
        return NoOpReranker()
    if cfg.provider == "bge":
        return BgeReranker(model_name=cfg.bge_model)
    if cfg.provider == "cohere":
        return CohereReranker(model=cfg.cohere_model)
    if cfg.provider == "llm":
        return LlmReranker(model=cfg.llm_model)
    raise ValueError(
        f"unknown reranker provider {cfg.provider!r}; "
        f"expected one of {list(VALID_PROVIDERS)}"
    )


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    *,
    cfg: RerankerCfg,
    provider: Reranker | None = None,
) -> list[RetrievalResult]:
    """Rerank ``candidates`` for ``query``; return the top-``cfg.top_k``.

    Drop-in safe — the original ``candidates`` list is never mutated.
    On any provider error, logs a warning and returns the input unchanged.

    The ``provider`` kwarg is the dependency-injection seam used by tests
    (StubReranker) and by ``RetrievalOrchestrator`` (cached singleton).
    Production code paths build a provider via ``build_reranker(cfg)``.
    """
    if not cfg.enabled:
        return candidates
    if not candidates:
        return []

    window = candidates[: cfg.top_n_input]
    rest = candidates[cfg.top_n_input:]

    if provider is None:
        try:
            provider = build_reranker(cfg)
        except Exception as e:
            logger.warning(
                "reranker: build failed (%s); falling back to no-op", e,
            )
            return candidates

    try:
        scores = provider.score(query, window)
    except Exception as e:
        logger.warning(
            "reranker: provider %s raised (%s); falling back to no-op",
            cfg.provider, e,
        )
        return candidates

    if len(scores) != len(window):
        logger.warning(
            "reranker: provider returned %d scores for %d candidates; "
            "falling back to no-op", len(scores), len(window),
        )
        return candidates

    # Pair each candidate with its score, sort by descending score, then
    # rebuild RetrievalResult with the new score so downstream stages
    # (graph BFS, MMR) see the rerank signal. ``stable=True`` preserves
    # the original RRF order for equal-scoring candidates.
    paired = list(zip(window, scores, strict=True))
    paired.sort(key=lambda p: p[1], reverse=True)

    reranked: list[RetrievalResult] = []
    for original, sc in paired[: cfg.top_k]:
        reranked.append(
            RetrievalResult(
                memory=original.memory,
                score=float(sc),
                match_source=original.match_source,
            )
        )

    # Append any candidates that exceeded ``top_n_input`` so we never
    # silently drop tail rows the orchestrator might still want for
    # graph-BFS context. They keep their original scores.
    return reranked + rest


__all__ = [
    "VALID_PROVIDERS",
    "BgeReranker",
    "CohereReranker",
    "LlmReranker",
    "NoOpReranker",
    "Reranker",
    "RerankerCfg",
    "build_reranker",
    "rerank",
]
