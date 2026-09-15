"""DenseEmbedder, SparseEncoder, and Reranker implementations via fastembed (local ONNX models).

Structurally implement the matching `core.protocols` Protocols (duck-typed — adapters do not
inherit from Protocol classes).
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from obsidian_local_rag.core.protocols import SparseVector
from obsidian_local_rag.domain.models import ScoredChunk


class FastEmbedDenseEmbedder:
    """`DenseEmbedder` backed by fastembed's local ONNX `TextEmbedding`.

    Models are downloaded on first use and cached under `cache_dir`; construction does not
    trigger a download (model loading is deferred to the first `embed` call).
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [vector.tolist() for vector in model.embed(texts)]


class FastEmbedSparseEncoder:
    """`SparseEncoder` backed by fastembed's `SparseTextEmbedding` (default: `Qdrant/bm25`)."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: SparseTextEmbedding | None = None

    def _ensure_model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
        return self._model

    def encode(self, texts: list[str]) -> list[SparseVector]:
        model = self._ensure_model()
        return [
            (embedding.indices.tolist(), embedding.values.tolist())
            for embedding in model.embed(texts)
        ]


class FastEmbedReranker:
    """`Reranker` backed by fastembed's local ONNX `TextCrossEncoder`.

    Invoked via `asyncio.to_thread` by callers (`core.rerank.rerank_candidates`); this class
    itself stays synchronous.
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None

    def _ensure_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._ensure_model()
        documents = [candidate.chunk.text for candidate in candidates]
        scores = list(model.rerank(query, documents))
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [
            ScoredChunk(chunk=candidate.chunk, score=float(score), source="rerank")
            for candidate, score in ranked[:top_n]
        ]
