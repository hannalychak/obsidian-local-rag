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

    Models are downloaded on first use and cached under `cache_dir`; construction must not
    trigger a download (defer model loading until first `embed` call).
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class FastEmbedSparseEncoder:
    """`SparseEncoder` backed by fastembed's `SparseTextEmbedding` (default: `Qdrant/bm25`)."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: SparseTextEmbedding | None = None

    def encode(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError


class FastEmbedReranker:
    """`Reranker` backed by fastembed's local ONNX `TextCrossEncoder`.

    Invoked via `asyncio.to_thread` by callers (`core.rerank.rerank_candidates`); this class
    itself stays synchronous.
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        raise NotImplementedError
