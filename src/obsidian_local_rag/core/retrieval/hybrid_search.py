"""Hybrid retrieval orchestration: dense + sparse search, fused via RRF."""

from __future__ import annotations

from obsidian_local_rag.core.protocols import DenseEmbedder, SparseEncoder, VectorStore
from obsidian_local_rag.domain.models import ScoredChunk, SearchFilters


async def hybrid_search(
    query: str,
    filters: SearchFilters | None,
    dense_embedder: DenseEmbedder,
    sparse_encoder: SparseEncoder,
    vector_store: VectorStore,
    candidate_k: int,
    rrf_k: int,
) -> list[ScoredChunk]:
    """Run dense and sparse retrieval for `query` and fuse the results.

    `filters` is applied identically to both retrievers so fusion operates over the same filtered
    universe. Embedding/encoding calls are CPU-bound; implementations should invoke them via
    `asyncio.to_thread`. Fusion is delegated to `fusion.reciprocal_rank_fusion`.
    """
    raise NotImplementedError
