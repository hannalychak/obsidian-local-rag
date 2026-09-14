"""Reranking orchestration: dedupe candidates, then batched cross-encoder scoring."""

from __future__ import annotations

from obsidian_local_rag.core.protocols import Reranker
from obsidian_local_rag.domain.models import ScoredChunk


async def rerank_candidates(
    query: str, candidates: list[ScoredChunk], reranker: Reranker, rerank_n: int, final_k: int
) -> list[ScoredChunk]:
    """Deduplicate `candidates` by `chunk_id`, rerank the top `rerank_n` against `query`, and
    return the top `final_k`.

    Cross-encoder inference is CPU-bound; invoked via `asyncio.to_thread` from this async
    function, batched rather than one call per candidate.
    """
    raise NotImplementedError
