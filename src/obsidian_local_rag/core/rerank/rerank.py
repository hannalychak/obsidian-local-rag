"""Reranking orchestration: dedupe candidates, then batched cross-encoder scoring."""

from __future__ import annotations

import asyncio

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
    seen: set[str] = set()
    deduped: list[ScoredChunk] = []
    for candidate in candidates:
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(candidate)

    top_candidates = deduped[:rerank_n]
    if not top_candidates:
        return []

    return await asyncio.to_thread(reranker.rerank, query, top_candidates, final_k)
