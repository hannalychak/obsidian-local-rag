"""Reciprocal Rank Fusion (RRF) — a pure function, deliberately not Qdrant's built-in fusion."""

from __future__ import annotations

from obsidian_local_rag.domain.models import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int, candidate_k: int
) -> list[ScoredChunk]:
    """Fuse multiple ranked candidate lists (e.g. dense, sparse) via RRF.

    `fused(d) = sum(1 / (k + rank_i(d)) for each list i containing d)`, ranks 1-based, computed
    over the top `candidate_k` of each input list. A document missing from a list contributes 0
    for that list. Deduplicated by `chunk_id`.

    Deterministic tie-breaking: (fused score desc, best single rank asc, chunk_id asc).

    Complexity: O(n log n) in total candidates.
    """
    fused_scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    representative: dict[str, ScoredChunk] = {}

    for ranked_list in ranked_lists:
        for rank, scored_chunk in enumerate(ranked_list[:candidate_k], start=1):
            chunk_id = scored_chunk.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                best_rank[chunk_id] = rank
            representative.setdefault(chunk_id, scored_chunk)

    ordered_ids = sorted(
        fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], best_rank[chunk_id], chunk_id)
    )

    return [
        ScoredChunk(
            chunk=representative[chunk_id].chunk,
            score=fused_scores[chunk_id],
            source=representative[chunk_id].source,
        )
        for chunk_id in ordered_ids
    ]
