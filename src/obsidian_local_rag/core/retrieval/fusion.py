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
    raise NotImplementedError
