"""Edge cases for Reciprocal Rank Fusion (CLAUDE.md, section 3.8)."""

from __future__ import annotations

from pathlib import Path

from obsidian_local_rag.core.retrieval.fusion import reciprocal_rank_fusion
from obsidian_local_rag.domain.models import Chunk, ScoredChunk


def _scored_chunk(chunk_id: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        note_id="note",
        path=Path("note.md"),
        header_path=(),
        text="text",
        start_line=1,
        end_line=1,
        token_estimate=1,
        tags=(),
        folder="",
        created=None,
    )
    return ScoredChunk(chunk=chunk, score=score, source="dense")


def test_rrf_missing_doc_contributes_zero() -> None:
    """A chunk present in only one ranked list still gets fused, contributing 0 for the list(s)
    it's absent from.
    """
    dense = [_scored_chunk("a", 0.9), _scored_chunk("b", 0.5)]
    sparse = [_scored_chunk("a", 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, candidate_k=50)
    assert {c.chunk.chunk_id for c in fused} == {"a", "b"}


def test_rrf_deduplicates_by_chunk_id() -> None:
    dense = [_scored_chunk("a", 0.9)]
    sparse = [_scored_chunk("a", 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, candidate_k=50)
    assert len(fused) == 1


def test_rrf_tie_break_by_best_rank_then_chunk_id() -> None:
    """Equal fused scores break ties by best single rank ascending, then `chunk_id` ascending."""
    list_a = [_scored_chunk("z", 1.0), _scored_chunk("a", 0.5)]
    fused = reciprocal_rank_fusion([list_a], k=60, candidate_k=50)
    assert [c.chunk.chunk_id for c in fused] == ["z", "a"]


def test_rrf_only_considers_top_candidate_k_of_each_list() -> None:
    dense = [_scored_chunk(str(i), 1.0 - i * 0.01) for i in range(100)]
    fused = reciprocal_rank_fusion([dense], k=60, candidate_k=10)
    assert len(fused) == 10
