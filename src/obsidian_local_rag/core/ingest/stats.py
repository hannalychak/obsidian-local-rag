"""Corpus statistics computed from the notes/chunks DataFrames, for the `stats` CLI command."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_TOP_N = 10


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Corpus-wide statistics rendered by the `stats` CLI command."""

    note_count: int
    chunk_count: int
    token_distribution: dict[str, float]
    top_tags: list[tuple[str, int]]
    top_folders: list[tuple[str, int]]
    orphan_note_ids: list[str]
    unresolved_link_targets: list[str]


def compute_corpus_stats(notes_df: pd.DataFrame, chunks_df: pd.DataFrame) -> CorpusStats:
    """Compute `CorpusStats` from the notes and chunks DataFrames.

    Pandas is the ingest & analytics plane only (CLAUDE.md, section 3.5): this function is never
    called at query time.

    Known gap: `orphan_note_ids` and `unresolved_link_targets` need the knowledge graph
    (`core.graph.builder.NoteGraph.orphans` / ghost nodes), which this function has no access to —
    it only sees the two DataFrames. They are always returned empty here. Whoever wires up
    `app.indexing_service.IndexingService.compute_stats` must compute those two fields from the
    `NoteGraph` separately and merge them into the `CorpusStats` this function returns.
    """
    top_tags: list[tuple[str, int]] = []
    top_folders: list[tuple[str, int]] = []

    if not notes_df.empty:
        if "tags" in notes_df.columns:
            tag_counts = notes_df["tags"].explode().dropna()
            top_tags = list(tag_counts.value_counts().head(_TOP_N).items())
        if "folder" in notes_df.columns:
            top_folders = list(notes_df["folder"].value_counts().head(_TOP_N).items())

    token_distribution: dict[str, float] = {}
    if not chunks_df.empty and "token_estimate" in chunks_df.columns:
        tokens = chunks_df["token_estimate"]
        token_distribution = {
            "min": float(tokens.min()),
            "max": float(tokens.max()),
            "mean": float(tokens.mean()),
            "median": float(tokens.median()),
        }

    return CorpusStats(
        note_count=len(notes_df),
        chunk_count=len(chunks_df),
        token_distribution=token_distribution,
        top_tags=top_tags,
        top_folders=top_folders,
        orphan_note_ids=[],
        unresolved_link_targets=[],
    )
