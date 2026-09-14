"""Corpus statistics computed from the notes/chunks DataFrames, for the `stats` CLI command."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    """
    raise NotImplementedError
