"""Chunk size control: paragraph/sentence splitting with overlap, small-section merging."""

from __future__ import annotations

from obsidian_local_rag.core.chunking.markdown import Section


def split_oversized_section(
    section: Section, max_tokens: int, overlap_tokens: int
) -> list[Section]:
    """Split `section` so no resulting piece exceeds `max_tokens`.

    Splits on paragraph boundaries first, falling back to sentence boundaries; consecutive pieces
    overlap by `overlap_tokens`. Never splits inside a fenced code block unless the block alone
    exceeds `max_tokens`.
    """
    raise NotImplementedError


def merge_undersized_sections(sections: list[Section], min_tokens: int) -> list[Section]:
    """Merge any section under `min_tokens` into its next sibling section under the same parent
    header, preserving document order.
    """
    raise NotImplementedError
