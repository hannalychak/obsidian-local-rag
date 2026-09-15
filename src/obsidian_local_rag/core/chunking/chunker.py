"""Chunk orchestration: Markdown structure + size control -> stable `Chunk` objects."""

from __future__ import annotations

import logging
from uuid import UUID, uuid5

from obsidian_local_rag.core.chunking.markdown import split_into_sections
from obsidian_local_rag.core.chunking.splitter import (
    estimate_tokens,
    merge_undersized_sections,
    split_oversized_section,
)
from obsidian_local_rag.domain.models import Chunk, Note

# Fixed namespace for chunk_id = uuid5(CHUNK_ID_NAMESPACE, ...); must never change once notes are
# indexed, or every chunk_id in every existing Qdrant collection becomes stale.
CHUNK_ID_NAMESPACE = UUID("b52d14a7-3f89-5467-9512-c3d5508b486d")

logger = logging.getLogger(__name__)


def _folder_of(note_id: str) -> str:
    return note_id.rsplit("/", maxsplit=1)[0] if "/" in note_id else ""


def chunk_note(note: Note, max_tokens: int, min_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Chunk `note` into header-aware, size-controlled `Chunk`s.

    Precondition: `max_tokens` must not exceed the configured dense model's max sequence length
    (validated by the caller at startup, not here).

    Behavior:
    - Embedded text (used only for embedding, never stored/displayed) is the breadcrumb
      `"{note title} > {H1} > {H2}"` + newline + chunk text; `Chunk.text` excludes the breadcrumb.
    - `chunk_id = uuid5(CHUNK_ID_NAMESPACE, f"{note.note_id}::{'/'.join(header_path)}::{ordinal}")`.
    - An empty note produces zero chunks (logged at debug level).
    - A header-less note is chunked by paragraphs with the same size limits.

    Complexity: O(number of lines in the note body).
    """
    sections = split_into_sections(note.body)
    if not sections:
        logger.debug("Note %s is empty; producing zero chunks", note.note_id)
        return []

    sections = merge_undersized_sections(sections, min_tokens)

    sized_sections = [
        piece
        for section in sections
        for piece in split_oversized_section(section, max_tokens, overlap_tokens)
    ]

    folder = _folder_of(note.note_id)
    ordinal_by_header_path: dict[tuple[str, ...], int] = {}
    chunks: list[Chunk] = []

    for section in sized_sections:
        ordinal = ordinal_by_header_path.get(section.header_path, 0)
        ordinal_by_header_path[section.header_path] = ordinal + 1

        chunk_id = str(
            uuid5(
                CHUNK_ID_NAMESPACE,
                f"{note.note_id}::{'/'.join(section.header_path)}::{ordinal}",
            )
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                note_id=note.note_id,
                path=note.path,
                header_path=section.header_path,
                text=section.text,
                start_line=section.start_line,
                end_line=section.end_line,
                token_estimate=estimate_tokens(section.text),
                tags=note.tags,
                folder=folder,
                created=note.created,
            )
        )

    return chunks
