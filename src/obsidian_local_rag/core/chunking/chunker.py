"""Chunk orchestration: Markdown structure + size control -> stable `Chunk` objects."""

from __future__ import annotations

from uuid import UUID

from obsidian_local_rag.domain.models import Chunk, Note

# Fixed namespace for chunk_id = uuid5(CHUNK_ID_NAMESPACE, ...); must never change once notes are
# indexed, or every chunk_id in every existing Qdrant collection becomes stale.
CHUNK_ID_NAMESPACE = UUID("b52d14a7-3f89-5467-9512-c3d5508b486d")


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
    raise NotImplementedError
