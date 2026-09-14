"""Pure domain data models.

No I/O, no third-party SDK types, no imports from other layers (see CLAUDE.md, section 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Note:
    """A parsed vault note.

    Invariants:
    - `note_id` is the vault-relative POSIX path without extension (stable, human-readable).
    - `content_hash` is used for incremental-index diffing; two `Note`s with equal `content_hash`
      are treated as unchanged.
    - `tags`/`aliases` are normalized (deduplicated, case-preserved) from frontmatter and, per
      Phase 1 decision, unioned with inline body `#tags`.
    """

    note_id: str
    path: Path
    title: str
    frontmatter: dict[str, Any]
    body: str
    content_hash: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    created: datetime | None
    modified: datetime | None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A header-aware slice of a note's body, ready for embedding.

    Invariants:
    - `chunk_id = uuid5(NAMESPACE, f"{note_id}::{'/'.join(header_path)}::{ordinal}")`, stable
      across re-runs as long as the note's structure and chunk ordinal don't change.
    - `text` is the stored/displayed text and excludes the embedding-time breadcrumb prefix.
    - `start_line`/`end_line` are 1-based, inclusive, into the source note body.
    """

    chunk_id: str
    note_id: str
    header_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    token_estimate: int
    tags: tuple[str, ...]
    folder: str
    created: datetime | None


@dataclass(frozen=True, slots=True)
class Link:
    """A single extracted link or embed, before or after resolution.

    Invariants:
    - `kind = "embed"` for `![[...]]` targets; `kind = "link"` for `[[...]]` and `[text](x.md)`.
    - `resolved` is False and `target_note_id` is None when the target could not be matched to any
      note in the vault (a "ghost" link).
    """

    source_note_id: str
    target_raw: str
    target_note_id: str | None
    alias: str | None
    heading: str | None
    block_id: str | None
    kind: Literal["link", "embed"]
    resolved: bool


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk carrying a retrieval or rerank score.

    `source` records which stage produced this score, for debugging and for the `/sources`
    slash command; it is informational only and never affects ranking logic downstream.
    """

    chunk: Chunk
    score: float
    source: Literal["dense", "sparse", "graph", "rerank"]


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation key (e.g. `[S3]`) mapped back to the chunk and location it refers to."""

    key: str
    chunk_id: str
    path: Path
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Query-time payload filters, applied identically to both retrievers before fusion."""

    tags: tuple[str, ...] | None
    folder: str | None
    created_after: datetime | None
    created_before: datetime | None


@dataclass(frozen=True, slots=True)
class NoteContent:
    """Full content of a single note, as returned by the `read_note` agent tool."""

    note_id: str
    path: Path
    body: str
    frontmatter: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A streamed text token/fragment from the LLM."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A tool-call request emitted by the LLM mid-stream."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StopEvent:
    """Terminal event for a single `LLMClient.stream` call."""

    reason: str
    usage: dict[str, int]


StreamEvent = TextDelta | ToolCallEvent | StopEvent
