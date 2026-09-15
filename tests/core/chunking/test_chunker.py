"""Edge cases for header-aware chunking (CLAUDE.md, sections 3.6)."""

from __future__ import annotations

from obsidian_local_rag.core.chunking.chunker import chunk_note
from obsidian_local_rag.core.chunking.markdown import split_into_sections
from obsidian_local_rag.core.chunking.splitter import (
    merge_undersized_sections,
    split_oversized_section,
)


def test_hash_inside_code_fence_is_not_header() -> None:
    body = "# Real Header\n\n```\n# not a header\n```\n\nAfter fence.\n"
    sections = split_into_sections(body)
    assert len(sections) == 1
    assert sections[0].header_path == ("Real Header",)


def test_inline_tag_without_space_is_not_header() -> None:
    body = "#tag-not-a-header\n\nSome text.\n"
    sections = split_into_sections(body)
    assert sections[0].header_path == ()


def test_headerless_note_produces_single_section() -> None:
    body = "Just a paragraph.\n\nAnother paragraph.\n"
    sections = split_into_sections(body)
    assert len(sections) == 1
    assert sections[0].header_path == ()


def test_empty_note_produces_zero_chunks() -> None:
    assert split_into_sections("") == []


def test_oversized_section_splits_with_overlap_and_never_inside_fence() -> None:
    section = split_into_sections("# H\n\n" + ("word " * 1000))[0]
    pieces = split_oversized_section(section, max_tokens=100, overlap_tokens=10)
    assert len(pieces) > 1


def test_undersized_section_merges_into_next_sibling() -> None:
    body = "# Parent\n\n## Tiny\n\nfew words\n\n## Sibling\n\nmore content here\n"
    sections = split_into_sections(body)
    merged = merge_undersized_sections(sections, min_tokens=50)
    assert all(s.header_path != ("Parent", "Tiny") for s in merged)


def test_chunk_id_is_stable_across_repeated_chunking() -> None:
    from pathlib import Path

    from obsidian_local_rag.domain.models import Note

    note = Note(
        note_id="folder/note",
        path=Path("folder/note.md"),
        title="note",
        frontmatter={},
        body="# H\n\nbody text\n",
        content_hash="abc",
        tags=(),
        aliases=(),
        created=None,
        modified=None,
    )
    first = chunk_note(note, max_tokens=512, min_tokens=64, overlap_tokens=64)
    second = chunk_note(note, max_tokens=512, min_tokens=64, overlap_tokens=64)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
