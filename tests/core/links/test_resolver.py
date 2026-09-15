"""Edge cases for Obsidian-style link resolution (CLAUDE.md, section 3.7)."""

from __future__ import annotations

from pathlib import Path

from obsidian_local_rag.core.links.resolver import resolve_links
from obsidian_local_rag.domain.models import Link, Note


def _note(note_id: str, aliases: tuple[str, ...] = ()) -> Note:
    return Note(
        note_id=note_id,
        path=Path(f"{note_id}.md"),
        title=note_id.rsplit("/", maxsplit=1)[-1],
        frontmatter={},
        body="",
        content_hash="hash",
        tags=(),
        aliases=aliases,
        created=None,
        modified=None,
    )


def _link(target_raw: str) -> Link:
    return Link(
        source_note_id="source",
        target_raw=target_raw,
        target_note_id=None,
        alias=None,
        heading=None,
        block_id=None,
        kind="link",
        resolved=False,
    )


def test_exact_path_match_takes_priority_over_basename() -> None:
    notes = {
        "folder/Target": _note("folder/Target"),
        "other/Target": _note("other/Target"),
    }
    resolved = resolve_links([_link("folder/Target")], notes)
    assert resolved[0].target_note_id == "folder/Target"


def test_case_insensitive_basename_match() -> None:
    notes = {"folder/Target": _note("folder/Target")}
    resolved = resolve_links([_link("target")], notes)
    assert resolved[0].target_note_id == "folder/Target"


def test_alias_match_resolves_link() -> None:
    notes = {"folder/Target": _note("folder/Target", aliases=("Friendly Name",))}
    resolved = resolve_links([_link("Friendly Name")], notes)
    assert resolved[0].target_note_id == "folder/Target"


def test_ambiguous_basename_resolves_to_shortest_path_then_lexicographic() -> None:
    notes = {
        "Duplicate Folder A/Duplicate Name": _note("Duplicate Folder A/Duplicate Name"),
        "Duplicate Folder B/Duplicate Name": _note("Duplicate Folder B/Duplicate Name"),
    }
    resolved = resolve_links([_link("Duplicate Name")], notes)
    assert resolved[0].target_note_id == "Duplicate Folder A/Duplicate Name"


def test_unresolved_target_becomes_ghost() -> None:
    resolved = resolve_links([_link("Nonexistent Note")], {})
    assert resolved[0].resolved is False
    assert resolved[0].target_note_id is None
