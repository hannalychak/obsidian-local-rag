"""Obsidian-style link target resolution."""

from __future__ import annotations

from collections.abc import Mapping

from obsidian_local_rag.domain.models import Link, Note


def _basename(note_id: str) -> str:
    return note_id.rsplit("/", maxsplit=1)[-1]


def _pick_shortest_then_lexicographic(candidates: list[str]) -> str:
    """Ambiguous-match tie-break: fewest characters first, then lexicographic order."""
    return min(candidates, key=lambda note_id: (len(note_id), note_id))


def resolve_links(links: list[Link], notes_by_id: Mapping[str, Note]) -> list[Link]:
    """Resolve each `Link.target_raw` to a `note_id`, mirroring Obsidian's resolution order:

    1. exact vault-relative path match
    2. case-insensitive basename match
    3. frontmatter `aliases` match

    Ambiguous basename matches resolve to the shortest path, ties broken lexicographically.
    Unresolved targets get `target_note_id=None`, `resolved=False` (a "ghost" reference).

    Complexity: O(len(links) * log(len(notes_by_id))) with a basename/alias index built once.
    """
    basename_index: dict[str, list[str]] = {}
    alias_index: dict[str, list[str]] = {}
    for note_id, note in notes_by_id.items():
        basename_index.setdefault(_basename(note_id).lower(), []).append(note_id)
        for alias in note.aliases:
            alias_index.setdefault(alias.lower(), []).append(note_id)

    resolved_links: list[Link] = []
    for link in links:
        target_note_id: str | None = None

        if link.target_raw in notes_by_id:
            target_note_id = link.target_raw
        else:
            basename_matches = basename_index.get(link.target_raw.lower())
            if basename_matches:
                target_note_id = _pick_shortest_then_lexicographic(basename_matches)
            else:
                alias_matches = alias_index.get(link.target_raw.lower())
                if alias_matches:
                    target_note_id = _pick_shortest_then_lexicographic(alias_matches)

        resolved_links.append(
            Link(
                source_note_id=link.source_note_id,
                target_raw=link.target_raw,
                target_note_id=target_note_id,
                alias=link.alias,
                heading=link.heading,
                block_id=link.block_id,
                kind=link.kind,
                resolved=target_note_id is not None,
            )
        )

    return resolved_links
