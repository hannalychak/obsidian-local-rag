"""Obsidian-style link target resolution."""

from __future__ import annotations

from collections.abc import Mapping

from obsidian_local_rag.domain.models import Link, Note


def resolve_links(links: list[Link], notes_by_id: Mapping[str, Note]) -> list[Link]:
    """Resolve each `Link.target_raw` to a `note_id`, mirroring Obsidian's resolution order:

    1. exact vault-relative path match
    2. case-insensitive basename match
    3. frontmatter `aliases` match

    Ambiguous basename matches resolve to the shortest path, ties broken lexicographically.
    Unresolved targets get `target_note_id=None`, `resolved=False` (a "ghost" reference).

    Complexity: O(len(links) * log(len(notes_by_id))) with a basename/alias index built once.
    """
    raise NotImplementedError
