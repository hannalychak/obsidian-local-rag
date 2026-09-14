"""Knowledge graph construction from resolved links."""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx

from obsidian_local_rag.domain.models import Link, Note


class NoteGraph:
    """Wraps a `networkx.DiGraph` of notes and their resolved links.

    Nodes are `note_id`s with attributes `title`, `ghost`, `tags`. Edges carry `kind` ("link" or
    "embed") and `count` (number of times that edge occurs between the same two notes).

    Rebuilt from ingest output on every run (cheap at this corpus scale); persistence is
    deliberately not implemented in v1 (CLAUDE.md: "must be justified" if added later).
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    @classmethod
    def build(cls, notes: Mapping[str, Note], resolved_links: list[Link]) -> NoteGraph:
        """Build a `NoteGraph` from all notes and their already-resolved links."""
        raise NotImplementedError

    def degree(self, note_id: str) -> int:
        """Total in+out degree of `note_id`, or 0 if the note has no edges."""
        raise NotImplementedError

    def neighbors(self, note_id: str) -> list[str]:
        """Both successors (outlinks) and predecessors (backlinks) of `note_id`."""
        raise NotImplementedError

    def orphans(self) -> list[str]:
        """note_ids with zero in+out degree."""
        raise NotImplementedError
