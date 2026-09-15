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
        """Build a `NoteGraph` from all notes and their already-resolved links.

        A node is added for every note in `notes`. An unresolved link's target has no `note_id`,
        so it becomes its own node keyed by the link's raw target text, with `ghost=True` — it is
        never a source of content, only a marker that something referenced but missing exists.
        Repeated links between the same ordered pair collapse into one edge with an incremented
        `count`, keeping the first-seen `kind`.
        """
        graph: nx.DiGraph = nx.DiGraph()
        for note_id, note in notes.items():
            graph.add_node(note_id, title=note.title, ghost=False, tags=note.tags)

        for link in resolved_links:
            if link.resolved and link.target_note_id is not None:
                target = link.target_note_id
            else:
                target = link.target_raw
                if target not in graph:
                    graph.add_node(target, title=target, ghost=True, tags=())

            if graph.has_edge(link.source_note_id, target):
                graph[link.source_note_id][target]["count"] += 1
            else:
                graph.add_edge(link.source_note_id, target, kind=link.kind, count=1)

        return cls(graph)

    def degree(self, note_id: str) -> int:
        """Total in+out degree of `note_id`, or 0 if the note has no edges."""
        if note_id not in self._graph:
            return 0
        return int(self._graph.degree(note_id))

    def neighbors(self, note_id: str) -> list[str]:
        """Both successors (outlinks) and predecessors (backlinks) of `note_id`."""
        if note_id not in self._graph:
            return []
        return [*self._graph.successors(note_id), *self._graph.predecessors(note_id)]

    def orphans(self) -> list[str]:
        """note_ids with zero in+out degree."""
        return [node for node, degree in self._graph.degree() if degree == 0]

    def is_ghost(self, note_id: str) -> bool:
        """True if `note_id` is a ghost node (an unresolved link target, not a real note)."""
        return bool(self._graph.nodes.get(note_id, {}).get("ghost", False))

    def ghosts(self) -> list[str]:
        """Raw target text of every unresolved link in the vault (the `stats` command's
        `unresolved_link_targets`).
        """
        return [node for node, data in self._graph.nodes(data=True) if data.get("ghost", False)]
