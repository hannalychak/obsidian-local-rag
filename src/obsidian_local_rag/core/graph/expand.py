"""Pure BFS graph expansion with hub-decay scoring."""

from __future__ import annotations

from obsidian_local_rag.core.graph.builder import NoteGraph


def expand(
    graph: NoteGraph,
    seeds: dict[str, float],
    max_depth: int,
    max_nodes: int,
    decay: float,
    hub_degree_threshold: int,
) -> list[tuple[str, float, int]]:
    """BFS-expand from `seeds` (note_id -> seed score) over both outlinks and backlinks.

    Returns `(note_id, score, depth)` tuples, seeds excluded, ghost nodes skipped.

    Rules:
    - A visited set makes this cycle-safe.
    - `score = seed_score * decay**depth / log2(2 + degree(note_id))`, suppressing hub notes
      (daily notes, MOCs, index pages).
    - Nodes whose degree exceeds `hub_degree_threshold` may be included in the result but their
      neighbors are never enqueued (no expansion *through* hubs).
    - Hard stop once `max_nodes` results have been produced.

    Complexity: O(V_visited + E_visited), bounded by `max_nodes`.
    """
    raise NotImplementedError
