"""Pure BFS graph expansion with hub-decay scoring."""

from __future__ import annotations

import math
from collections import deque

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
      neighbors are never enqueued (no expansion *through* hubs). Seeds are exempt from this
      throttle — they are the caller's deliberately chosen starting points, not nodes the BFS
      wandered into, so a well-connected seed still expands normally.
    - Hard stop once `max_nodes` results have been produced.

    Complexity: O(V_visited + E_visited), bounded by `max_nodes`.
    """
    visited: set[str] = set(seeds)
    queue: deque[tuple[str, float, int]] = deque(
        (note_id, score, 0) for note_id, score in seeds.items()
    )
    results: list[tuple[str, float, int]] = []

    while queue and len(results) < max_nodes:
        note_id, seed_score, depth = queue.popleft()

        if depth > 0 and not graph.is_ghost(note_id):
            degree = graph.degree(note_id)
            score = seed_score * (decay**depth) / math.log2(2 + degree)
            results.append((note_id, score, depth))
            if len(results) >= max_nodes:
                break

        if depth >= max_depth:
            continue
        if depth > 0 and graph.degree(note_id) > hub_degree_threshold:
            continue

        for neighbor_id in graph.neighbors(note_id):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, seed_score, depth + 1))

    return results
