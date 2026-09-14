"""Edge cases for BFS graph expansion with hub-decay scoring (CLAUDE.md, section 3.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_local_rag.core.graph.builder import NoteGraph
from obsidian_local_rag.core.graph.expand import expand
from obsidian_local_rag.domain.models import Link, Note


def _note(note_id: str) -> Note:
    return Note(
        note_id=note_id,
        path=Path(f"{note_id}.md"),
        title=note_id,
        frontmatter={},
        body="",
        content_hash="hash",
        tags=(),
        aliases=(),
        created=None,
        modified=None,
    )


def _link(source: str, target: str, kind: str = "link") -> Link:
    return Link(
        source_note_id=source,
        target_raw=target,
        target_note_id=target,
        alias=None,
        heading=None,
        block_id=None,
        kind=kind,  # type: ignore[arg-type]
        resolved=True,
    )


@pytest.mark.skip(reason="not implemented")
def test_bfs_terminates_on_cycle() -> None:
    """`Cycle A` <-> `Cycle B` must not cause infinite expansion."""
    notes = {"Cycle A": _note("Cycle A"), "Cycle B": _note("Cycle B")}
    graph = NoteGraph.build(notes, [_link("Cycle A", "Cycle B"), _link("Cycle B", "Cycle A")])
    result = expand(
        graph,
        seeds={"Cycle A": 1.0},
        max_depth=10,
        max_nodes=20,
        decay=0.5,
        hub_degree_threshold=50,
    )
    assert len(result) <= 1  # only "Cycle B" reachable, visited set prevents re-visiting "Cycle A"


@pytest.mark.skip(reason="not implemented")
def test_hub_node_not_expanded_through() -> None:
    """A node whose degree exceeds `hub_degree_threshold` may appear in results, but its own
    neighbors must never be enqueued.
    """
    notes = {f"n{i}": _note(f"n{i}") for i in range(5)}
    notes["hub"] = _note("hub")
    notes["beyond_hub"] = _note("beyond_hub")
    links = [_link(f"n{i}", "hub") for i in range(5)] + [_link("hub", "beyond_hub")]
    graph = NoteGraph.build(notes, links)
    result = expand(
        graph,
        seeds={"n0": 1.0},
        max_depth=3,
        max_nodes=20,
        decay=0.5,
        hub_degree_threshold=2,
    )
    result_ids = {note_id for note_id, _score, _depth in result}
    assert "beyond_hub" not in result_ids


@pytest.mark.skip(reason="not implemented")
def test_expansion_respects_max_nodes() -> None:
    notes = {f"n{i}": _note(f"n{i}") for i in range(30)}
    links = [_link("n0", f"n{i}") for i in range(1, 30)]
    graph = NoteGraph.build(notes, links)
    result = expand(
        graph, seeds={"n0": 1.0}, max_depth=5, max_nodes=5, decay=0.9, hub_degree_threshold=100
    )
    assert len(result) <= 5


@pytest.mark.skip(reason="not implemented")
def test_seeds_excluded_from_results() -> None:
    notes = {"a": _note("a"), "b": _note("b")}
    graph = NoteGraph.build(notes, [_link("a", "b")])
    result = expand(
        graph, seeds={"a": 1.0}, max_depth=2, max_nodes=10, decay=0.5, hub_degree_threshold=50
    )
    assert "a" not in {note_id for note_id, _score, _depth in result}
