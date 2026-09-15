"""`obsidian-rag graph-info` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from obsidian_local_rag.app.indexing_service import load_notes_and_graph
from obsidian_local_rag.cli.rendering import print_error
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.graph.builder import NoteGraph
from obsidian_local_rag.domain.models import Note

console = Console()

_MAX_LISTED = 10


def graph_info(
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Path to the Obsidian vault.")
    ] = None,
    note: Annotated[
        str | None, typer.Option("--note", help="Show details for a single note_id.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Show knowledge-graph info: degree, neighbors, orphans, hubs."""
    try:
        settings = (
            Settings(vault_path=vault) if vault is not None else Settings()  # type: ignore[call-arg]  # required fields resolved from env/.env at runtime
        )
        notes, graph = load_notes_and_graph(settings.vault_path, settings.exclude_globs)

        if note is not None:
            _print_note_info(notes, graph, note, settings.graph_hub_degree_threshold, debug)
        else:
            _print_overview(notes, graph, settings.graph_hub_degree_threshold)
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc


def _print_overview(notes: dict[str, Note], graph: NoteGraph, hub_threshold: int) -> None:
    orphans = graph.orphans()
    ghosts = graph.ghosts()
    hubs = sorted(
        (note_id for note_id in notes if graph.degree(note_id) > hub_threshold),
        key=graph.degree,
        reverse=True,
    )

    overview = Table(title="Knowledge graph overview")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("Notes", str(len(notes)))
    overview.add_row("Orphan notes", str(len(orphans)))
    overview.add_row("Unresolved link targets (ghosts)", str(len(ghosts)))
    overview.add_row(f"Hub notes (degree > {hub_threshold})", str(len(hubs)))
    console.print(overview)

    if hubs:
        hub_table = Table(title="Hub notes")
        hub_table.add_column("Note")
        hub_table.add_column("Degree", justify="right")
        for note_id in hubs[:_MAX_LISTED]:
            hub_table.add_row(note_id, str(graph.degree(note_id)))
        console.print(hub_table)

    if orphans:
        shown = ", ".join(sorted(orphans)[:_MAX_LISTED])
        suffix = f" (+{len(orphans) - _MAX_LISTED} more)" if len(orphans) > _MAX_LISTED else ""
        console.print(f"[dim]Orphans: {shown}{suffix}[/dim]")

    if ghosts:
        shown = ", ".join(sorted(ghosts)[:_MAX_LISTED])
        suffix = f" (+{len(ghosts) - _MAX_LISTED} more)" if len(ghosts) > _MAX_LISTED else ""
        console.print(f"[dim]Unresolved: {shown}{suffix}[/dim]")


def _print_note_info(
    notes: dict[str, Note], graph: NoteGraph, note_id: str, hub_threshold: int, debug: bool
) -> None:
    if note_id not in notes and not graph.is_ghost(note_id):
        print_error(f"No such note: {note_id!r}", debug=debug)
        raise typer.Exit(code=1)

    degree = graph.degree(note_id)
    neighbors = sorted(graph.neighbors(note_id))

    table = Table(title=f"Graph info: {note_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Degree", str(degree))
    table.add_row("Is hub", "yes" if degree > hub_threshold else "no")
    table.add_row("Is ghost", "yes" if graph.is_ghost(note_id) else "no")
    table.add_row("Neighbors", str(len(neighbors)))
    console.print(table)

    for neighbor in neighbors:
        console.print(f"  - {neighbor}")
