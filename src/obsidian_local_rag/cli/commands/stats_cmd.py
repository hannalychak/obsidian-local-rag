"""`obsidian-rag stats` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from obsidian_local_rag.app.composition import build_services
from obsidian_local_rag.app.indexing_service import IndexingService
from obsidian_local_rag.cli.rendering import print_error
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.ingest.stats import CorpusStats

console = Console()


def stats(
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Path to the Obsidian vault.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Show corpus statistics: note/chunk counts, token distribution, top tags/folders, orphans."""
    try:
        settings = (
            Settings(vault_path=vault) if vault is not None else Settings()  # type: ignore[call-arg]  # required fields resolved from env/.env at runtime
        )
        services = build_services(settings)
        indexing_service = IndexingService(settings, services)
        _print_stats(indexing_service.compute_stats())
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc


def _print_stats(corpus_stats: CorpusStats) -> None:
    overview = Table(title="Corpus stats")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("Notes", str(corpus_stats.note_count))
    overview.add_row("Chunks", str(corpus_stats.chunk_count))
    for label, value in corpus_stats.token_distribution.items():
        overview.add_row(f"Tokens/chunk ({label})", f"{value:.1f}")
    overview.add_row("Orphan notes", str(len(corpus_stats.orphan_note_ids)))
    overview.add_row("Unresolved links", str(len(corpus_stats.unresolved_link_targets)))
    console.print(overview)

    if corpus_stats.top_tags:
        tags_table = Table(title="Top tags")
        tags_table.add_column("Tag")
        tags_table.add_column("Count", justify="right")
        for tag, count in corpus_stats.top_tags:
            tags_table.add_row(tag, str(count))
        console.print(tags_table)

    if corpus_stats.top_folders:
        folders_table = Table(title="Top folders")
        folders_table.add_column("Folder")
        folders_table.add_column("Count", justify="right")
        for folder, count in corpus_stats.top_folders:
            folders_table.add_row(folder or "(vault root)", str(count))
        console.print(folders_table)
