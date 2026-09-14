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
from obsidian_local_rag.cli.rendering import print_error, print_not_implemented
from obsidian_local_rag.config.settings import Settings

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
        corpus_stats = indexing_service.compute_stats()
        table = Table(title="Corpus stats")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Notes", str(corpus_stats.note_count))
        table.add_row("Chunks", str(corpus_stats.chunk_count))
        console.print(table)
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    except NotImplementedError:
        print_not_implemented("stats", debug=debug)
