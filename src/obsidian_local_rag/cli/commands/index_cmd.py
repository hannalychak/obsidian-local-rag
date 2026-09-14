"""`obsidian-rag index` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from obsidian_local_rag.app.composition import build_services
from obsidian_local_rag.app.indexing_service import IndexingService
from obsidian_local_rag.cli.rendering import print_error, print_not_implemented
from obsidian_local_rag.config.settings import Settings


def index(
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Path to the Obsidian vault.")
    ] = None,
    full: Annotated[
        bool, typer.Option("--full", help="Force a full re-index, ignoring the manifest.")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Incrementally index the vault into the local Qdrant collection."""
    try:
        settings = (
            Settings(vault_path=vault) if vault is not None else Settings()  # type: ignore[call-arg]  # required fields resolved from env/.env at runtime
        )
        services = build_services(settings)
        indexing_service = IndexingService(settings, services)
        asyncio.run(indexing_service.run_index(full=full))
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    except NotImplementedError:
        print_not_implemented("index", debug=debug)
