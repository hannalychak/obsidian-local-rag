"""`obsidian-rag graph-info` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from obsidian_local_rag.app.composition import build_services
from obsidian_local_rag.cli.rendering import print_error, print_not_implemented
from obsidian_local_rag.config.settings import Settings


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
        build_services(settings)
        raise NotImplementedError
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    except NotImplementedError:
        print_not_implemented("graph-info", debug=debug)
