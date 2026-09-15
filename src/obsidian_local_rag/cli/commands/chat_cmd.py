"""`obsidian-rag chat` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from obsidian_local_rag.app.agent_loop import AgentLoop
from obsidian_local_rag.app.chat_session import ChatSession
from obsidian_local_rag.app.composition import build_services
from obsidian_local_rag.cli.chat_repl import run_chat_repl
from obsidian_local_rag.cli.rendering import print_error, print_not_implemented
from obsidian_local_rag.config.settings import Settings


def chat(
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Path to the Obsidian vault.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Start an interactive chat session over the indexed vault."""
    try:
        settings = (
            Settings(vault_path=vault) if vault is not None else Settings()  # type: ignore[call-arg]  # required fields resolved from env/.env at runtime
        )
        services = build_services(settings)
        agent_loop = AgentLoop(
            services.llm_client, {}, settings.agent_max_iterations, settings.context_token_budget
        )
        chat_session = ChatSession(agent_loop, system_prompt="")
        asyncio.run(run_chat_repl(chat_session, settings))
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    except NotImplementedError:
        print_not_implemented("chat", debug=debug)
