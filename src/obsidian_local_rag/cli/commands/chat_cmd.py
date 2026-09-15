"""`obsidian-rag chat` command."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer
from pydantic import ValidationError

from obsidian_local_rag.adapters.llm_ollama import ensure_ollama_running, stop_ollama
from obsidian_local_rag.app.agent_loop import AgentLoop
from obsidian_local_rag.app.chat_session import ChatSession
from obsidian_local_rag.app.composition import build_services
from obsidian_local_rag.app.indexing_service import load_notes_and_graph
from obsidian_local_rag.app.tools import build_tool_handlers
from obsidian_local_rag.cli.chat_repl import run_chat_repl
from obsidian_local_rag.cli.rendering import console, print_error
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.protocols import LLMConnectionError

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's Obsidian vault. "
    "Use the search_vault tool to find relevant notes before answering questions about their "
    "content; when you don't need a filter, omit the filters argument entirely rather than "
    "passing an empty value. Use expand_graph to explore notes linked from ones you've already "
    "found; use read_note to read a specific note's full content when you know its note_id. "
    "Cite retrieved context with its citation key, e.g. [S1], directly in your answer wherever "
    "you use it. Only cite keys you were actually given — never invent one. If a tool call "
    "fails, try again with corrected arguments rather than answering from memory."
)


def _raise_system_exit(signum: int, frame: FrameType | None) -> None:
    """Convert SIGTERM/SIGHUP into a catchable exception.

    Python's default handling of these (unlike SIGINT/Ctrl+C, which already raises
    KeyboardInterrupt) terminates the process immediately without running `finally` blocks —
    closing the terminal window running `chat` sends SIGHUP, which silently skipped the Ollama
    auto-stop cleanup below. Verified: reproduced with a minimal asyncio.run() + finally script,
    confirmed the finally block is skipped by default and runs correctly with this handler.
    """
    raise SystemExit(128 + signum)


def chat(
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Path to the Obsidian vault.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Start an interactive chat session over the indexed vault."""
    started_ollama = False
    previous_term_handler = signal.signal(signal.SIGTERM, _raise_system_exit)
    previous_hup_handler = signal.signal(signal.SIGHUP, _raise_system_exit)
    try:
        settings = (
            Settings(vault_path=vault) if vault is not None else Settings()  # type: ignore[call-arg]  # required fields resolved from env/.env at runtime
        )

        console.print("[dim]Checking Ollama...[/dim]")
        started_ollama = ensure_ollama_running(settings.ollama_base_url)
        if started_ollama:
            console.print("[dim]Ollama started.[/dim]")

        services = build_services(settings)
        _, graph = load_notes_and_graph(settings.vault_path, settings.exclude_globs)
        tool_handlers = build_tool_handlers(settings, services, graph)
        agent_loop = AgentLoop(
            services.llm_client,
            tool_handlers,
            settings.agent_max_iterations,
            settings.context_token_budget,
        )
        chat_session = ChatSession(agent_loop, system_prompt=_SYSTEM_PROMPT)
        asyncio.run(run_chat_repl(chat_session, settings))
    except ValidationError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    except LLMConnectionError as exc:
        print_error(str(exc), debug=debug)
        raise typer.Exit(code=1) from exc
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        signal.signal(signal.SIGHUP, previous_hup_handler)
        if started_ollama:
            console.print("[dim]Stopping Ollama...[/dim]")
            stop_ollama()
