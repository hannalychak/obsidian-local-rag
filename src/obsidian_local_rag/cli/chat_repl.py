"""Interactive chat REPL: prompt_toolkit input + rich.Live streaming output."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from obsidian_local_rag.app.chat_session import ChatSession
from obsidian_local_rag.cli.rendering import print_error, render_sources_footer
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.protocols import LLMConnectionError

_HELP_TEXT = "Ask a question, or use /sources, /clear, /exit. Ctrl+D also exits."


async def run_chat_repl(chat_session: ChatSession, settings: Settings) -> None:
    """Run the interactive chat REPL until `/exit` or Ctrl+D.

    - `prompt_toolkit.PromptSession.prompt_async` with a persistent history file under
      `settings.data_dir`.
    - Streamed output via `rich.Live` + `rich.markdown.Markdown`, throttled to <= 10 fps;
      `rich.Live` is scoped tightly around each answer, so it's never active while the prompt is.
    - Ctrl+C during generation cancels the current response task only (session continues);
      Ctrl+C at the prompt clears the line (prompt_toolkit's default); Ctrl+D exits.
    - Slash commands: `/sources` (last turn's sources), `/clear` (reset conversation), `/exit`.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(settings.data_dir / "chat_history"))
    prompt_session: PromptSession[str] = PromptSession(history=history)
    console = Console()

    console.print(_HELP_TEXT)

    while True:
        try:
            user_text = (await prompt_session.prompt_async("you> ")).strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            continue

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            chat_session.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if user_text == "/sources":
            _print_sources(console, chat_session, settings)
            continue

        await _ask_and_stream(console, chat_session, user_text)


async def _ask_and_stream(console: Console, chat_session: ChatSession, user_text: str) -> None:
    buffer: list[str] = []
    try:
        with Live(console=console, refresh_per_second=10) as live:

            def on_delta(text: str) -> None:
                buffer.append(text)
                live.update(Markdown("".join(buffer)))

            await chat_session.ask(user_text, on_delta)
    except KeyboardInterrupt:
        console.print("\n[dim][cancelled][/dim]")
        return
    except LLMConnectionError as exc:
        print_error(str(exc))
        return


def _print_sources(console: Console, chat_session: ChatSession, settings: Settings) -> None:
    sources = chat_session.last_sources()
    if not sources:
        console.print("[dim]No sources for the last turn.[/dim]")
        return
    console.print(render_sources_footer(sources, settings.vault_path.name))
