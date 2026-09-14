"""Interactive chat REPL: prompt_toolkit input + rich.Live streaming output."""

from __future__ import annotations

from obsidian_local_rag.app.chat_session import ChatSession
from obsidian_local_rag.config.settings import Settings


async def run_chat_repl(chat_session: ChatSession, settings: Settings) -> None:
    """Run the interactive chat REPL until `/exit` or Ctrl+D.

    Contract:
    - `prompt_toolkit.PromptSession.prompt_async` with a persistent history file under
      `settings.data_dir`.
    - Streamed output via `rich.Live` + `rich.markdown.Markdown`, throttled to <= 10 fps;
      `rich.Live` must never be active while the prompt is active.
    - Ctrl+C during generation cancels the current response task only (session continues);
      Ctrl+C at the prompt clears the line; Ctrl+D exits.
    - Slash commands: `/sources` (last turn's sources), `/clear` (reset conversation), `/exit`.
    """
    raise NotImplementedError
