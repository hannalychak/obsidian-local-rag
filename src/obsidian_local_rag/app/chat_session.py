"""ChatSession: manages conversation state across turns, drives AgentLoop + citation validation."""

from __future__ import annotations

from obsidian_local_rag.app.agent_loop import AgentLoop
from obsidian_local_rag.core.protocols import Message
from obsidian_local_rag.domain.models import Citation


class ChatSession:
    """Holds conversation history for one `chat` REPL session.

    `/clear` resets the conversation; `/sources` reads the last turn's citation map without
    re-running retrieval.
    """

    def __init__(self, agent_loop: AgentLoop, system_prompt: str) -> None:
        self._agent_loop = agent_loop
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._last_turn_citations: dict[str, Citation] = {}

    async def ask(self, user_text: str) -> str:
        """Append `user_text`, run one agent turn, append the answer, and return it.

        Validates citations (`core.citations.validate_citations`) against the turn's citation map
        before returning, and updates `last_sources()`.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Reset the conversation (the `/clear` slash command)."""
        raise NotImplementedError

    def last_sources(self) -> dict[str, Citation]:
        """Return the citation map from the most recent `ask` call (the `/sources` command)."""
        raise NotImplementedError
