"""ChatSession: manages conversation state across turns, drives AgentLoop + citation validation."""

from __future__ import annotations

from collections.abc import Callable

from obsidian_local_rag.app.agent_loop import AgentLoop
from obsidian_local_rag.app.tools import TOOL_SPECS
from obsidian_local_rag.core.citations import extract_citation_keys, validate_citations
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

    async def ask(self, user_text: str, on_text_delta: Callable[[str], None] | None = None) -> str:
        """Append `user_text`, run one agent turn, append the answer, and return it.

        `on_text_delta`, if given, is forwarded to `AgentLoop.run_turn` for live streaming to the
        terminal; see its docstring for exactly when it fires.

        Validates citations (`core.citations.validate_citations`) against the turn's citation map
        before returning, and updates `last_sources()`.
        """
        self._messages.append(Message(role="user", content=user_text))
        answer, turn_citations = await self._agent_loop.run_turn(
            self._messages, TOOL_SPECS, self._system_prompt, on_text_delta
        )
        self._messages.append(Message(role="assistant", content=answer))

        cited_keys = extract_citation_keys(answer)
        valid_citations = validate_citations(cited_keys, turn_citations)
        self._last_turn_citations = {citation.key: citation for citation in valid_citations}

        return answer

    def clear(self) -> None:
        """Reset the conversation (the `/clear` slash command)."""
        self._messages = []
        self._last_turn_citations = {}

    def last_sources(self) -> dict[str, Citation]:
        """Return the citation map from the most recent `ask` call (the `/sources` command)."""
        return self._last_turn_citations
