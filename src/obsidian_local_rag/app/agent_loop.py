"""AgentLoop: bounded tool-calling loop over the LLMClient Protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from obsidian_local_rag.core.protocols import LLMClient, Message, ToolSpec
from obsidian_local_rag.domain.models import Citation, NoteContent, ScoredChunk

# Maps a tool name (e.g. "search_vault") to its handler: raw (unvalidated) arguments in, tool
# result out. Handlers are responsible for their own Pydantic argument validation.
ToolHandler = Callable[[dict[str, Any]], Awaitable[list[ScoredChunk] | NoteContent]]


class AgentLoop:
    """Drives one user turn: streams the model, dispatches tool calls, and enforces the bound.

    Iterates until the model stops calling tools or `max_iterations` is reached; on the limit,
    forces a final answer turn with no tools offered. Tool-call arguments are validated with
    Pydantic by each handler in `tool_handlers`; a `pydantic.ValidationError` is caught here and
    returned to the model as the tool result text (never raised — no crash).
    """

    def __init__(
        self, llm_client: LLMClient, tool_handlers: Mapping[str, ToolHandler], max_iterations: int
    ) -> None:
        self._llm_client = llm_client
        self._tool_handlers = tool_handlers
        self._max_iterations = max_iterations

    async def run_turn(
        self, messages: list[Message], tools: list[ToolSpec], system: str
    ) -> tuple[str, dict[str, Citation]]:
        """Run the bounded tool-calling loop for one user turn.

        Returns `(final_answer_text, turn_citations)`; `turn_citations` feeds
        `core.citations.validate_citations` before the answer is rendered.
        """
        raise NotImplementedError
