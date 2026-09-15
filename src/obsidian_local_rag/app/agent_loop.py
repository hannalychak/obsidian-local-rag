"""AgentLoop: bounded tool-calling loop over the LLMClient Protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from obsidian_local_rag.app.context import assemble_context
from obsidian_local_rag.core.protocols import LLMClient, Message, ToolSpec
from obsidian_local_rag.domain.models import (
    Citation,
    NoteContent,
    ScoredChunk,
    TextDelta,
    ToolCallEvent,
)

# Maps a tool name (e.g. "search_vault") to its handler: raw (unvalidated) arguments in, tool
# result out. Handlers are responsible for their own Pydantic argument validation.
ToolHandler = Callable[[dict[str, Any]], Awaitable[list[ScoredChunk] | NoteContent]]


class AgentLoop:
    """Drives one user turn: streams the model, dispatches tool calls, and enforces the bound.

    Iterates until the model stops calling tools or `max_iterations` is reached; on the limit,
    forces a final answer turn with no tools offered. Tool-call arguments are validated with
    Pydantic by each handler in `tool_handlers`; any exception a handler raises — a
    `pydantic.ValidationError` from bad arguments, or a runtime failure like a missing note — is
    caught here and returned to the model as the tool result text (never raised — no crash).

    Each tool call's `list[ScoredChunk]` result is turned into citation-keyed context text via
    `app.context.assemble_context`, with `start_index` advanced across calls within the same turn
    so citation keys never collide; a `NoteContent` result (from `read_note`) is presented as
    plain note text with no citation key, since it isn't a ranked/cited search result.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_handlers: Mapping[str, ToolHandler],
        max_iterations: int,
        context_token_budget: int,
    ) -> None:
        self._llm_client = llm_client
        self._tool_handlers = tool_handlers
        self._max_iterations = max_iterations
        self._context_token_budget = context_token_budget

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Citation]]:
        """Run the bounded tool-calling loop for one user turn.

        `on_text_delta`, if given, is called with each text fragment as it streams in — across
        every round, not just the final one, so any narration the model produces alongside a tool
        call is visible too. The complete answer is still returned (and still only usable) once
        the whole turn finishes, since a tool-calling round's text isn't the final answer.

        Returns `(final_answer_text, turn_citations)`; `turn_citations` feeds
        `core.citations.validate_citations` before the answer is rendered.
        """
        conversation = list(messages)
        turn_citations: dict[str, Citation] = {}

        for _iteration in range(self._max_iterations):
            text, tool_calls = await self._stream_once(conversation, tools, system, on_text_delta)
            if not tool_calls:
                return text, turn_citations

            conversation.append(
                Message(role="assistant", content=text, tool_calls=tuple(tool_calls))
            )
            for call in tool_calls:
                result_text, new_citations = await self._execute_tool(call, len(turn_citations) + 1)
                turn_citations.update(new_citations)
                conversation.append(
                    Message(role="tool", content=result_text, tool_call_id=call.call_id)
                )

        # Bound exhausted and the model still wants tools: force a final answer, no tools offered.
        text, _ = await self._stream_once(conversation, [], system, on_text_delta)
        return text, turn_citations

    async def _stream_once(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str,
        on_text_delta: Callable[[str], None] | None,
    ) -> tuple[str, list[ToolCallEvent]]:
        text_parts: list[str] = []
        tool_calls: list[ToolCallEvent] = []
        async for event in self._llm_client.stream(messages, tools, system):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                if on_text_delta is not None:
                    on_text_delta(event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(event)
        return "".join(text_parts), tool_calls

    async def _execute_tool(
        self, call: ToolCallEvent, next_citation_index: int
    ) -> tuple[str, dict[str, Citation]]:
        handler = self._tool_handlers.get(call.name)
        if handler is None:
            return f"Error: unknown tool '{call.name}'", {}

        try:
            result = await handler(call.arguments)
        except Exception as exc:
            return f"Error: '{call.name}' failed: {exc}", {}

        if isinstance(result, NoteContent):
            return f"# {result.note_id}\n\n{result.body}", {}

        return assemble_context(result, self._context_token_budget, next_citation_index)
