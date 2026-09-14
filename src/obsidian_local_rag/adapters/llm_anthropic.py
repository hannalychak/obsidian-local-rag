"""LLMClient implementation via the official `anthropic` SDK (Decision D2).

Structurally implements `core.protocols.LLMClient` (duck-typed — adapters do not inherit from
Protocol classes). The model id always comes from `Settings.llm_model`, never hardcoded here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from obsidian_local_rag.core.protocols import Message, ToolSpec
from obsidian_local_rag.domain.models import StreamEvent


class AnthropicLLMClient:
    """`LLMClient` backed by `anthropic.AsyncAnthropic`, streaming text deltas and tool calls.

    The API key is read from the `ANTHROPIC_API_KEY` environment variable by the SDK itself; it
    is never accepted as a constructor argument, logged, or persisted.
    """

    def __init__(self, model: str, max_tokens: int) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client: AsyncAnthropic | None = None

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec], system: str
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield  # pragma: no cover - unreachable; marks this as an async generator for mypy
