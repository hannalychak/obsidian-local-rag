"""LLMClient implementation via a local Ollama instance — the only LLM backend, by design.

Structurally implements `core.protocols.LLMClient` (duck-typed — adapters do not inherit from
Protocol classes). Wired up by `app.composition.build_services`; nothing above this adapter
touches Ollama's HTTP API directly, only the `LLMClient` Protocol.

Wire format verified against a live local Ollama 0.34.0 server + `llama3.1` (2026-09-15):
streaming NDJSON lines, each `{"message": {"role", "content", "tool_calls"?}, "done", ...}`;
`tool_calls[].function.arguments` arrives as an already-parsed JSON object, not a string;
`options.num_predict` caps output length; a bare `{"role": "system", ...}` message is honored.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx

from obsidian_local_rag.core.protocols import LLMConnectionError, Message, ToolSpec
from obsidian_local_rag.domain.models import StopEvent, StreamEvent, TextDelta, ToolCallEvent


def is_ollama_running(base_url: str) -> bool:
    """Best-effort check: True if something answers `{base_url}/api/version`."""
    try:
        return httpx.get(f"{base_url}/api/version", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ollama_running(base_url: str, *, startup_timeout: float = 30.0) -> bool:
    """Start the Ollama service via `brew services` if it isn't already reachable.

    Returns True if this call started it (the caller should stop it again when done); False if
    it was already running (leave it alone — something else may be using it, e.g. the user
    started it manually).
    """
    if is_ollama_running(base_url):
        return False

    try:
        subprocess.run(
            ["brew", "services", "start", "ollama"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LLMConnectionError(
            "Could not start Ollama via `brew services start ollama`. Start it yourself "
            "(`brew services start ollama`) and try again."
        ) from exc

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_ollama_running(base_url):
            return True
        time.sleep(0.5)

    raise LLMConnectionError(
        f"Started Ollama via `brew services start ollama` but it didn't come up at {base_url} "
        f"within {startup_timeout:.0f}s."
    )


def stop_ollama() -> None:
    """Stop the Ollama service via `brew services`. Best-effort — never raises."""
    subprocess.run(["brew", "services", "stop", "ollama"], check=False, capture_output=True)


def _to_ollama_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return payload


def _to_ollama_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _events_from_line(line: str) -> list[StreamEvent]:
    """Translate one NDJSON response line into zero or more `StreamEvent`s.

    A single line can carry a text delta, one or more tool calls, and/or the terminal `done`
    marker together — each becomes its own event rather than only the first found.
    """
    data = json.loads(line)
    message = data.get("message") or {}
    events: list[StreamEvent] = []

    content = message.get("content") or ""
    if content:
        events.append(TextDelta(text=content))

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        events.append(
            ToolCallEvent(
                call_id=call.get("id") or f"call_{uuid4().hex[:12]}",
                name=function.get("name", ""),
                arguments=function.get("arguments") or {},
            )
        )

    if data.get("done"):
        events.append(
            StopEvent(
                reason=data.get("done_reason", "stop"),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
        )

    return events


class OllamaClient:
    """`LLMClient` backed by a local Ollama server's `/api/chat` streaming endpoint.

    No API key: this is the offline path. `base_url` defaults to `http://localhost:11434`
    (`Settings.ollama_base_url`); the model id comes from `Settings.llm_model` and must already be
    pulled locally (`ollama pull <model>`) — this adapter does not pull models on demand.
    """

    def __init__(self, model: str, base_url: str, max_tokens: int) -> None:
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(base_url=base_url, timeout=None)

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec], system: str
    ) -> AsyncIterator[StreamEvent]:
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(_to_ollama_message(message) for message in messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            "options": {"num_predict": self._max_tokens},
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(tool) for tool in tools]

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    for event in _events_from_line(line):
                        yield event
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}. "
                f"Is it running? (`ollama serve`, or `brew services start ollama`)"
            ) from exc
