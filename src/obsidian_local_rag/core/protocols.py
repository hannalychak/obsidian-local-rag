"""Protocol interfaces for infrastructure seams.

`core` depends only on these Protocols, never on concrete `adapters` (see CLAUDE.md, section
3.1). Wiring concrete implementations to these Protocols happens in exactly one place:
`app.composition`.

Per the anti-over-engineering rule (CLAUDE.md, section 3.2), these five Protocols are the only
abstract interfaces in the codebase. Do not add more.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from obsidian_local_rag.domain.models import (
    Chunk,
    ScoredChunk,
    SearchFilters,
    StreamEvent,
    ToolCallEvent,
)

# A sparse vector as (indices, values), mirroring fastembed's SparseEmbedding / Qdrant's
# SparseVector shape without importing either SDK into `core`.
SparseVector = tuple[Sequence[int], Sequence[float]]


class LLMConnectionError(RuntimeError):
    """Raised by an `LLMClient` implementation when it cannot reach its backend.

    Deliberately not a third-party exception type (e.g. `httpx.ConnectError`): callers above
    `adapters` — `app`, `cli` — must be able to catch connection failures without importing the
    SDK that produced them (CLAUDE.md's adapters-only-SDK-imports rule).
    """


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation passed to `LLMClient.stream`.

    `tool_calls` is populated only on assistant messages that requested tool calls (empty
    otherwise), so a full conversation history round-trips correctly across `stream` calls — the
    model needs to see its own prior tool-call requests, not just their results.
    """

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCallEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declares one tool the LLM may call, in provider-agnostic form."""

    name: str
    description: str
    input_schema: dict[str, Any]


class DenseEmbedder(Protocol):
    """Produces dense embedding vectors for a batch of texts, in input order."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SparseEncoder(Protocol):
    """Produces sparse (BM25-style) vectors for a batch of texts, in input order."""

    def encode(self, texts: list[str]) -> list[SparseVector]: ...


class VectorStore(Protocol):
    """The embedded Qdrant collection: named `dense` and `sparse` vectors plus payload filters.

    Implementations lock their storage directory for the process lifetime (embedded Qdrant,
    `path=` mode); callers must surface a clear error if the lock is already held rather than
    letting the underlying exception propagate as a traceback.
    """

    def ensure_collection(self, dense_dim: int) -> None:
        """Create the backing collection (named `dense` size `dense_dim`, named `sparse`) if it
        doesn't already exist. Idempotent — safe to call before every indexing run.
        """
        ...

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector] | None
    ) -> None: ...

    def delete(self, note_ids: list[str]) -> None:
        """Delete every point belonging to any of `note_ids` (not `chunk_id`s).

        A deleted or re-chunked note's old `chunk_id`s aren't knowable from the caller's side
        (its file is gone, or its chunking changed) — only its `note_id` is stable, so deletion
        is note-scoped, matching how incremental indexing actually needs it.
        """
        ...

    def search_dense(
        self, vector: list[float], k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]: ...

    def search_sparse(
        self, vector: SparseVector, k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]: ...

    def get_by_note_ids(self, note_ids: list[str]) -> list[Chunk]:
        """Fetch every chunk belonging to any of `note_ids`, unranked (no query vector involved).

        For `expand_graph`: graph expansion scores *notes*, not chunks, and has no query to search
        with — this is a plain filtered lookup, not a similarity search.
        """
        ...


class Reranker(Protocol):
    """Cross-encoder reranking of candidates against a query, best match first."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    """Streams a model turn: text deltas, tool calls, and a terminal stop-with-usage event.

    Implemented by `adapters.llm_ollama.OllamaClient` — the only LLM backend by design, so vault
    content never leaves the machine. `core`, the rest of `app`, and `cli` depend only on this
    Protocol, never on the concrete adapter; `app.composition.build_services` is the only place
    that imports it.
    """

    def stream(
        self, messages: list[Message], tools: list[ToolSpec], system: str
    ) -> AsyncIterator[StreamEvent]: ...
