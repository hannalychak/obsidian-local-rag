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

from obsidian_local_rag.domain.models import Chunk, ScoredChunk, SearchFilters, StreamEvent

# A sparse vector as (indices, values), mirroring fastembed's SparseEmbedding / Qdrant's
# SparseVector shape without importing either SDK into `core`.
SparseVector = tuple[Sequence[int], Sequence[float]]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation passed to `LLMClient.stream`."""

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


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

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector] | None
    ) -> None: ...

    def delete(self, chunk_ids: list[str]) -> None: ...

    def search_dense(
        self, vector: list[float], k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]: ...

    def search_sparse(
        self, vector: SparseVector, k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    """Cross-encoder reranking of candidates against a query, best match first."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    """Streams a model turn: text deltas, tool calls, and a terminal stop-with-usage event."""

    def stream(
        self, messages: list[Message], tools: list[ToolSpec], system: str
    ) -> AsyncIterator[StreamEvent]: ...
