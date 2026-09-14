"""VectorStore implementation: embedded Qdrant (`path=` mode) with named dense/sparse vectors.

Structurally implements `core.protocols.VectorStore` (duck-typed, per CLAUDE.md's Protocol-only
abstraction rule — adapters do not inherit from Protocol classes).
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from obsidian_local_rag.core.protocols import SparseVector
from obsidian_local_rag.domain.models import Chunk, ScoredChunk, SearchFilters

COLLECTION_NAME = "chunks"


class QdrantVectorStore:
    """Embedded Qdrant collection, one process at a time (the storage directory is file-locked).

    Payload per point: `note_id`, `path`, `header_path`, `start_line`, `end_line`, `tags`,
    `folder`, `created`, `text`. Payload indexes on `tags`, `folder`, `created`.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._client: QdrantClient | None = None

    def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection (named `dense` size `dense_dim`, named `sparse`, payload
        indexes on `tags`/`folder`/`created`) if it doesn't already exist.

        Must surface a clear, actionable error (not a raw exception) if the embedded storage
        directory lock is already held by another process (e.g. `index` running while `chat` is
        open).
        """
        raise NotImplementedError

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector] | None
    ) -> None:
        raise NotImplementedError

    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    def search_dense(
        self, vector: list[float], k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    def search_sparse(
        self, vector: SparseVector, k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]:
        raise NotImplementedError
