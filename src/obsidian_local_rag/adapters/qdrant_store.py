"""VectorStore implementation: embedded Qdrant (`path=` mode) with named dense/sparse vectors.

Structurally implements `core.protocols.VectorStore` (duck-typed, per CLAUDE.md's Protocol-only
abstraction rule — adapters do not inherit from Protocol classes).

Verified live against qdrant-client 1.19.0 embedded mode (2026-09-15): `query_points` (not
`search`, removed in this version) for search, `DatetimeRange` for date filters, and payload
indexes are silently no-ops in embedded/local mode (Qdrant warns and still filters correctly via
full scan — fine at personal-vault scale). A second `QdrantClient` on the same `path` raises a
plain `RuntimeError` with an already-clear, actionable message ("already accessed by another
instance... use Qdrant server instead"), so it's allowed to propagate as-is rather than being
re-wrapped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from obsidian_local_rag.core.protocols import SparseVector
from obsidian_local_rag.domain.models import Chunk, ScoredChunk, SearchFilters

COLLECTION_NAME = "chunks"


def _to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "note_id": chunk.note_id,
        "path": str(chunk.path),
        "header_path": list(chunk.header_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "token_estimate": chunk.token_estimate,
        "tags": list(chunk.tags),
        "folder": chunk.folder,
        "created": chunk.created.isoformat() if chunk.created else None,
        "text": chunk.text,
    }


def _chunk_from_payload(chunk_id: str, payload: dict[str, Any]) -> Chunk:
    created_raw = payload.get("created")
    return Chunk(
        chunk_id=chunk_id,
        note_id=payload["note_id"],
        path=Path(payload["path"]),
        header_path=tuple(payload.get("header_path", ())),
        text=payload["text"],
        start_line=payload["start_line"],
        end_line=payload["end_line"],
        token_estimate=payload["token_estimate"],
        tags=tuple(payload.get("tags", ())),
        folder=payload.get("folder", ""),
        created=datetime.fromisoformat(created_raw) if created_raw else None,
    )


def _build_filter(filters: SearchFilters | None) -> qm.Filter | None:
    if filters is None:
        return None

    conditions: list[qm.FieldCondition] = []
    if filters.tags:
        conditions.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=list(filters.tags))))
    if filters.folder is not None:
        conditions.append(
            qm.FieldCondition(key="folder", match=qm.MatchValue(value=filters.folder))
        )
    if filters.created_after is not None or filters.created_before is not None:
        conditions.append(
            qm.FieldCondition(
                key="created",
                range=qm.DatetimeRange(gte=filters.created_after, lte=filters.created_before),
            )
        )

    if not conditions:
        return None
    # mypy's list invariance flags list[FieldCondition] against Filter.must's broader condition
    # union, even though every FieldCondition is a valid member of it — a known false positive.
    return qm.Filter(must=conditions)  # type: ignore[arg-type]


class QdrantVectorStore:
    """Embedded Qdrant collection, one process at a time (the storage directory is file-locked).

    Payload per point: `note_id`, `path`, `header_path`, `start_line`, `end_line`, `token_estimate`,
    `tags`, `folder`, `created`, `text` — enough to fully reconstruct a `Chunk` from a search hit.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._client = QdrantClient(path=str(data_dir / "qdrant"))

    def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection (named `dense` size `dense_dim`, named `sparse`, payload
        indexes on `tags`/`folder`/`created`) if it doesn't already exist.
        """
        if self._client.collection_exists(COLLECTION_NAME):
            return
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        self._client.create_payload_index(
            COLLECTION_NAME, field_name="tags", field_schema=qm.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            COLLECTION_NAME, field_name="folder", field_schema=qm.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            COLLECTION_NAME, field_name="created", field_schema=qm.PayloadSchemaType.DATETIME
        )

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector] | None
    ) -> None:
        points = []
        for i, chunk in enumerate(chunks):
            vector: dict[str, Any] = {"dense": dense[i]}
            if sparse is not None:
                indices, values = sparse[i]
                vector["sparse"] = qm.SparseVector(indices=list(indices), values=list(values))
            points.append(
                qm.PointStruct(id=chunk.chunk_id, vector=vector, payload=_to_payload(chunk))
            )
        self._client.upsert(COLLECTION_NAME, points=points)

    def delete(self, note_ids: list[str]) -> None:
        if not note_ids:
            return
        selector = qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="note_id", match=qm.MatchAny(any=note_ids))]
            )
        )
        self._client.delete(COLLECTION_NAME, points_selector=selector)

    def search_dense(
        self, vector: list[float], k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]:
        response = self._client.query_points(
            COLLECTION_NAME,
            query=vector,
            using="dense",
            limit=k,
            query_filter=_build_filter(filters),
        )
        return [
            ScoredChunk(
                chunk=_chunk_from_payload(str(point.id), point.payload or {}),
                score=point.score,
                source="dense",
            )
            for point in response.points
        ]

    def search_sparse(
        self, vector: SparseVector, k: int, filters: SearchFilters | None
    ) -> list[ScoredChunk]:
        indices, values = vector
        response = self._client.query_points(
            COLLECTION_NAME,
            query=qm.SparseVector(indices=list(indices), values=list(values)),
            using="sparse",
            limit=k,
            query_filter=_build_filter(filters),
        )
        return [
            ScoredChunk(
                chunk=_chunk_from_payload(str(point.id), point.payload or {}),
                score=point.score,
                source="sparse",
            )
            for point in response.points
        ]

    def get_by_note_ids(self, note_ids: list[str]) -> list[Chunk]:
        if not note_ids:
            return []
        records, _next_offset = self._client.scroll(
            COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="note_id", match=qm.MatchAny(any=note_ids))]
            ),
            limit=1000,
        )
        return [_chunk_from_payload(str(record.id), record.payload or {}) for record in records]
