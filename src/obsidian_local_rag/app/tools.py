"""Agent tool implementations: search_vault, expand_graph, read_note (CLAUDE.md, section 1).

Tool-call argument schemas are Pydantic v2 models — boundary validation of untrusted,
LLM-generated tool-call arguments (CLAUDE.md, section 3.3). Dependencies are passed as explicit
function arguments (no DI container, per the anti-over-engineering rule in section 3.2).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.graph.builder import NoteGraph
from obsidian_local_rag.core.protocols import DenseEmbedder, Reranker, SparseEncoder, VectorStore
from obsidian_local_rag.domain.models import NoteContent, ScoredChunk, SearchFilters


class SearchVaultArgs(BaseModel):
    query: str
    filters: SearchFilters | None = None


class ExpandGraphArgs(BaseModel):
    note_ids: list[str]
    depth: int


class ReadNoteArgs(BaseModel):
    note_id: str


async def search_vault(
    args: SearchVaultArgs,
    dense_embedder: DenseEmbedder,
    sparse_encoder: SparseEncoder,
    vector_store: VectorStore,
    reranker: Reranker,
    settings: Settings,
) -> list[ScoredChunk]:
    """Hybrid dense+sparse retrieval (via `core.retrieval.hybrid_search`), fused with RRF and
    passed through `core.rerank.rerank_candidates` before returning to the agent loop.
    """
    raise NotImplementedError


async def expand_graph(
    args: ExpandGraphArgs,
    graph: NoteGraph,
    reranker: Reranker,
    settings: Settings,
) -> list[ScoredChunk]:
    """BFS graph expansion (`core.graph.expand`) from `args.note_ids`, seeded at score 1.0.

    Expanded notes' best-matching chunks are candidates only: they MUST pass through
    `core.rerank.rerank_candidates` before returning to the agent loop (CLAUDE.md, section 3.7).
    """
    raise NotImplementedError


async def read_note(args: ReadNoteArgs, vault_path: Path) -> NoteContent:
    """Read a single note's full content by `note_id`, for the `read_note` agent tool."""
    raise NotImplementedError
