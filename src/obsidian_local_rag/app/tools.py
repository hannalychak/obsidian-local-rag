"""Agent tool implementations: search_vault, expand_graph, read_note (CLAUDE.md, section 1).

Tool-call argument schemas are Pydantic v2 models — boundary validation of untrusted,
LLM-generated tool-call arguments (CLAUDE.md, section 3.3). Dependencies are passed as explicit
function arguments (no DI container, per the anti-over-engineering rule in section 3.2).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from obsidian_local_rag.app.agent_loop import ToolHandler
from obsidian_local_rag.app.composition import Services
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.graph.builder import NoteGraph
from obsidian_local_rag.core.graph.expand import expand
from obsidian_local_rag.core.ingest.frontmatter import parse_frontmatter
from obsidian_local_rag.core.ingest.walker import read_note_text
from obsidian_local_rag.core.protocols import ToolSpec
from obsidian_local_rag.core.rerank.rerank import rerank_candidates
from obsidian_local_rag.core.retrieval.hybrid_search import hybrid_search
from obsidian_local_rag.domain.models import NoteContent, ScoredChunk, SearchFilters


class SearchVaultArgs(BaseModel):
    query: str
    filters: SearchFilters | None = None


class ExpandGraphArgs(BaseModel):
    note_ids: list[str]
    depth: int
    # No query in the tool's own I/O contract per CLAUDE.md section 1's signature, but reranking
    # (required by section 3.7) needs one: the calling agent already has the user's question in
    # context, so it's a small ask to have it re-supply it here for scoring purposes.
    query: str


class ReadNoteArgs(BaseModel):
    note_id: str


# Declared once from the Pydantic models above so the LLM-facing schema can never drift from the
# actual validation the handlers perform.
SEARCH_VAULT_SPEC = ToolSpec(
    name="search_vault",
    description="Search the Obsidian vault for relevant notes and passages.",
    input_schema=SearchVaultArgs.model_json_schema(),
)
EXPAND_GRAPH_SPEC = ToolSpec(
    name="expand_graph",
    description=(
        "Expand outward from already-found notes via their wiki-links to discover related "
        "context. `query` should restate what you're still trying to answer."
    ),
    input_schema=ExpandGraphArgs.model_json_schema(),
)
READ_NOTE_SPEC = ToolSpec(
    name="read_note",
    description="Read one note's full content by its note_id.",
    input_schema=ReadNoteArgs.model_json_schema(),
)
TOOL_SPECS = [SEARCH_VAULT_SPEC, EXPAND_GRAPH_SPEC, READ_NOTE_SPEC]


async def search_vault(
    args: SearchVaultArgs,
    services: Services,
    settings: Settings,
) -> list[ScoredChunk]:
    """Hybrid dense+sparse retrieval (via `core.retrieval.hybrid_search`), fused with RRF and
    passed through `core.rerank.rerank_candidates` before returning to the agent loop.
    """
    fused = await hybrid_search(
        args.query,
        args.filters,
        services.dense_embedder,
        services.sparse_encoder,
        services.vector_store,
        candidate_k=settings.candidate_k,
        rrf_k=settings.rrf_k,
    )
    return await rerank_candidates(
        args.query, fused, services.reranker, rerank_n=settings.rerank_n, final_k=settings.final_k
    )


async def expand_graph(
    args: ExpandGraphArgs,
    graph: NoteGraph,
    services: Services,
    settings: Settings,
) -> list[ScoredChunk]:
    """BFS graph expansion (`core.graph.expand`) from `args.note_ids`, seeded at score 1.0.

    Expanded notes' best-matching chunks are candidates only: they MUST pass through
    `core.rerank.rerank_candidates` before returning to the agent loop (CLAUDE.md, section 3.7).
    """
    seeds = dict.fromkeys(args.note_ids, 1.0)
    expanded = expand(
        graph,
        seeds,
        max_depth=args.depth,
        max_nodes=settings.graph_max_nodes,
        decay=settings.graph_decay,
        hub_degree_threshold=settings.graph_hub_degree_threshold,
    )
    if not expanded:
        return []

    score_by_note_id = {note_id: score for note_id, score, _depth in expanded}
    chunks = await asyncio.to_thread(services.vector_store.get_by_note_ids, list(score_by_note_id))
    candidates = [
        ScoredChunk(chunk=chunk, score=score_by_note_id[chunk.note_id], source="graph")
        for chunk in chunks
    ]
    return await rerank_candidates(
        args.query,
        candidates,
        services.reranker,
        rerank_n=settings.rerank_n,
        final_k=settings.final_k,
    )


async def read_note(args: ReadNoteArgs, vault_path: Path) -> NoteContent:
    """Read a single note's full content by `note_id`, for the `read_note` agent tool."""
    path = vault_path / f"{args.note_id}.md"
    if not path.is_file():
        raise ValueError(f"No such note: {args.note_id!r}")
    meta, body = parse_frontmatter(read_note_text(path))
    return NoteContent(note_id=args.note_id, path=path, body=body, frontmatter=meta)


def build_tool_handlers(
    settings: Settings, services: Services, graph: NoteGraph
) -> dict[str, ToolHandler]:
    """Build the `{tool_name: handler}` mapping `AgentLoop` needs, closing over dependencies.

    Each handler validates its own raw (untrusted, LLM-generated) arguments into the matching
    Pydantic model; a `pydantic.ValidationError` here propagates to `AgentLoop`, which turns any
    tool-handler exception into an error message for the model rather than crashing the turn.
    """

    async def _search_vault(raw_args: dict[str, Any]) -> list[ScoredChunk]:
        return await search_vault(SearchVaultArgs(**raw_args), services, settings)

    async def _expand_graph(raw_args: dict[str, Any]) -> list[ScoredChunk]:
        return await expand_graph(ExpandGraphArgs(**raw_args), graph, services, settings)

    async def _read_note(raw_args: dict[str, Any]) -> NoteContent:
        return await read_note(ReadNoteArgs(**raw_args), settings.vault_path)

    return {
        "search_vault": _search_vault,
        "expand_graph": _expand_graph,
        "read_note": _read_note,
    }
