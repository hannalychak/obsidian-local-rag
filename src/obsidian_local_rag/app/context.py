"""Context assembly: token-budget-aware chunk presentation with citation keys."""

from __future__ import annotations

from obsidian_local_rag.domain.models import Citation, ScoredChunk


def assemble_context(
    chunks: list[ScoredChunk], token_budget: int
) -> tuple[str, dict[str, Citation]]:
    """Assemble the context block shown to the LLM plus its citation-key map.

    Each chunk is presented with a short citation key (e.g. `[S3]`) mapped to its `chunk_id`.
    Respects `token_budget`: lowest-ranked chunks are dropped first when the budget is exceeded.

    Returns `(context_text, key -> Citation)`.
    """
    raise NotImplementedError
