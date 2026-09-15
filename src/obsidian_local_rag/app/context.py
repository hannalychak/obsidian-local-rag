"""Context assembly: token-budget-aware chunk presentation with citation keys."""

from __future__ import annotations

from obsidian_local_rag.domain.models import Citation, ScoredChunk


def assemble_context(
    chunks: list[ScoredChunk], token_budget: int, start_index: int = 1
) -> tuple[str, dict[str, Citation]]:
    """Assemble the context block shown to the LLM plus its citation-key map.

    Each chunk is presented with a short citation key (e.g. `[S3]`) mapped to its `chunk_id`.
    `chunks` is assumed already ranked best-first (fusion/rerank output); once `token_budget` is
    exhausted the remaining, lowest-ranked chunks are dropped rather than sampled. The first chunk
    is always included even if it alone exceeds the budget, so a turn never gets an empty context.

    `start_index` lets a caller assemble multiple chunk batches into one turn without citation-key
    collisions (e.g. one call per tool invocation this turn): pass `len(existing_citations) + 1`.

    Returns `(context_text, key -> Citation)`.
    """
    lines: list[str] = []
    citations: dict[str, Citation] = {}
    used_tokens = 0
    index = start_index

    for scored in chunks:
        chunk = scored.chunk
        if citations and used_tokens + chunk.token_estimate > token_budget:
            break

        key = f"S{index}"
        index += 1
        used_tokens += chunk.token_estimate

        breadcrumb = " > ".join((chunk.note_id, *chunk.header_path))
        lines.append(f"[{key}] {breadcrumb}\n{chunk.text}")
        citations[key] = Citation(
            key=key,
            chunk_id=chunk.chunk_id,
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

    return "\n\n".join(lines), citations
