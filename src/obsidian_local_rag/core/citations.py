"""Citation guard: ensures the model only cites chunks retrieved in the current turn."""

from __future__ import annotations

from obsidian_local_rag.domain.models import Citation


def extract_citation_keys(answer_text: str) -> list[str]:
    """Parse citation keys (e.g. `[S3]`) out of `answer_text`, in order of first appearance."""
    raise NotImplementedError


def validate_citations(
    cited_keys: list[str], turn_citations: dict[str, Citation]
) -> list[Citation]:
    """Keep only `cited_keys` that map to a chunk retrieved in this turn (`turn_citations`).

    Unknown keys are dropped and logged, never raised — the answer must still render.
    """
    raise NotImplementedError
