"""Citation guard: ensures the model only cites chunks retrieved in the current turn."""

from __future__ import annotations

import logging
import re

from obsidian_local_rag.domain.models import Citation

logger = logging.getLogger(__name__)

_CITATION_KEY_RE = re.compile(r"\[(S\d+)\]")


def extract_citation_keys(answer_text: str) -> list[str]:
    """Parse citation keys (e.g. `[S3]`) out of `answer_text`, in order of first appearance."""
    return _CITATION_KEY_RE.findall(answer_text)


def validate_citations(
    cited_keys: list[str], turn_citations: dict[str, Citation]
) -> list[Citation]:
    """Keep only `cited_keys` that map to a chunk retrieved in this turn (`turn_citations`).

    Unknown keys are dropped and logged, never raised — the answer must still render. Repeated
    citations of the same key are deduplicated (first occurrence wins), since the result feeds a
    sources footer where each source should appear once.
    """
    result: list[Citation] = []
    seen: set[str] = set()
    for key in cited_keys:
        if key in seen:
            continue
        citation = turn_citations.get(key)
        if citation is None:
            logger.warning("Dropping unknown citation key: %s", key)
            continue
        seen.add(key)
        result.append(citation)
    return result
