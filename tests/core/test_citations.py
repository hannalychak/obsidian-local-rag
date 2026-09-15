"""Edge cases for the citation guard (CLAUDE.md, section 3.10)."""

from __future__ import annotations

from pathlib import Path

from obsidian_local_rag.core.citations import extract_citation_keys, validate_citations
from obsidian_local_rag.domain.models import Citation


def test_extract_citation_keys_order_preserved() -> None:
    keys = extract_citation_keys("As shown in [S1] and later [S3], also [S1] again.")
    assert keys == ["S1", "S3", "S1"]


def test_known_citation_key_kept() -> None:
    turn_citations = {
        "S1": Citation(key="S1", chunk_id="c1", path=Path("n.md"), start_line=1, end_line=2)
    }
    kept = validate_citations(["S1"], turn_citations)
    assert [c.key for c in kept] == ["S1"]


def test_unknown_citation_dropped() -> None:
    turn_citations = {
        "S1": Citation(key="S1", chunk_id="c1", path=Path("n.md"), start_line=1, end_line=2)
    }
    kept = validate_citations(["S1", "S99"], turn_citations)
    assert [c.key for c in kept] == ["S1"]
