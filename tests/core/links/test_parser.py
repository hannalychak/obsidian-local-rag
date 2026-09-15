"""Edge cases for link/embed extraction (CLAUDE.md, section 3.7)."""

from __future__ import annotations

from obsidian_local_rag.core.links.parser import extract_links


def test_extracts_plain_wikilink() -> None:
    links = extract_links("note", "See [[Target Note]] for details.")
    assert links[0].target_raw == "Target Note"
    assert links[0].kind == "link"


def test_extracts_alias_link() -> None:
    links = extract_links("note", "See [[Target Note|Friendly Name]].")
    assert links[0].target_raw == "Target Note"
    assert links[0].alias == "Friendly Name"


def test_extracts_heading_link() -> None:
    links = extract_links("note", "See [[Target Note#Some Heading]].")
    assert links[0].heading == "Some Heading"


def test_extracts_block_link() -> None:
    links = extract_links("note", "See [[Target Note#^abc123]].")
    assert links[0].block_id == "abc123"


def test_extracts_embed_as_embed_kind() -> None:
    links = extract_links("note", "![[Some Note]]")
    assert links[0].kind == "embed"


def test_extracts_standard_markdown_link_to_note() -> None:
    links = extract_links("note", "[text](relative/note.md)")
    assert links[0].target_raw == "relative/note.md"


def test_ignores_external_url() -> None:
    links = extract_links("note", "[external](https://example.com)")
    assert links == []


def test_ignores_non_markdown_embed() -> None:
    links = extract_links("note", "![[diagram.png]]")
    assert links == []


def test_link_inside_fenced_code_block_is_ignored() -> None:
    body = "```\n[[Not A Link]]\n```\n"
    assert extract_links("note", body) == []


def test_link_inside_inline_code_is_ignored() -> None:
    links = extract_links("note", "Use `[[Not A Link]]` as an example.")
    assert links == []
