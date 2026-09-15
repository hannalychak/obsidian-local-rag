"""Extraction of wiki-links, embeds, and markdown links from note bodies."""

from __future__ import annotations

import re

from obsidian_local_rag.domain.models import Link

_INLINE_CODE_RE = re.compile(r"`[^`]*`")

_WIKILINK_RE = re.compile(
    r"(?P<embed>!)?\[\[(?P<target>[^\]|#]+?)"
    r"(?:#(?P<is_block>\^)?(?P<frag>[^\]|]+))?"
    r"(?:\|(?P<alias>[^\]]+))?\]\]"
)

_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^\s)]+\.md)\)")

_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]+$")


def _has_non_markdown_extension(target: str) -> bool:
    """True if `target` ends in a file extension other than `.md` (e.g. `.png`, `.pdf`)."""
    match = _EXTENSION_RE.search(target)
    return match is not None and match.group(0).lower() != ".md"


def _strip_code_spans(line: str) -> str:
    """Blank out inline-code spans (`` `...` ``) so link regexes never match inside them."""
    return _INLINE_CODE_RE.sub(" ", line)


def extract_links(note_id: str, body: str) -> list[Link]:
    """Extract every link/embed reference from `body`, before resolution.

    Recognizes, outside code fences and inline code:
    - `[[target]]`, `[[target|alias]]`, `[[target#Heading]]`, `[[target#^block-id]]`
    - `![[embed]]`
    - standard `[text](relative/note.md)` links

    External URLs and non-markdown embeds (images, PDFs, etc.) are ignored for graph purposes.
    Each returned `Link` has `target_note_id=None` and `resolved=False`; resolution happens
    separately in `resolver.resolve_links`.

    Complexity: O(length of `body`).
    """
    links: list[Link] = []
    in_fence = False
    fence_marker = ""

    for raw_line in body.splitlines():
        stripped = raw_line.strip()

        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            continue
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            continue

        line = _strip_code_spans(raw_line)

        for match in _WIKILINK_RE.finditer(line):
            target = match.group("target").strip()
            if not target:
                continue
            is_embed = match.group("embed") is not None
            if is_embed and _has_non_markdown_extension(target):
                continue

            frag = match.group("frag")
            is_block = match.group("is_block") is not None
            alias = match.group("alias")

            links.append(
                Link(
                    source_note_id=note_id,
                    target_raw=target,
                    target_note_id=None,
                    alias=alias.strip() if alias else None,
                    heading=frag.strip() if frag and not is_block else None,
                    block_id=frag.strip() if frag and is_block else None,
                    kind="embed" if is_embed else "link",
                    resolved=False,
                )
            )

        for match in _MARKDOWN_LINK_RE.finditer(line):
            links.append(
                Link(
                    source_note_id=note_id,
                    target_raw=match.group("target"),
                    target_note_id=None,
                    alias=None,
                    heading=None,
                    block_id=None,
                    kind="link",
                    resolved=False,
                )
            )

    return links
