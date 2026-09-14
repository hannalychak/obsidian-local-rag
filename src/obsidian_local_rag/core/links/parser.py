"""Extraction of wiki-links, embeds, and markdown links from note bodies."""

from __future__ import annotations

from obsidian_local_rag.domain.models import Link


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
    raise NotImplementedError
