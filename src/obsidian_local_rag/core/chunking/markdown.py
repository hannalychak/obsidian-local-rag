"""Header-aware Markdown structure parsing.

Line-by-line parse with a fence-state tracker: lines inside ``` or ~~~ fenced code blocks are
never headers; `#tag` (no space after `#`) is never a header; only ATX headers `#`-`######` count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Section:
    """One header-delimited section of a note body (or the whole body, if header-less)."""

    header_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str


def split_into_sections(body: str) -> list[Section]:
    """Split `body` into header-delimited sections using a fence-state tracker.

    Edge cases that must hold:
    - `#` inside a fenced code block (``` or ~~~) is never treated as a header.
    - `#tag` (no space after `#`) is never treated as a header.
    - Only ATX headers (`#` through `######`, followed by a space) count.
    - A header-less body produces exactly one `Section` with an empty `header_path`.
    - An empty body produces an empty list.

    Complexity: O(number of lines in `body`).
    """
    raise NotImplementedError
