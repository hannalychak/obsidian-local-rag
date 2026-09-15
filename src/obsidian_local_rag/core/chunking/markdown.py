"""Header-aware Markdown structure parsing.

Line-by-line parse with a fence-state tracker: lines inside ``` or ~~~ fenced code blocks are
never headers; `#tag` (no space after `#`) is never a header; only ATX headers `#`-`######` count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Up to 3 leading spaces allowed (CommonMark), 1-6 `#`, then a required space before the title.
_ATX_HEADER_RE = re.compile(r"^ {0,3}(#{1,6}) +(.*)$")
_TRAILING_HASHES_RE = re.compile(r"\s+#+\s*$")


@dataclass(frozen=True, slots=True)
class Section:
    """One header-delimited section of a note body (or the whole body, if header-less)."""

    header_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str


def _parse_header(line: str) -> tuple[int, str] | None:
    """Return `(level, title)` if `line` is an ATX header, else None."""
    match = _ATX_HEADER_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    title = _TRAILING_HASHES_RE.sub("", match.group(2)).strip()
    return level, title


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
    lines = body.splitlines()
    if not lines:
        return []

    sections: list[Section] = []
    stack: list[str] = []
    current_lines: list[str] = []
    current_start = 1
    in_fence = False
    fence_marker = ""

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            current_lines.append(raw_line)
            continue
        if in_fence:
            current_lines.append(raw_line)
            if stripped.startswith(fence_marker):
                in_fence = False
            continue

        header = _parse_header(raw_line)
        if header is not None:
            if current_lines:
                sections.append(
                    Section(
                        header_path=tuple(stack),
                        start_line=current_start,
                        end_line=line_number - 1,
                        text="\n".join(current_lines),
                    )
                )
            level, title = header
            del stack[level - 1 :]
            stack.append(title)
            current_lines = [raw_line]
            current_start = line_number
            continue

        current_lines.append(raw_line)

    if current_lines:
        sections.append(
            Section(
                header_path=tuple(stack),
                start_line=current_start,
                end_line=len(lines),
                text="\n".join(current_lines),
            )
        )

    return sections
