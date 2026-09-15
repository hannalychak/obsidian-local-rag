"""Chunk size control: paragraph/sentence splitting with overlap, small-section merging."""

from __future__ import annotations

import re
from itertools import pairwise

from obsidian_local_rag.core.chunking.markdown import Section

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Whitespace-split word count, used as a token proxy throughout chunking.

    `core` cannot import the real fastembed tokenizer (third-party SDK imports are adapters-only,
    per CLAUDE.md's layering rule), so this heuristic stands in for it. Good enough for size
    control; not a substitute for the dense model's own tokenizer at embedding time.
    """
    return len(text.split())


def _split_into_blocks(text: str) -> list[str]:
    """Split `text` into paragraph blocks, treating each fenced code block as one atomic block."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            current.append(raw_line)
            continue
        if in_fence:
            current.append(raw_line)
            if stripped.startswith(fence_marker):
                in_fence = False
                blocks.append("\n".join(current))
                current = []
            continue

        if stripped == "":
            if current:
                blocks.append("\n".join(current))
                current = []
            continue

        current.append(raw_line)

    if current:
        blocks.append("\n".join(current))
    return blocks


def _is_fence_block(block: str) -> bool:
    first_line = block.splitlines()[0].strip() if block else ""
    return first_line.startswith("```") or first_line.startswith("~~~")


def _hard_split_by_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort split with no natural boundary to split on: fixed-size word groups."""
    words = text.split()
    if not words:
        return [text]
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _split_by_sentences(text: str, max_tokens: int) -> list[str]:
    """Greedily pack sentences into groups <= max_tokens; hard-split any single oversized one."""
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current:
                groups.append(" ".join(current))
                current, current_tokens = [], 0
            groups.extend(_hard_split_by_words(sentence, max_tokens))
            continue
        if current and current_tokens + sentence_tokens > max_tokens:
            groups.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        groups.append(" ".join(current))
    return groups


def _pack_units(units: list[str], max_tokens: int) -> list[str]:
    """Greedily pack already-sized units into pieces <= max_tokens."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _apply_overlap(pieces: list[str], overlap_tokens: int) -> list[str]:
    """Prepend the trailing `overlap_tokens` words of each piece to the next piece."""
    if overlap_tokens <= 0 or len(pieces) < 2:
        return pieces
    result = [pieces[0]]
    for previous, current in pairwise(pieces):
        overlap_text = " ".join(previous.split()[-overlap_tokens:])
        result.append(f"{overlap_text}\n\n{current}" if overlap_text else current)
    return result


def split_oversized_section(
    section: Section, max_tokens: int, overlap_tokens: int
) -> list[Section]:
    """Split `section` so no resulting piece exceeds `max_tokens`.

    Splits on paragraph boundaries first, falling back to sentence boundaries; consecutive pieces
    overlap by `overlap_tokens`. Never splits inside a fenced code block unless the block alone
    exceeds `max_tokens` (in which case it falls back to a fixed-size word split, since a code
    block has no paragraph/sentence structure to split on).
    """
    if estimate_tokens(section.text) <= max_tokens:
        return [section]

    units: list[str] = []
    for block in _split_into_blocks(section.text):
        block_tokens = estimate_tokens(block)
        if block_tokens <= max_tokens:
            units.append(block)
        elif _is_fence_block(block):
            units.extend(_hard_split_by_words(block, max_tokens))
        else:
            units.extend(_split_by_sentences(block, max_tokens))

    pieces = _apply_overlap(_pack_units(units, max_tokens), overlap_tokens)

    result: list[Section] = []
    line_cursor = section.start_line
    for piece_text in pieces:
        line_count = piece_text.count("\n") + 1
        result.append(
            Section(
                header_path=section.header_path,
                start_line=line_cursor,
                end_line=line_cursor + line_count - 1,
                text=piece_text,
            )
        )
        line_cursor += line_count
    return result


def merge_undersized_sections(sections: list[Section], min_tokens: int) -> list[Section]:
    """Merge any section under `min_tokens` into its next sibling section under the same parent
    header, preserving document order. A section with no eligible next sibling (e.g. the last one
    in its subtree) is left as-is rather than dropped.
    """
    sections = list(sections)
    merged: list[Section] = []
    i = 0
    n = len(sections)

    while i < n:
        current = sections[i]
        if estimate_tokens(current.text) < min_tokens:
            parent = current.header_path[:-1]
            sibling_index: int | None = None
            for j in range(i + 1, n):
                candidate = sections[j]
                if len(candidate.header_path) < len(current.header_path):
                    break
                if (
                    len(candidate.header_path) == len(current.header_path)
                    and candidate.header_path[:-1] == parent
                ):
                    sibling_index = j
                    break
            if sibling_index is not None:
                sibling = sections[sibling_index]
                sections[sibling_index] = Section(
                    header_path=sibling.header_path,
                    start_line=current.start_line,
                    end_line=sibling.end_line,
                    text=f"{current.text}\n\n{sibling.text}",
                )
                i += 1
                continue
        merged.append(current)
        i += 1

    return merged
