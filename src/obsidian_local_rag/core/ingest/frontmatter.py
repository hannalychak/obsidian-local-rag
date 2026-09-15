"""Frontmatter parsing and key normalization."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

import frontmatter
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_INLINE_TAG_RE = re.compile(r"(?<!\S)#([A-Za-z0-9_/-]+)")


def parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the note body using `python-frontmatter`.

    On malformed YAML, logs a warning and returns `({}, raw_text)` (frontmatter discarded, full
    original text kept as body) rather than raising.
    """
    try:
        metadata, content = frontmatter.parse(raw_text)
    except yaml.YAMLError as exc:
        logger.warning("Malformed frontmatter, keeping body as-is: %s", exc)
        return {}, raw_text
    return metadata, content


def _strip_code(body: str) -> str:
    """Blank out fenced code blocks and inline code spans before scanning for inline tags."""
    lines: list[str] = []
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
        lines.append(_INLINE_CODE_RE.sub(" ", raw_line))
    return "\n".join(lines)


def _add_unique(target: list[str], seen: set[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in seen:
        seen.add(cleaned)
        target.append(cleaned)


def normalize_tags(raw_tags: Any, body: str) -> tuple[str, ...]:
    """Normalize a frontmatter `tags` value into a deduplicated tuple, order-preserving.

    Accepts a string or list, with or without a leading `#`. Per Phase 1 decision, also extracts
    inline body `#tags` (outside code fences and inline code) from `body` and unions them in,
    since this vault's notes carry zero frontmatter but use inline tags exclusively.
    """
    tags: list[str] = []
    seen: set[str] = set()

    if isinstance(raw_tags, str):
        for part in raw_tags.split(","):
            _add_unique(tags, seen, part.lstrip("#"))
    elif isinstance(raw_tags, list):
        for item in raw_tags:
            _add_unique(tags, seen, str(item).lstrip("#"))

    for match in _INLINE_TAG_RE.finditer(_strip_code(body)):
        _add_unique(tags, seen, match.group(1))

    return tuple(tags)


def normalize_aliases(raw_aliases: Any) -> tuple[str, ...]:
    """Normalize a frontmatter `aliases` value (string or list) into a deduplicated tuple."""
    aliases: list[str] = []
    seen: set[str] = set()

    if isinstance(raw_aliases, str):
        _add_unique(aliases, seen, raw_aliases)
    elif isinstance(raw_aliases, list):
        for item in raw_aliases:
            _add_unique(aliases, seen, str(item))

    return tuple(aliases)


def parse_note_date(raw_value: Any) -> datetime | None:
    """Parse a frontmatter `created`/`date` value; returns None (never raises) if unparseable."""
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value
    if isinstance(raw_value, date):
        # PyYAML auto-converts bare "YYYY-MM-DD" values to `date`, not `datetime`.
        return datetime(raw_value.year, raw_value.month, raw_value.day)

    parsed = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if isinstance(result, datetime) else None
