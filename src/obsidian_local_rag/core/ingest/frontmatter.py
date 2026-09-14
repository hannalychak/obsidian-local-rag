"""Frontmatter parsing and key normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the note body using `python-frontmatter`.

    On malformed YAML, logs a warning and returns `({}, raw_text)` (frontmatter discarded, full
    original text kept as body) rather than raising.
    """
    raise NotImplementedError


def normalize_tags(raw_tags: Any, body: str) -> tuple[str, ...]:
    """Normalize a frontmatter `tags` value into a deduplicated tuple, order-preserving.

    Accepts a string or list, with or without a leading `#`. Per Phase 1 decision, also extracts
    inline body `#tags` (outside code fences and inline code) from `body` and unions them in,
    since this vault's notes carry zero frontmatter but use inline tags exclusively.
    """
    raise NotImplementedError


def normalize_aliases(raw_aliases: Any) -> tuple[str, ...]:
    """Normalize a frontmatter `aliases` value (string or list) into a deduplicated tuple."""
    raise NotImplementedError


def parse_note_date(raw_value: Any) -> datetime | None:
    """Parse a frontmatter `created`/`date` value; returns None (never raises) if unparseable."""
    raise NotImplementedError
