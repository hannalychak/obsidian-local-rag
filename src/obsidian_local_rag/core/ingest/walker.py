"""Vault filesystem walk: discovers markdown notes to ingest."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def iter_vault_markdown_files(vault_path: Path, exclude_globs: list[str]) -> Iterator[Path]:
    """Yield every ``*.md`` file under `vault_path`.

    Always excludes `.obsidian/`, `.trash/`, and any dotfolder (name starting with `.`), in
    addition to `exclude_globs` (matched vault-relative, e.g. ``"trash/**"``). Skips iCloud
    placeholder files (``*.icloud``) and logs a warning for each one skipped.

    Complexity: O(number of filesystem entries under `vault_path`).
    """
    for path in sorted(vault_path.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(vault_path)

        if any(part.startswith(".") for part in relative.parts[:-1]):
            continue  # inside a dotfolder (.obsidian/, .trash/, plugin data dirs, etc.)

        if path.name.endswith(".md.icloud"):
            logger.warning("Skipping undownloaded iCloud placeholder: %s", relative)
            continue

        if path.suffix != ".md" or path.name.startswith("."):
            continue

        relative_posix = relative.as_posix()
        if any(fnmatch.fnmatch(relative_posix, pattern) for pattern in exclude_globs):
            continue

        yield path


def read_note_text(path: Path) -> str:
    """Read `path` as UTF-8 with ``errors="replace"``, logging a warning if replacement occurred."""
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    if "�" in text:
        logger.warning("Non-UTF-8 bytes replaced while reading: %s", path)
    return text
