"""Vault filesystem walk: discovers markdown notes to ingest."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_vault_markdown_files(vault_path: Path, exclude_globs: list[str]) -> Iterator[Path]:
    """Yield every ``*.md`` file under `vault_path`.

    Always excludes `.obsidian/`, `.trash/`, and any dotfolder (name starting with `.`), in
    addition to `exclude_globs` (matched vault-relative, e.g. ``"trash/**"``). Skips iCloud
    placeholder files (``*.icloud``) and logs a warning for each one skipped.

    Complexity: O(number of filesystem entries under `vault_path`).
    """
    raise NotImplementedError


def read_note_text(path: Path) -> str:
    """Read `path` as UTF-8 with ``errors="replace"``, logging a warning if replacement occurred."""
    raise NotImplementedError
