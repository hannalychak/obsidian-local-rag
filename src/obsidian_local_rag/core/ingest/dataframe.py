"""Notes DataFrame construction — the ingest & analytics plane (never used at query time)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from obsidian_local_rag.domain.models import Note

_COLUMNS = [
    "note_id",
    "path",
    "title",
    "body",
    "content_hash",
    "tags",
    "aliases",
    "created",
    "modified",
    "folder",
    "token_estimate",
]


def build_notes_dataframe(notes: Sequence[Note]) -> pd.DataFrame:
    """Build the notes DataFrame from a list of `Note` records in a single vectorized pass.

    Never appends rows in a loop: constructed once from `notes` via `pd.DataFrame.from_records`
    (or equivalent). Derived columns (folder, token estimate, parsed dates via
    `pd.to_datetime(errors="coerce")`, exploded/normalized tags, content hash) are computed with
    vectorized pandas operations — no `iterrows`, no row-wise `apply` where a vectorized
    alternative exists.

    Complexity: O(len(notes)).
    """
    if not notes:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame.from_records(
        [
            {
                "note_id": note.note_id,
                "path": note.path,
                "title": note.title,
                "body": note.body,
                "content_hash": note.content_hash,
                "tags": note.tags,
                "aliases": note.aliases,
                "created": note.created,
                "modified": note.modified,
            }
            for note in notes
        ]
    )

    has_folder = df["note_id"].str.contains("/")
    df["folder"] = df["note_id"].str.rsplit("/", n=1).str[0].where(has_folder, "")
    df["token_estimate"] = df["body"].str.split().str.len()
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["modified"] = pd.to_datetime(df["modified"], errors="coerce")

    return df[_COLUMNS]
