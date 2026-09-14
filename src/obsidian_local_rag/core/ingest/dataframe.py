"""Notes DataFrame construction — the ingest & analytics plane (never used at query time)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from obsidian_local_rag.domain.models import Note


def build_notes_dataframe(notes: Sequence[Note]) -> pd.DataFrame:
    """Build the notes DataFrame from a list of `Note` records in a single vectorized pass.

    Never appends rows in a loop: constructed once from `notes` via `pd.DataFrame.from_records`
    (or equivalent). Derived columns (folder, token estimate, parsed dates via
    `pd.to_datetime(errors="coerce")`, exploded/normalized tags, content hash) are computed with
    vectorized pandas operations — no `iterrows`, no row-wise `apply` where a vectorized
    alternative exists.

    Complexity: O(len(notes)).
    """
    raise NotImplementedError
