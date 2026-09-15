"""Incremental indexing: content-hash manifest diffing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_MANIFEST_COLUMNS = ["note_id", "content_hash"]
_MANIFEST_FILENAME = "manifest.csv"


@dataclass(frozen=True, slots=True)
class ManifestDiff:
    """Sets of `note_id`s that changed since the previous manifest, disjoint by construction."""

    added: frozenset[str]
    modified: frozenset[str]
    deleted: frozenset[str]


def load_previous_manifest(data_dir: Path) -> pd.DataFrame:
    """Load the previous run's manifest (columns: `note_id`, `content_hash`) from `data_dir`.

    Returns an empty DataFrame with the right columns if no manifest exists yet (first run).
    """
    manifest_path = data_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    return pd.read_csv(manifest_path, dtype={"note_id": str, "content_hash": str})


def diff_manifest(current: pd.DataFrame, previous: pd.DataFrame) -> ManifestDiff:
    """Compare `current` and `previous` manifests via vectorized set/merge operations.

    Only `added`/`modified` note_ids need re-chunking/re-embedding; only `deleted` note_ids need
    their points removed from Qdrant. A note_id present in both with an unchanged `content_hash`
    is in none of the three sets.

    Complexity: O(len(current) + len(previous)).
    """
    merged = current[_MANIFEST_COLUMNS].merge(
        previous[_MANIFEST_COLUMNS],
        on="note_id",
        how="outer",
        suffixes=("_current", "_previous"),
        indicator=True,
    )

    added = frozenset(merged.loc[merged["_merge"] == "left_only", "note_id"])
    deleted = frozenset(merged.loc[merged["_merge"] == "right_only", "note_id"])

    both = merged.loc[merged["_merge"] == "both"]
    changed = both["content_hash_current"] != both["content_hash_previous"]
    modified = frozenset(both.loc[changed, "note_id"])

    return ManifestDiff(added=added, modified=modified, deleted=deleted)


def save_manifest(data_dir: Path, current: pd.DataFrame) -> None:
    """Persist `current` as the manifest for the next incremental run."""
    data_dir.mkdir(parents=True, exist_ok=True)
    current[_MANIFEST_COLUMNS].to_csv(data_dir / _MANIFEST_FILENAME, index=False)
