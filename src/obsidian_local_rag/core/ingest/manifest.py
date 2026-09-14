"""Incremental indexing: content-hash manifest diffing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
    raise NotImplementedError


def diff_manifest(current: pd.DataFrame, previous: pd.DataFrame) -> ManifestDiff:
    """Compare `current` and `previous` manifests via vectorized set/merge operations.

    Only `added`/`modified` note_ids need re-chunking/re-embedding; only `deleted` note_ids need
    their points removed from Qdrant. A note_id present in both with an unchanged `content_hash`
    is in none of the three sets.

    Complexity: O(len(current) + len(previous)).
    """
    raise NotImplementedError


def save_manifest(data_dir: Path, current: pd.DataFrame) -> None:
    """Persist `current` as the manifest for the next incremental run."""
    raise NotImplementedError
