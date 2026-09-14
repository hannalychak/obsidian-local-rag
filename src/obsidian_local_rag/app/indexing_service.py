"""IndexingService: orchestrates the full ingest -> chunk -> graph -> embed -> upsert pipeline."""

from __future__ import annotations

from collections.abc import Callable

from obsidian_local_rag.app.composition import Services
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.ingest.stats import CorpusStats

# (stage_name, current, total) -> None; lets the `cli` layer drive a `rich` progress bar without
# `app` importing anything terminal-related (cli is the only layer that touches the terminal).
ProgressCallback = Callable[[str, int, int], None]


class IndexingService:
    """Stateful orchestrator for the `index` and `stats` CLI commands.

    Incremental by default: only notes in the manifest diff's `added`/`modified` sets (see
    `core.ingest.manifest`) are re-chunked and re-embedded; `deleted` notes' points are removed
    from Qdrant. `full=True` forces a full re-index regardless of the manifest.
    """

    def __init__(self, settings: Settings, services: Services) -> None:
        self._settings = settings
        self._services = services

    async def run_index(
        self, *, full: bool = False, on_progress: ProgressCallback | None = None
    ) -> None:
        """Run the ingest pipeline: walk -> parse -> chunk -> link/graph -> embed -> upsert.

        Calls `on_progress` (if given) at each pipeline stage so the caller can render progress.
        """
        raise NotImplementedError

    def compute_stats(self) -> CorpusStats:
        """Compute corpus statistics for the `stats` CLI command."""
        raise NotImplementedError
