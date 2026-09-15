"""IndexingService: orchestrates the full ingest -> chunk -> graph -> embed -> upsert pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from obsidian_local_rag.app.composition import Services
from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.chunking.chunker import chunk_note
from obsidian_local_rag.core.graph.builder import NoteGraph
from obsidian_local_rag.core.ingest.dataframe import build_notes_dataframe
from obsidian_local_rag.core.ingest.frontmatter import (
    normalize_aliases,
    normalize_tags,
    parse_frontmatter,
    parse_note_date,
)
from obsidian_local_rag.core.ingest.manifest import (
    diff_manifest,
    load_previous_manifest,
    save_manifest,
)
from obsidian_local_rag.core.ingest.stats import CorpusStats, compute_corpus_stats
from obsidian_local_rag.core.ingest.walker import iter_vault_markdown_files, read_note_text
from obsidian_local_rag.core.links.parser import extract_links
from obsidian_local_rag.core.links.resolver import resolve_links
from obsidian_local_rag.domain.models import Chunk, Note

# (stage_name, current, total) -> None; lets the `cli` layer drive a `rich` progress bar without
# `app` importing anything terminal-related (cli is the only layer that touches the terminal).
ProgressCallback = Callable[[str, int, int], None]

_MANIFEST_COLUMNS = ["note_id", "content_hash"]


def load_notes(vault_path: Path, exclude_globs: list[str]) -> dict[str, Note]:
    """Walk and parse every note in `vault_path`. Shared by indexing, stats, and chat startup
    (graph expansion needs the same note set `compute_stats` does).
    """
    notes: dict[str, Note] = {}
    for path in iter_vault_markdown_files(vault_path, exclude_globs):
        text = read_note_text(path)
        meta, body = parse_frontmatter(text)
        note_id = str(path.relative_to(vault_path).with_suffix(""))
        notes[note_id] = Note(
            note_id=note_id,
            path=path,
            title=note_id.rsplit("/", maxsplit=1)[-1],
            frontmatter=meta,
            body=body,
            content_hash=hashlib.sha256(body.encode()).hexdigest(),
            tags=normalize_tags(meta.get("tags"), body),
            aliases=normalize_aliases(meta.get("aliases")),
            created=parse_note_date(meta.get("created")),
            modified=None,
        )
    return notes


def load_notes_and_graph(
    vault_path: Path, exclude_globs: list[str]
) -> tuple[dict[str, Note], NoteGraph]:
    """`load_notes` plus the resolved-link knowledge graph built from them."""
    notes = load_notes(vault_path, exclude_globs)
    raw_links = [
        link for note_id, note in notes.items() for link in extract_links(note_id, note.body)
    ]
    resolved_links = resolve_links(raw_links, notes)
    return notes, NoteGraph.build(notes, resolved_links)


class IndexingService:
    """Stateful orchestrator for the `index` and `stats` CLI commands.

    Incremental by default: only notes in the manifest diff's `added`/`modified` sets (see
    `core.ingest.manifest`) are re-chunked and re-embedded; `deleted` notes' points are removed
    from Qdrant. `full=True` forces a full re-index regardless of the manifest.

    `compute_stats` never touches `self._services` (no embeddings/vector store/LLM needed) — it
    walks, chunks, and graphs the vault fresh every call, same as `run_index`'s first stages.
    """

    def __init__(self, settings: Settings, services: Services) -> None:
        self._settings = settings
        self._services = services

    def _chunk_notes(self, notes: list[Note]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for note in notes:
            chunks.extend(
                chunk_note(
                    note,
                    self._settings.chunk_max_tokens,
                    self._settings.chunk_min_tokens,
                    self._settings.chunk_overlap_tokens,
                )
            )
        return chunks

    def _embedding_text(self, chunk: Chunk, note_title: str) -> str:
        """The breadcrumb-prefixed text used for embedding; `Chunk.text` itself stays clean."""
        breadcrumb = " > ".join((note_title, *chunk.header_path))
        return f"{breadcrumb}\n{chunk.text}"

    async def run_index(
        self, *, full: bool = False, on_progress: ProgressCallback | None = None
    ) -> None:
        """Run the ingest pipeline: walk -> parse -> chunk -> link/graph -> embed -> upsert.

        Calls `on_progress` (if given) at each pipeline stage so the caller can render progress.
        """

        def report(stage: str, current: int, total: int) -> None:
            if on_progress is not None:
                on_progress(stage, current, total)

        report("walk", 0, 1)
        notes = load_notes(self._settings.vault_path, self._settings.exclude_globs)
        report("walk", 1, 1)

        current_manifest = pd.DataFrame(
            [{"note_id": n.note_id, "content_hash": n.content_hash} for n in notes.values()],
            columns=_MANIFEST_COLUMNS,
        )

        if full:
            to_upsert_ids: frozenset[str] = frozenset(notes.keys())
            to_delete_ids: frozenset[str] = frozenset()
        else:
            previous_manifest = load_previous_manifest(self._settings.data_dir)
            diff = diff_manifest(current_manifest, previous_manifest)
            to_upsert_ids = diff.added | diff.modified
            to_delete_ids = diff.deleted

        dense_dim = len(self._services.dense_embedder.embed(["dimension probe"])[0])
        self._services.vector_store.ensure_collection(dense_dim=dense_dim)

        purge_ids = to_delete_ids | to_upsert_ids
        if purge_ids:
            report("purge", 0, 1)
            self._services.vector_store.delete(list(purge_ids))
            report("purge", 1, 1)

        notes_to_chunk = [notes[note_id] for note_id in to_upsert_ids]
        report("chunk", 0, len(notes_to_chunk))
        chunks: list[Chunk] = []
        for i, note in enumerate(notes_to_chunk):
            chunks.extend(
                chunk_note(
                    note,
                    self._settings.chunk_max_tokens,
                    self._settings.chunk_min_tokens,
                    self._settings.chunk_overlap_tokens,
                )
            )
            report("chunk", i + 1, len(notes_to_chunk))

        if chunks:
            embedding_texts = [
                self._embedding_text(chunk, notes[chunk.note_id].title) for chunk in chunks
            ]
            report("embed", 0, len(chunks))
            dense_vectors = self._services.dense_embedder.embed(embedding_texts)
            sparse_vectors = self._services.sparse_encoder.encode(embedding_texts)
            report("embed", len(chunks), len(chunks))

            report("upsert", 0, 1)
            self._services.vector_store.upsert(chunks, dense_vectors, sparse_vectors)
            report("upsert", 1, 1)

        save_manifest(self._settings.data_dir, current_manifest)

    def compute_stats(self) -> CorpusStats:
        """Compute corpus statistics for the `stats` CLI command."""
        notes, graph = load_notes_and_graph(self._settings.vault_path, self._settings.exclude_globs)
        chunks = self._chunk_notes(list(notes.values()))

        notes_df = build_notes_dataframe(list(notes.values()))
        chunks_df = (
            pd.DataFrame.from_records([{"token_estimate": c.token_estimate} for c in chunks])
            if chunks
            else pd.DataFrame(columns=["token_estimate"])
        )
        stats = compute_corpus_stats(notes_df, chunks_df)

        return replace(
            stats,
            orphan_note_ids=sorted(graph.orphans()),
            unresolved_link_targets=sorted(graph.ghosts()),
        )
