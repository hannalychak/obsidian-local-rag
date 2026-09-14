"""The single composition point: build concrete adapters and wire them behind core Protocols.

This is the ONLY module allowed to import both `core.protocols` and concrete `adapters` types
together (CLAUDE.md, section 3.1) — nothing else in `core` may import from `adapters`.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidian_local_rag.config.settings import Settings
from obsidian_local_rag.core.protocols import (
    DenseEmbedder,
    LLMClient,
    Reranker,
    SparseEncoder,
    VectorStore,
)


@dataclass(frozen=True, slots=True)
class Services:
    """All wired infrastructure services, ready for `app` orchestration to consume."""

    dense_embedder: DenseEmbedder
    sparse_encoder: SparseEncoder
    vector_store: VectorStore
    reranker: Reranker
    llm_client: LLMClient


def build_services(settings: Settings) -> Services:
    """Construct every adapter (`adapters.*`) from `settings` and wire it behind its Protocol."""
    raise NotImplementedError
