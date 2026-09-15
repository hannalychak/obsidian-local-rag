"""The single composition point: build concrete adapters and wire them behind core Protocols.

This is the ONLY module allowed to import both `core.protocols` and concrete `adapters` types
together (CLAUDE.md, section 3.1) — nothing else in `core` may import from `adapters`.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidian_local_rag.adapters.fastembed_encoders import (
    FastEmbedDenseEmbedder,
    FastEmbedReranker,
    FastEmbedSparseEncoder,
)
from obsidian_local_rag.adapters.llm_ollama import OllamaClient
from obsidian_local_rag.adapters.qdrant_store import QdrantVectorStore
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
    """Construct every adapter (`adapters.*`) from `settings` and wire it behind its Protocol.

    `Services.llm_client` is always `adapters.llm_ollama.OllamaClient` — the only LLM backend, by
    design, so vault content never leaves the machine. This is the ONLY module that imports it;
    everything else — `core`, the rest of `app`, `cli` — depends only on the `LLMClient` Protocol.
    """
    cache_dir = str(settings.data_dir / "models")
    return Services(
        dense_embedder=FastEmbedDenseEmbedder(settings.dense_model, cache_dir),
        sparse_encoder=FastEmbedSparseEncoder(settings.sparse_model, cache_dir),
        vector_store=QdrantVectorStore(settings.data_dir),
        reranker=FastEmbedReranker(settings.rerank_model, cache_dir),
        llm_client=OllamaClient(
            settings.llm_model, settings.ollama_base_url, settings.llm_max_tokens
        ),
    )
