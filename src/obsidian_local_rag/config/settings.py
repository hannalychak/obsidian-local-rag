"""Application settings.

Real code (no algorithms) per CLAUDE.md's Settings exception: field declarations plus boundary
validation of the vault path.
"""

from __future__ import annotations

from pathlib import Path

import platformdirs
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, env prefix ``OBSIDIAN_RAG_``.

    Vault path resolution order: ``--vault`` CLI option (passed as an explicit constructor
    argument, which pydantic-settings already prioritizes over every other source) >
    ``OBSIDIAN_RAG_VAULT_PATH`` env var (including via the gitignored local ``.env``) > a
    project/user settings file. There is no default value.

    Known gap for a future session: the third tier ("settings file", distinct from ``.env``) is
    not yet wired up as a pydantic-settings source. Until then, resolution only covers the first
    two tiers; the CLI layer must surface a clear, actionable error when neither is present
    (`vault_path` has no default, so construction raises `pydantic.ValidationError` when unset).

    LLM backend is Ollama only, by design: this keeps every part of the pipeline — embeddings,
    vector store, and the LLM call itself — fully local, so vault content never leaves the
    machine. No API key field exists on this model for that reason. `llm_model` is a locally
    pulled Ollama model tag (e.g. `"llama3.1"`; run `ollama pull <tag>` first — this adapter does
    not pull models on demand). `ollama_base_url` defaults to the standard local Ollama port.
    """

    model_config = SettingsConfigDict(env_prefix="OBSIDIAN_RAG_", env_file=".env", extra="ignore")

    vault_path: Path
    exclude_globs: list[str] = Field(default_factory=lambda: ["trash/**"])
    data_dir: Path = Field(
        default_factory=lambda: Path(platformdirs.user_data_dir("obsidian-local-rag"))
    )

    # Verified against the installed fastembed's actual model registry (2026-09-15) — a Phase 1
    # dense-model choice ("intfloat/multilingual-e5-small") turned out not to exist in it; this
    # is a real, smaller multilingual model instead (~220MB, 384-dim, ~50 languages), a good fit
    # for constrained hardware (see README's hardware note).
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 64
    chunk_overlap_tokens: int = 64

    candidate_k: int = 50
    rrf_k: int = 60
    rerank_n: int = 30
    final_k: int = 8

    graph_max_depth: int = 1
    graph_max_nodes: int = 20
    graph_decay: float = 0.5
    graph_hub_degree_threshold: int = 50

    llm_model: str
    llm_max_tokens: int
    ollama_base_url: str = "http://localhost:11434"
    agent_max_iterations: int = 4
    # Retrieved-chunk text budget per app.context.assemble_context call, not the model's full
    # context window. Conservative default: Ollama's own context window defaults to 4096 tokens
    # (observed directly on this machine), which also has to fit the system prompt, conversation
    # history, and the model's own output alongside retrieved chunks.
    context_token_budget: int = 2000

    @field_validator("vault_path")
    @classmethod
    def _validate_vault_path(cls, value: Path) -> Path:
        """Reject anything that isn't an existing directory containing `.obsidian/`."""
        if not value.is_dir():
            raise ValueError(f"Vault path does not exist or is not a directory: {value}")
        if not (value / ".obsidian").is_dir():
            raise ValueError(
                f"Vault path does not look like an Obsidian vault (missing .obsidian/): {value}"
            )
        return value
