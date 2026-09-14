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

    API keys (e.g. `ANTHROPIC_API_KEY`) are intentionally NOT fields on this model: they are read
    directly from the environment by the adapter that needs them, and must never be logged or
    persisted to a settings file.
    """

    model_config = SettingsConfigDict(env_prefix="OBSIDIAN_RAG_", env_file=".env", extra="ignore")

    vault_path: Path
    exclude_globs: list[str] = Field(default_factory=lambda: ["trash/**"])
    data_dir: Path = Field(
        default_factory=lambda: Path(platformdirs.user_data_dir("obsidian-local-rag"))
    )

    dense_model: str = "intfloat/multilingual-e5-small"
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
    agent_max_iterations: int = 4

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
