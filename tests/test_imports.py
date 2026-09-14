"""Smoke test: every module in the package must import cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "obsidian_local_rag",
    "obsidian_local_rag.domain.models",
    "obsidian_local_rag.config.settings",
    "obsidian_local_rag.core.protocols",
    "obsidian_local_rag.core.ingest.walker",
    "obsidian_local_rag.core.ingest.frontmatter",
    "obsidian_local_rag.core.ingest.dataframe",
    "obsidian_local_rag.core.ingest.manifest",
    "obsidian_local_rag.core.ingest.stats",
    "obsidian_local_rag.core.chunking.markdown",
    "obsidian_local_rag.core.chunking.splitter",
    "obsidian_local_rag.core.chunking.chunker",
    "obsidian_local_rag.core.links.parser",
    "obsidian_local_rag.core.links.resolver",
    "obsidian_local_rag.core.graph.builder",
    "obsidian_local_rag.core.graph.expand",
    "obsidian_local_rag.core.retrieval.fusion",
    "obsidian_local_rag.core.retrieval.hybrid_search",
    "obsidian_local_rag.core.rerank.rerank",
    "obsidian_local_rag.core.citations",
    "obsidian_local_rag.adapters.qdrant_store",
    "obsidian_local_rag.adapters.fastembed_encoders",
    "obsidian_local_rag.adapters.llm_anthropic",
    "obsidian_local_rag.app.composition",
    "obsidian_local_rag.app.indexing_service",
    "obsidian_local_rag.app.tools",
    "obsidian_local_rag.app.context",
    "obsidian_local_rag.app.agent_loop",
    "obsidian_local_rag.app.chat_session",
    "obsidian_local_rag.cli.app",
    "obsidian_local_rag.cli.rendering",
    "obsidian_local_rag.cli.chat_repl",
    "obsidian_local_rag.cli.commands.index_cmd",
    "obsidian_local_rag.cli.commands.chat_cmd",
    "obsidian_local_rag.cli.commands.stats_cmd",
    "obsidian_local_rag.cli.commands.graph_info_cmd",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
