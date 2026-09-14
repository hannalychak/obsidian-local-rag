# obsidian-local-rag

A terminal-based agentic RAG assistant over a local Obsidian vault: hybrid (dense + sparse)
retrieval, knowledge-graph expansion over wiki-links, cross-encoder reranking, and a bounded
tool-calling agent loop that streams answers with verified citations back to your source notes.

Everything runs locally except the LLM call itself. The vault is read-only — this tool never
writes, moves, renames, or deletes anything inside it.

## Install

Requires macOS (Apple Silicon), Python 3.12, and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

- `OBSIDIAN_RAG_VAULT_PATH` — path to your Obsidian vault (must contain `.obsidian/`). Can also
  be passed per-command via `--vault`, which takes priority over the env var.
- `ANTHROPIC_API_KEY` — your Anthropic API key.

The vault path is never hardcoded anywhere in this repository.

## Commands

```bash
uv run obsidian-rag index [--vault PATH] [--full]   # incremental index (default), or full re-index
uv run obsidian-rag chat [--vault PATH]              # interactive chat session
uv run obsidian-rag stats [--vault PATH]             # corpus statistics
uv run obsidian-rag graph-info [--vault PATH] [--note NOTE_ID]  # knowledge-graph info
```

Add `--debug` to any command to show full tracebacks instead of a short error panel.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

See `CLAUDE.md` for the architecture and coding rules this project follows.

## PyCharm setup

Point the project interpreter at the `uv`-managed virtualenv: **Settings → Project → Python
Interpreter → Add Interpreter → Existing** and select `.venv/bin/python` in the project root.
