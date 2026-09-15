# obsidian-local-rag

A terminal-based agentic RAG assistant over a local Obsidian vault: hybrid (dense + sparse)
retrieval, knowledge-graph expansion over wiki-links, cross-encoder reranking, and a bounded
tool-calling agent loop that streams answers with verified citations back to your source notes.

Everything runs locally, by design — embeddings, the vector store, and the LLM itself (via
[Ollama](https://ollama.com)). Vault content never leaves the machine. The vault is read-only —
this tool never writes, moves, renames, or deletes anything inside it.

## Install

Requires macOS (Apple Silicon), Python 3.12, [`uv`](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) running locally with a model already pulled (`ollama pull llama3.2:3b`).

```bash
uv sync
```

### Hardware note

Running the LLM locally means it competes with everything else on your machine for RAM while
it's loaded — this is inherent to local inference, not something this tool can fully hide.
Measured on an M3 MacBook with 8GB RAM:

| Model | Size on disk | RAM while loaded | Typical response time |
|---|---|---|---|
| `llama3.2:3b` (default) | ~2 GB | manageable | a few seconds |
| `llama3.1` (8B) | ~4.9 GB | over half the machine's RAM | tens of seconds, and the rest of the machine visibly lags |

If you have 16GB+ RAM, a bigger model (`llama3.1` or similar) will give noticeably better
answers. On 8GB machines, stick with a 3B-class model like the default.

Ollama keeps a model resident in memory for a while after use (its own idle timeout) rather than
releasing it immediately, which is why it can still show up in Activity Monitor after you've
closed `chat`. Stop it manually anytime with `brew services stop ollama` (or `ollama stop
<model>` to unload just the model); start it again with `brew services start ollama`.

## Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

- `OBSIDIAN_RAG_VAULT_PATH` — path to your Obsidian vault (must contain `.obsidian/`). Can also
  be passed per-command via `--vault`, which takes priority over the env var.
- `OBSIDIAN_RAG_LLM_MODEL` — a model tag you've already pulled with `ollama pull` (default:
  `llama3.2:3b`; see the hardware note above before switching to a bigger one).
- `OBSIDIAN_RAG_OLLAMA_BASE_URL` — defaults to `http://localhost:11434`.

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
