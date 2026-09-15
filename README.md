# obsidian-local-rag

A privacy-first, terminal-based agentic RAG assistant over a local Obsidian vault: hybrid (dense
+ sparse) retrieval, knowledge-graph expansion over wiki-links, cross-encoder reranking, and a
bounded tool-calling agent loop that streams answers with verified citations back to your source
notes.

**Privacy guarantee: everything runs locally, by design** — embeddings, the vector store, and the
LLM itself (via [Ollama](https://ollama.com)) all execute on your machine. Vault content never
leaves it: no note text, question, or answer is ever sent to a third-party server. The vault is
also read-only — this tool never writes, moves, renames, or deletes anything inside it.

## Why

A vault of course notes, research notes, and daily jottings grows fast — dozens of files across
multiple subjects, split across languages, cross-referenced with wiki-links that make sense in
the moment but are hard to hold in your head a month later. Full-text search finds a note; it
doesn't answer "what did I write about X" when the answer is spread across three notes linked to
each other, or phrased differently than you're searching for.

Tools like [NotebookLM](https://notebooklm.google) solve the "chat with my documents" problem the
obvious way — upload your notes to their servers and let a cloud model index them. That means
handing over coursework, unfinished ideas, and private research just to search your own files.
This project exists to get the same agentic RAG experience — hybrid search, graph-aware expansion
of related notes, cited answers — without that trade-off: embeddings, the vector store, and the
LLM itself all run on the machine the notes already live on. Nothing about the vault's content is
ever uploaded anywhere. Privacy isn't a mode you opt into — the architecture has no code path that
could send vault content to a network service even if you wanted it to (see `CLAUDE.md`'s
"LLM backend: Ollama only, by design").

## Install

Requires macOS (Apple Silicon), Python 3.12, [`uv`](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) installed via Homebrew with a model already pulled
(`brew install ollama && ollama pull llama3.2:3b`) — `chat` starts and stops the Ollama service
for you (see below), so you don't need to run it yourself.

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

`chat` manages the Ollama service automatically: it starts it (`brew services start ollama`) if
it isn't already running, and stops it again when you exit — but only if it was the one that
started it, so it never interrupts an Ollama instance you were already running yourself. No
manual `brew services` commands needed for normal use.

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

Every command accepts `--vault PATH` (overrides `OBSIDIAN_RAG_VAULT_PATH`) and `--debug` (shows
a full traceback instead of a short error panel).

### `index` — build/update the local search index

```bash
uv run obsidian-rag index [--vault PATH] [--full]
```

Walks the vault, chunks each note, computes local embeddings, and stores everything in an
embedded Qdrant collection on disk. Incremental by default — only re-processes notes whose
content changed since the last run (tracked via a content-hash manifest); `--full` ignores that
and rebuilds everything. Run this at least once before `chat`; doesn't need Ollama.

### `stats` — corpus overview

```bash
uv run obsidian-rag stats [--vault PATH]
```

Note/chunk counts, token distribution per chunk, top tags, top folders, orphan note count,
unresolved-link count. Doesn't need `index` to have run first (walks the vault fresh each time)
or Ollama.

### `graph-info` — the wiki-link knowledge graph

```bash
uv run obsidian-rag graph-info [--vault PATH] [--note NOTE_ID]
```

No `--note`: vault-wide overview — orphan notes (no links in or out), unresolved link targets
("ghosts" — links to notes that don't exist), and hub notes (heavily-linked notes, suppressed
during graph expansion so one index page doesn't dominate every answer).
`--note "path/to/Note"`: that note's degree, hub/ghost status, and full neighbor list (both
outlinks and backlinks). Also doesn't need `index` or Ollama.

### `chat` — interactive agentic assistant

```bash
uv run obsidian-rag chat [--vault PATH]
```

Needs the vault already `index`ed. Starts Ollama automatically if it isn't running already, and
stops it again on exit (only if this command was the one that started it — see the hardware note
above). The agent decides on its own when to call `search_vault` (hybrid dense+sparse retrieval,
reranked), `expand_graph` (follow wiki-links from notes it already found), and `read_note` (read
one note in full), and cites what it used with `[S1]`-style keys tied to `path:line`.

**In the REPL:**

| Input | Effect |
|---|---|
| *(a question)* | Sends it to the agent; the answer streams in live |
| `/sources` | Shows citations from the last answer (file, line range, `obsidian://` link) |
| `/clear` | Wipes conversation history, starts fresh |
| `/exit` | Quit |
| Ctrl+D | Also quits |
| Ctrl+C (while a response is streaming) | Cancels that response; session stays open |
| Ctrl+C (at an empty prompt) | Clears the input line |

Conversation history persists across turns within one session (not saved between sessions);
prompt history (what you typed) does persist across sessions, stored under
`~/Library/Application Support/obsidian-local-rag/`.

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
