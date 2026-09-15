# obsidian-local-rag

## Layering (dependency direction strictly downward)

```
cli        -> typer commands, chat REPL, rendering (only layer that touches the terminal)
app        -> orchestration: IndexingService, ChatSession, AgentLoop, context assembly
core       -> ingest, chunking, links/graph, retrieval (dense, sparse, fusion), rerank
              + Protocol interfaces for infrastructure
adapters   -> Qdrant store, fastembed encoders/reranker, LLM client implementations
domain     -> pure data models, no I/O, imports nothing from other layers
config     -> Settings (pydantic-settings)
```

- `core` depends on Protocols (`core/protocols.py`), never on `adapters`. Wiring happens in one
  composition function: `app/composition.py::build_services`.
- Third-party SDK imports (`qdrant_client`, `fastembed`, `httpx`) appear ONLY in `adapters`.

## LLM backend: Ollama only, by design

`adapters/llm_ollama.py::OllamaClient` is the only `LLMClient` implementation — a deliberate
privacy decision, not a temporary gap: embeddings and the vector store were already fully local,
and this closes the last path by which vault content could leave the machine (no cloud LLM API,
no API key anywhere in this codebase). `core`, the rest of `app`, and `cli` depend only on the
`LLMClient` Protocol; `app/composition.py::build_services` is the only place that imports the
concrete adapter. Do not add a second LLM backend without the user explicitly asking for it —
that would reopen the privacy question this decision settled.

## Anti-over-engineering rules

- Protocols ONLY at real seams: `DenseEmbedder`, `SparseEncoder`, `VectorStore`, `Reranker`,
  `LLMClient`. No other abstract interfaces or base classes.
- One responsibility per module; target < 300 lines per module. No `utils.py` / `helpers.py` /
  `common.py`.
- No speculative config options, no features beyond the current technical spec.
- Pure algorithms (RRF, BFS expansion scoring, chunk splitting, link parsing, link resolution,
  citation validation) are plain module-level functions, not methods on stateful classes.
- Classes are for things with state or lifecycle (stores, sessions, services, graph wrapper).

## Typing rules

- Full annotations everywhere; must pass `mypy --strict`. No `Any` without a justifying comment.
- Pydantic v2 at boundaries only: `Settings`, LLM tool input/output schemas, frontmatter
  validation. `@dataclass(frozen=True, slots=True)` for internal domain objects.
- All filesystem paths via `pathlib.Path`. Vault paths may contain spaces — always quote paths in
  shell commands; never build paths via string concatenation.
- Note identity: `note_id` = vault-relative POSIX path without extension.

## Vault is read-only

Never write, move, rename, or delete anything inside the vault. The vault path is a runtime
input only (`--vault` CLI option > `OBSIDIAN_RAG_VAULT_PATH` env var), never hardcoded in source,
tests, or docs.

## Verification commands

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run obsidian-rag --help
uv run obsidian-rag index --help
```

Never weaken configuration (disabling rules, loosening mypy, deleting tests) to make these pass.
