# lilbee — Development Guide

## Project
Local RAG knowledge base. Python 3.11+, Ollama for LLM/embeddings, LanceDB for vectors. Managed with `uv`. Task tracking with `beads` (`bd`).

## Task Tracking (beads)
```bash
bd ready                      # See what's ready to work on
bd update <id> -s in_progress # Claim a task
bd close <id>                 # Mark done
bd list                       # All issues
```
Every code change MUST be tracked as a beads task. Create tasks before starting work, close them when done.

## Commands
```bash
uv sync                       # Install dependencies
make check                    # Run all checks: lint, format, typecheck, test (same as CI)
make test                     # Tests with coverage
make lint                     # Ruff linting
make typecheck                # Mypy
make format                   # Auto-format code
uv run lilbee sync            # Sync documents to vector DB
uv run lilbee ask "question"
uv run lilbee chat
uv run lilbee status
uv run lilbee rebuild
```

## Architecture
- `src/lilbee/` — All source code
- Documents + data stored in platform-standard location (see `lilbee status`)
  - macOS: `~/Library/Application Support/lilbee/`
  - Linux: `~/.local/share/lilbee/`
  - Windows: `%LOCALAPPDATA%/lilbee/`
  - Override: `LILBEE_DATA=/path` or `--data-dir`
- All settings configurable via `LILBEE_*` env vars or CLI flags
- Auto-sync: documents/ is source of truth, data/ is rebuilt from it

## Configuration
All settings override via environment variables:
- `LILBEE_DATA` — data directory path
- `LILBEE_CHAT_MODEL` — LLM model (default: `mistral`)
- `LILBEE_EMBEDDING_MODEL` — embedding model (default: `nomic-embed-text`)
- `LILBEE_EMBEDDING_DIM` — embedding dimensions (default: `768`)
- `LILBEE_CHUNK_SIZE` — tokens per chunk (default: `512`)
- `LILBEE_CHUNK_OVERLAP` — overlap tokens (default: `100`)
- `LILBEE_TOP_K` — retrieval result count (default: `5`)

CLI also accepts `--model` / `-m` for chat model and `--data-dir` / `-d`.

## Code Quality Rules

### Testing
- **100% test coverage required** — enforced by `pytest-cov` with `fail_under = 100`
- Every public function MUST have at least one test
- Mock all external dependencies (Ollama, filesystem I/O where needed) — tests must run without a live server
- Use `pytest.mark.skipif` only for integration tests that genuinely require live services
- Use `tmp_path` fixtures for filesystem tests — never write to real paths

### Code Style
- No LangChain — raw Ollama SDK
- Type hints on all public functions
- Small functions — max ~20 lines, max 2 levels of nesting
- Low cyclomatic complexity — extract helpers when branches exceed 3
- Dataclasses for structured return types (not raw dicts)
- Logging with `logging.getLogger(__name__)` — no bare `except: pass`
- No hardcoded values — all configurable through `config.py` with env var overrides
- Imports: stdlib first, then third-party, then local — no star imports
- Lazy imports in CLI callbacks (avoid loading heavy deps at import time)
- Linting: `ruff check` + `ruff format` (line length 100)
- Type checking: `mypy` with strict settings

### Git & Workflow
- Every change tracked as a beads task
- Run `make check` before closing any task — it mirrors CI exactly
- Tests, lint, and type checks must pass before closing a task
- CI runs on every push and PR

## Key Files
- `config.py` — All settings (env-var configurable)
- `ingest.py` — Document sync engine (hash-based change detection)
- `query.py` — RAG pipeline (embed → search → generate)
- `store.py` — LanceDB operations
- `chunker.py` — Text chunking (token-based recursive)
- `code_chunker.py` — Code chunking (tree-sitter AST)
- `embedder.py` — Ollama embedding wrapper
- `cli.py` — Typer CLI with --model and --data-dir flags
