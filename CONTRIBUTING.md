# Contributing to lilbee

Thanks for your interest in contributing. lilbee is a local search engine you can talk to: index your files, code, and websites, then query them with models that run on your own machine.

For day-to-day coding conventions and the full set of project rules, [`AGENTS.md`](AGENTS.md) is the source of truth. This file is the short on-ramp.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- `git`

No external services are required. lilbee downloads and runs models locally via in-process `llama.cpp` (native GGUF). There is no Ollama dependency, though lilbee can talk to a running Ollama or LM Studio if you have one.

## Getting started

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
uv run lilbee self-check   # downloads a tiny model, runs an inference + embedding
uv run lilbee              # launch the terminal app
```

Optional extras (only needed if you work on those areas):

```bash
uv sync --extra crawler    # website crawling (/crawl, Playwright)
uv sync --extra litellm    # hosted-provider bridge (Ollama, LM Studio, cloud)
uv sync --extra graph      # concept-graph search
```

## Project layout

All source lives under `src/lilbee/`:

| Path | Purpose |
|------|---------|
| `app/` | Shared use-case orchestration consumed by the CLI, server, MCP, and TUI |
| `cli/` | Typer CLI and the Textual TUI |
| `core/config/` | All settings (env-var and `config.toml` configurable) |
| `data/ingest/` | Document sync engine (hash-based change detection) |
| `data/store/` | LanceDB vector-store operations |
| `data/chunk.py`, `data/code_chunker.py` | Text and tree-sitter code chunking |
| `retrieval/query/` | RAG pipeline (embed → search → generate) |
| `retrieval/embedder.py` | Embedding wrapper over the provider abstraction |
| `providers/` | LLM provider abstraction (llama-cpp default, litellm bridge, factory) |
| `catalog/`, `modelhub/` | Model discovery (Hugging Face) and lifecycle (install, remove, run) |
| `crawler/` | Website crawling to markdown |
| `wiki/` | Concept/entity wiki layer |
| `server/`, `mcp_server.py` | HTTP API and MCP server |
| `runtime/` | Process lifecycle (launcher, asyncio loop, cancellation, progress, locks) |

Tests live under `tests/`, mirroring the source layout, with `tests/integration/` for multi-system flows.

## Before submitting

```bash
make format        # auto-format (ruff)
make check         # lint + format-check + typecheck + tests (same as CI)
```

`make check` is the gate CI runs. Run it locally and make sure it's green before opening a PR.

- **100% test coverage is enforced** (`pytest-cov` with `fail_under = 100`). Add tests for any new code; every test must assert something meaningful (no coverage padding).
- **Tests run without a live server** — external dependencies are mocked. Integration tests under `tests/integration/` exercise real backends.
- `make lint` also runs a diff-scoped code-smell gate (`scripts/check_style_rules.py`) against `main`. Write rule-compliant code the first time rather than copying older adjacent patterns.

Useful individual targets:

```bash
make test          # tests with coverage
make lint          # ruff + style-rule check
make typecheck     # mypy (strict)
make format-check  # verify formatting without changing files
```

## Code style and guidelines

- Open an issue before large changes so we can agree on the approach first.
- Type hints on all public functions; mypy runs in strict mode. Annotate types precisely instead of reaching for `getattr(obj, "field", default)`.
- One-line docstrings by default. Describe what the code is, not what it used to be.
- No agent frameworks (LangChain, etc.); calls go through lilbee's own provider abstraction.
- Keep changes minimal and focused. Don't add re-export shims or back-compat scaffolding when moving symbols. Update every importer and mock target in the same commit.
- When adding an enum variant, config field, or new entry point (CLI flag, env var, server route, MCP param), update every parallel dispatch/validation site in the same commit.

See [`AGENTS.md`](AGENTS.md) for the complete set of rules (DRY, layering, TUI conventions, cross-cutting-change checklist, self-review checklist).

## Commits and PRs

- Keep commits focused and descriptive. No "Phase N" numbering, no `Co-Authored-By` lines.
- PR descriptions are short and human-readable: a `## Problem` and `## Solution` paragraph. No implementation dumps, internal names, or test-plan sections.
- Run the full `make check` gate before every push.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
