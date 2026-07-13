# Development

## Setup

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
```

## Commands

```bash
make check      # Run all checks: lint, format, typecheck, test (same as CI)
make test       # Tests with coverage
make lint       # Ruff linting
make typecheck  # Mypy
make format     # Auto-format code
```

## Tech stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11+ |
| Package manager | uv |
| LLM runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, managed by lilbee (native GGUF) |
| Embeddings | nomic-embed-text (configurable) |
| Vector DB | LanceDB (embedded, Rust-based) |
| PDF extraction | pymupdf4llm |
| Office docs | python-docx, openpyxl, python-pptx |
| eBooks | ebooklib + BeautifulSoup |
| Image OCR | pytesseract + Pillow |
| Code parsing | tree-sitter |
| CLI | Typer + Rich |

## Key files

| File | Purpose |
|------|---------|
| `config.py` | All settings (env-var configurable) |
| `ingest.py` | Document sync engine (hash-based change detection) |
| `query.py` | RAG pipeline (embed → search → generate) |
| `store.py` | LanceDB operations |
| `chunker.py` | Text chunking (token-based recursive) |
| `code_chunker.py` | Code chunking (tree-sitter AST) |
| `embedder.py` | Thin wrapper around the LLM provider's embeddings API |
| `cli.py` | Typer CLI with --model, --data-dir, and --json flags |

## Testing

```bash
make test       # Unit tests
uv run pytest   # Full suite including RAG accuracy
```

- **100% coverage required** — enforced by `pytest-cov` with `fail_under = 100`
- All external dependencies are mocked — tests run without a live server
- Accuracy tests generate a PDF with known facts and verify correct retrieval

## Code style

- Linting: `ruff check` + `ruff format` (line length 100)
- Type checking: `mypy` with strict settings
- Type hints on all public functions
- Lazy imports in CLI callbacks
- No LangChain or other agent frameworks; direct provider calls only

## Releasing

Releases are build-once, promote-by-pointer: every `v*` tag builds all artifacts
exactly once, and the dev/beta/stable channels are pointers to tags recorded in
`packaging/channels.json`. Promoting a build republishes existing assets, never
rebuilds. The full pipeline (diagram, gates, per-manager channel matrix) is in
[architecture.md](architecture.md#release-pipeline).

```bash
make release                                 # bump version, tag, push; CI builds once, enters dev
gh workflow run verify-release.yml -f tag=vX # fires automatically after a green candidate too
make promote TAG=vX CHANNEL=beta             # verified tag -> beta surfaces
make promote TAG=vX CHANNEL=stable           # beta-soaked tag -> PyPI + stable surfaces
```

Rollback is promoting an older tag again. To promote past a flaky verify run,
re-dispatch `promote.yml` with `force=skip-verify`.

## License

MIT
