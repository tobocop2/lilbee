# lilbee

[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Experimental** — under active development, expect breaking changes.

<p align="center">
  <img src="demo.gif" alt="lilbee demo" width="800">
</p>

A local RAG knowledge base for humans and coding agents. Drop in PDFs, source code, markdown, or HTML — query from the terminal or let your AI coding agent search your documents via structured JSON output. Everything runs on your machine via [Ollama](https://ollama.com) — no cloud APIs, no API keys, no data leaves your computer.

**For humans:** Ask questions in the terminal, get answers with source citations from a local LLM.

**For coding agents:** Use `lilbee --json search` as a local document retrieval backend — no LLM needed. Any AI coding agent can search your knowledge base and use its own LLM to reason over the results.

## What it does

1. **Drop documents** into a folder — PDFs, `.md`, `.txt`, `.html`, `.rst`, or source code (`.py`, `.js`, `.go`, `.rs`, etc.)
2. **lilbee auto-ingests** them: extracts text, chunks it intelligently (token-based for text, AST-aware for code via tree-sitter), embeds with a local model, and stores vectors in LanceDB
3. **Query it** — ask questions from the terminal, or let coding agents search via `--json` output

```
$ lilbee ask "What is the recommended oil change interval?"

The recommended oil change interval is every 7,500 miles using
5W-30 full synthetic oil.

Sources:
  → vehicle_manual.pdf, pages 42-43
```

```bash
$ lilbee --json search "oil change interval"
{"command": "search", "query": "oil change interval", "results": [{"source": "vehicle_manual.pdf", "chunk": "...", "distance": 0.23, ...}]}
```

## Install

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Pull models (one-time):
  ```bash
  ollama pull mistral && ollama pull nomic-embed-text
  ```

### Install

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
pip install .        # or: uv tool install .
```

After this, `lilbee` is available as a command.

### Development (run from source)

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
# Prefix all commands with uv run:
uv run lilbee ask "question"
```

## Quick start

```bash
# Drop files into the documents directory
# Default location varies by platform (see Data Location below)
lilbee status  # Shows the documents path

# Ask a question (auto-ingests new documents)
lilbee ask "How do I change the oil?"

# Interactive chat
lilbee chat

# Use a different LLM
lilbee ask "Explain this code" --model llama3
```

> Running from source? Prefix with `uv run` (e.g. `uv run lilbee ask ...`).

## Commands

| Command | Description |
|---------|-------------|
| `lilbee ask "question"` | One-shot question with auto-sync |
| `lilbee search "query"` | Search for relevant chunks (no LLM needed) |
| `lilbee chat` | Interactive chat loop |
| `lilbee add <paths>` | Copy files/dirs into knowledge base and ingest |
| `lilbee sync` | Manually trigger document sync |
| `lilbee rebuild` | Nuke DB and re-ingest everything |
| `lilbee status` | Show indexed documents, paths, and models |

All commands accept `--data-dir PATH` to override the data location and `--model NAME` to override the chat model.

### JSON output for agents

Add `--json` (or `-j`) before any subcommand to get structured JSON output. This lets coding agents use lilbee as a RAG backend:

```bash
# Search documents without needing Ollama
lilbee --json search "engine specs" --top-k 5

# Full RAG answer via local Ollama
lilbee --json ask "What is the oil capacity?"

# Check index status
lilbee --json status
```

Each command returns a single JSON object to stdout. See [AGENTS.md](AGENTS.md) for the full JSON API reference.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path |
| `LILBEE_CHAT_MODEL` | `mistral` | Ollama chat model |
| `LILBEE_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding vector dimensions |
| `LILBEE_CHUNK_SIZE` | `512` | Tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk for embedding |
| `LILBEE_TOP_K` | `10` | Number of retrieval results |

## How it works

```
documents/          LanceDB             Ollama
┌──────────┐       ┌──────────┐       ┌──────────┐
│ PDFs     │──────→│ vectors  │──────→│ LLM      │
│ code     │ hash  │ + chunks │ top-K │ (any     │
│ markdown │ sync  │          │ search│  model)  │
└──────────┘       └──────────┘       └──────────┘
     ↑                                     │
     │              answer + citations     │
     └─────────────────────────────────────┘
```

- **Auto-sync**: SHA-256 hashes track which files changed. New files are ingested, modified files re-ingested, deleted files removed from the index.
- **Text chunking**: Token-based recursive splitting (512 tokens, 100 overlap) on paragraph/sentence/word boundaries.
- **Code chunking**: Tree-sitter AST parsing extracts functions and classes as natural chunks. Supports Python, JavaScript, TypeScript, Go, Rust, Java, C, C++.
- **PDF extraction**: `pymupdf4llm` converts PDFs to markdown with page tracking.

## Limitations

lilbee is designed for **local-only** RAG with small models. Keep these constraints in mind:

- **Answer quality depends on the model** — small local models (7B–13B parameters) may miss nuances or misinterpret complex questions that larger cloud models handle easily
- **Context window constraints** — very large documents produce many chunks; the model only sees the top-K most relevant ones, so some detail may be lost
- **No cloud API support by design** — everything stays on your machine, which means no access to frontier models
- **Embedding model token limits** — the default embedding model (nomic-embed-text) has a ~2000 character effective limit per chunk; longer chunks are truncated (configurable via `LILBEE_MAX_EMBED_CHARS`)

## Data location

lilbee stores documents and its vector database in a platform-standard location:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

Inside that directory:
```
lilbee/
├── documents/    # Drop your files here
└── data/
    └── lancedb/  # Vector database (auto-managed)
```

Override with `LILBEE_DATA=/path/to/dir` or `--data-dir`.

## Tech stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11+ |
| Package manager | uv |
| LLM runtime | Ollama (local, any model) |
| Embeddings | nomic-embed-text (configurable) |
| Vector DB | LanceDB (embedded, Rust-based) |
| PDF extraction | pymupdf4llm |
| Code parsing | tree-sitter |
| CLI | Typer + Rich |

## Testing

```bash
# Unit tests (no Ollama needed)
make test

# Full suite including RAG accuracy (requires Ollama + models)
uv run pytest tests/ -v
```

Accuracy tests generate a PDF with known facts and verify that lilbee retrieves correct answers with proper source attribution. Unit tests mock all external dependencies and target 100% coverage.

## License

MIT
