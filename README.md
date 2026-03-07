# lilbee

[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Local RAG for the terminal. Ground your LLM answers in real documents — no hallucinations, no cloud, no Docker.

---

- [Why lilbee](#why-lilbee)
- [Demos](#demos)
- [Install](#install)
- [Quick start](#quick-start)
- [Interactive chat](#interactive-chat)
- [Agent integration](#agent-integration)
- [Supported formats](#supported-formats)
- [Configuration](#configuration)
- [How it works](#how-it-works)

---

## Why lilbee

Index your documents and code into a local knowledge base, then ask questions grounded in what's actually there. Most tools like this only handle code. lilbee handles PDFs, Word docs, epics — and code too, with AST-aware chunking.

- **Documents and code alike** — add anything from a vehicle manual to an entire codebase
- **Fully offline** — runs on your machine with [Ollama](https://ollama.com) and LanceDB, no cloud APIs or Docker
- **Works with AI agents** — MCP server and JSON CLI so agents can search your knowledge base too

Add files (`lilbee add`), then ask questions or search. Once indexed, `search` works without Ollama — agents use their own LLM to reason over the retrieved chunks.

## Demos

<details>
<summary><b>AI agent using lilbee (opencode)</b></summary>

![opencode + lilbee](demos/opencode.gif)

An AI coding agent shells out to `lilbee --json search` to ground its answers in your documents.
</details>

<details>
<summary><b>Interactive local offline chat</b></summary>

> [!NOTE]
> Entirely local on a 2021 M1 Pro with 32 GB RAM.

Model switching via tab completion, then a Q&A grounded in an indexed PDF.

![Interactive local offline chat](demos/chat.gif)

</details>

<details>
<summary><b>Code index and search</b></summary>

![Code search](demos/code-search.gif)

Add a codebase and search with natural language. Tree-sitter provides AST-aware chunking.
</details>

<details>
<summary><b>JSON output</b></summary>

![JSON output](demos/json.gif)

Structured JSON output for agents and scripts.
</details>

## Install

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) — only the embedding model is required for indexing and search (which is all agents need):
  ```bash
  ollama pull nomic-embed-text    # required — used for embedding during sync
  ```
  If you want to use lilbee as a standalone local chat (no cloud LLM), also pull a chat model:
  ```bash
  ollama pull mistral             # or qwen3, llama3, etc.
  ```
- **Optional** (for image OCR): `brew install tesseract` / `apt install tesseract-ocr`

### Install

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
pip install .        # or: uv tool install .
```

### Development (run from source)

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
uv run lilbee
```

## Quick start

```bash
# Check version
lilbee --version

# Chat with a local LLM (requires Ollama)
lilbee

# Add documents to your knowledge base
lilbee add ~/Documents/manual.pdf ~/notes/

# Ask questions — answers come from your documents via a local LLM
lilbee ask "What is the recommended oil change interval?"

# Search documents — returns raw chunks, no LLM needed at query time
lilbee search "oil change interval"

# Remove a document from the knowledge base
lilbee remove manual.pdf

# Use a different local chat model (requires ollama pull <model>)
lilbee ask "Explain this" --model qwen3

# Check what's indexed
lilbee status
```


## Interactive chat

Running `lilbee` or `lilbee chat` enters an interactive REPL with conversation history, streaming responses, and slash commands:

| Command | Description |
|---------|-------------|
| `/status` | Show indexed documents and config |
| `/add [path]` | Add a file or directory (tab-completes paths) |
| `/model [name]` | Show or switch chat model (tab-completes Ollama models) |
| `/version` | Show lilbee version |
| `/reset` | Delete all documents and data (asks for confirmation) |
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM.

## Agent integration

lilbee can serve as a local retrieval backend for AI coding agents via MCP or JSON CLI. See [docs/agent-integration.md](docs/agent-integration.md) for setup and usage.

## Supported formats

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | — |
| Office | `.docx`, `.xlsx`, `.pptx` | — |
| eBook | `.epub` | — |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | — |
| Text | `.md`, `.txt`, `.html`, `.rst` | — |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | — |

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path |
| `LILBEE_CHAT_MODEL` | `mistral` | Ollama chat model |
| `LILBEE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding dimensions |
| `LILBEE_CHUNK_SIZE` | `512` | Tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk for embedding |
| `LILBEE_TOP_K` | `10` | Number of retrieval results |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | Custom system prompt for RAG answers |

CLI also accepts `--model` / `-m`, `--data-dir` / `-d`, and `--version` / `-V` flags.

## How it works

Documents are hashed and synced automatically — new files get ingested, modified files re-ingested, deleted files removed. [Kreuzberg](https://github.com/Goldziher/kreuzberg) handles extraction and chunking across all document formats (PDF, Office, images via OCR, etc.), while [tree-sitter](https://tree-sitter.github.io/tree-sitter/) provides AST-aware chunking for code. Chunks are embedded via [Ollama](https://ollama.com) and stored in [LanceDB](https://lancedb.com). Ollama uses llama.cpp with native Metal support, which is significantly faster than in-process alternatives like ONNX Runtime — CoreML can't accelerate nomic-embed-text's rotary embeddings, making CPU the only ONNX path on macOS (~170ms/chunk vs near-instant with Ollama's GPU inference). Queries embed the question, find the most relevant chunks by vector similarity, and pass them as context to the LLM.

### Data location

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

Override with `LILBEE_DATA=/path` or `--data-dir`.

## License

MIT
