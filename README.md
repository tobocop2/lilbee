# lilbee

> NOTE: this is an early experimental project and things are likely unstable

[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Local RAG for the terminal. `pip install`, point at a folder, ask questions. Agents get JSON.

## What it does

lilbee is a local RAG tool that runs entirely on your machine. No Docker, no external databases, no cloud APIs — just Python and Ollama. You add documents, it chunks and embeds them into LanceDB, and you query from the terminal or via JSON for coding agents.

**For humans:** Drop files into a folder and ask questions. lilbee extracts text, chunks it, embeds it, and stores vectors locally. When you ask a question, it finds the most relevant chunks and passes them to a local LLM via [Ollama](https://ollama.com) to get an answer with source citations.

**For coding agents:** `lilbee search` returns relevant document chunks as JSON — no LLM needed, no Ollama required. Any agent can use it as a local retrieval backend and reason over the results with its own model. See [agent integration](docs/agent-integration.md) for the full JSON API.

## Supported formats

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | — |
| Office | `.docx`, `.xlsx`, `.pptx` | — |
| eBook | `.epub` | — |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | — |
| Text | `.md`, `.txt`, `.html`, `.rst` | — |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp` | — |

## Install

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) — only needed for standalone use (`ask`, `chat`), not for agent search
  ```bash
  ollama pull mistral && ollama pull nomic-embed-text
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
# Chat with a local LLM (requires Ollama)
lilbee

# Add documents to your knowledge base
lilbee add ~/Documents/manual.pdf ~/notes/

# Ask questions — answers come from your documents via a local LLM
lilbee ask "What is the recommended oil change interval?"

# Search documents — returns raw chunks, no LLM needed
lilbee search "oil change interval"

# Use a different model
lilbee ask "Explain this" --model llama3

# Check what's indexed
lilbee status
```

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

CLI also accepts `--data-dir` and `--model` flags.

## How it works

Documents are hashed and synced automatically — new files get ingested, modified files re-ingested, deleted files removed. Text is split into overlapping chunks (token-based for prose, AST-aware via tree-sitter for code), embedded with a local model, and stored in LanceDB. Queries embed the question, find the most relevant chunks by vector similarity, and pass them as context to the LLM.

### Data location

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

Override with `LILBEE_DATA=/path` or `--data-dir`.

## Lore

lilbee started as a personal project to chat with documents in the terminal and augment local LLMs with private knowledge they weren't trained on. It grew into a general-purpose document intelligence tool that both humans and AI coding agents can use.

## License

MIT
