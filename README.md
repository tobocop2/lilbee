# lilbee

> NOTE: this is an early experimental project and things are likely unstable

[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Local RAG for the terminal. `pip install`, point at a folder, ask questions. Agents get JSON.

---

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [Interactive chat](#interactive-chat)
- [Agent integration](#agent-integration)
- [Supported formats](#supported-formats)
- [Configuration](#configuration)
- [How it works](#how-it-works)

---

## What it does

lilbee is a local RAG tool that runs entirely on your machine. No Docker, no external databases, no cloud APIs — just Python and [Ollama](https://ollama.com).

**For humans:** Drop files into a folder and ask questions in an interactive chat with slash commands, tab completion, and streaming responses. lilbee extracts text, chunks it, embeds it with a local model, and stores vectors in LanceDB. When you ask a question, it finds the most relevant chunks and passes them to a local LLM to get an answer with source citations.

**For coding agents:** lilbee exposes an MCP server and a JSON CLI so any agent can search your indexed documents directly. `search` returns pre-embedded chunks without calling Ollama at query time — agents use their own LLM to reason over the results.

**Ollama is needed for two things:** (1) embedding documents during `sync`/indexing, and (2) interactive chat and `ask`. Once documents are indexed, `search` works without Ollama.

## Install

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) — needed for embedding (during sync) and interactive chat/ask, not for `search`
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

# Use a different model
lilbee ask "Explain this" --model llama3

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
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM.

## Agent integration

lilbee can serve as a local retrieval backend for AI coding agents. Two integration methods:

### MCP server

For agents that support [MCP](https://modelcontextprotocol.io), lilbee ships a built-in MCP server:

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "lilbee",
      "args": ["mcp"]
    }
  }
}
```

Add this to your MCP client's config and the following tools become available:

| Tool | Description | Requires Ollama |
|------|-------------|-----------------|
| `lilbee_search(query, top_k)` | Search for relevant document chunks | No |
| `lilbee_ask(question)` | Ask a question answered by local RAG | Yes |
| `lilbee_status()` | Show indexed documents, config, chunk counts | No |
| `lilbee_sync()` | Sync documents directory with vector store | Yes (embedding) |

Prefer `lilbee_search` — it queries pre-computed embeddings without calling Ollama at query time.

### JSON CLI

For agents that shell out, every command supports `--json` (or `-j`) for structured output:

```bash
# Search — no Ollama needed at query time
lilbee --json search "query" --top-k 5
# → {"command": "search", "query": "...", "results": [...]}

# Ask — full local RAG via Ollama
lilbee --json ask "question"
# → {"command": "ask", "question": "...", "answer": "...", "sources": [...]}

# Check what's indexed
lilbee --json status
# → {"command": "status", "config": {...}, "sources": [...], "total_chunks": N}

# Trigger document sync
lilbee --json sync
# → {"command": "sync", "added": [...], "updated": [...], "removed": [...], ...}
```

All JSON commands return a single object on stdout. Errors return non-zero exit + `{"error": "message"}`. See [docs/agent-integration.md](docs/agent-integration.md) for the full reference.

## Supported formats

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | — |
| Office | `.docx`, `.xlsx`, `.pptx` | — |
| eBook | `.epub` | — |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | — |
| Text | `.md`, `.txt`, `.html`, `.rst` | — |
| Code | 154 extensions via [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) — `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.cs`, `.swift`, `.kt`, `.scala`, `.lua`, `.zig`, `.ex`, `.hs`, `.dart`, `.jl`, `.r`, `.ml`, `.el`, `.clj`, `.sol`, `.sql`, `.sh`, `.yaml`, `.json`, `.toml`, `.tf`, and [many more](src/lilbee/code_chunker.py) | — |

Code files use AST-aware chunking via tree-sitter — functions and classes are kept intact rather than split mid-definition. Languages with AST definition rules get structural chunking; all others fall back to token-based chunking.

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

Documents are hashed and synced automatically — new files get ingested, modified files re-ingested, deleted files removed. Text is split into overlapping chunks (token-based for prose, AST-aware via tree-sitter for code), embedded with a local model, and stored in LanceDB. Queries embed the question, find the most relevant chunks by vector similarity, and pass them as context to the LLM.

### Data location

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

Override with `LILBEE_DATA=/path` or `--data-dir`.

## License

MIT
