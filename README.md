# lilbee

> This is an experimental tool and a work in progress. There will be issues with some formats and performance at scale is unknown at this time

[![PyPI](https://img.shields.io/pypi/v/lilbee)](https://pypi.org/project/lilbee/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://tobocop2.github.io/lilbee/)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/lilbee)](https://pypi.org/project/lilbee/)

> Local knowledge base for documents and code. Search, ask questions, or chat — standalone or as a retrieval backend for AI agents via MCP. Fully offline, powered by Ollama.

---

- [Why lilbee](#why-lilbee)
- [Demos](#demos)
- [Install](#install)
- [Quick start](#quick-start)
- [Agent integration](#agent-integration)
- [Interactive chat](#interactive-chat)
- [Supported formats](#supported-formats)
- [Vision OCR (optional)](#vision-ocr-optional)
- [Configuration](#configuration)
- [How it works](#how-it-works)

---

## Why lilbee

lilbee indexes documents and code into a searchable local knowledge base. Use it standalone — search, ask questions, chat — or plug it into AI coding agents as a retrieval backend via MCP.

Most tools like this only handle code. lilbee handles PDFs, Word docs, spreadsheets, images (OCR) — and code too, with AST-aware chunking.

- **Standalone knowledge base** — add documents, search, ask questions, or chat interactively with model switching and slash commands
- **AI agent backend** — MCP server and JSON CLI so coding agents (Claude Code, OpenCode, etc.) can search your indexed docs as context
- **Per-project databases** — `lilbee init` creates a `.lilbee/` directory (like `.git/`) so each project gets its own isolated index
- **Documents and code alike** — PDFs, Office docs, spreadsheets, images, ebooks, and [150+ code languages](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter
- **Fully offline** — runs on your machine with [Ollama] and LanceDB, no cloud APIs or Docker

Add files (`lilbee add`), then search or ask questions. Once indexed, `search` works without Ollama — agents use their own LLM to reason over the retrieved chunks.

## Demos

<details>
<summary><b>AI agent</b> — lilbee search vs web search (<a href="docs/benchmarks/godot-level-generator.md">detailed analysis</a>)</summary>

[opencode] + [minimax-m2.5-free][opencode], single prompt, no follow-ups. The [Godot 4.4 XML class reference][godot-docs] (917 files) is indexed in lilbee. The baseline uses [Exa AI][exa] code search instead.

**⚠️ Caution:** minimax-m2.5-free is a cloud model — retrieved chunks are sent to an external API. Use a local model if your documents are private.

| | API hallucinations | Lines |
|---|---|---|
| **With lilbee** ([code](demos/godot-with-lilbee/level_generator.gd) · [config](demos/godot-with-lilbee/)) | 0 | 261 |
| **Without lilbee** ([code](demos/godot-without-lilbee/level_generator.gd) · [config](demos/godot-without-lilbee/)) | 4 (~22% error rate) | 213 |

<details>
<summary><b>With lilbee</b> — all Godot API calls match the class reference</summary>

![With lilbee MCP](demos/godot-with-lilbee.gif)
</details>

<details>
<summary><b>Without lilbee</b> — 4 hallucinated APIs (<a href="docs/benchmarks/godot-level-generator.md#without-lilbee-213-lines--4-bugs">details</a>)</summary>

![Without lilbee](demos/godot-without-lilbee.gif)
</details>

If you spot issues with these benchmarks, please [open an issue](https://github.com/tobocop2/lilbee/issues).

</details>

### Vision OCR

<details>
<summary><b>Scanned PDF → searchable knowledge base</b></summary>

A scanned 1998 Star Wars: X-Wing Collector's Edition manual indexed with vision OCR ([LightOnOCR-2][lightonocr]), then queried in lilbee's interactive chat (`qwen3-coder:30b`, fully local). Three questions about dev team credits, energy management, and starfighter speeds — all answered from the OCR'd content.

![Vision OCR demo](demos/vision-ocr.gif)

See [benchmarks, test documents, and sample output](docs/benchmarks/vision-ocr.md) for model comparisons.
</details>

### Standalone

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
- [Ollama] — the embedding model (`nomic-embed-text`) is auto-pulled on first sync. If no chat model is installed, lilbee prompts you to pick and download one.
- **Optional** (for image OCR): `brew install tesseract` / `apt install tesseract-ocr`

> **First-time download:** If you're new to Ollama, expect the first run to take a while — models are large files that need to be downloaded once. For example, `qwen3:8b` is ~5 GB and the embedding model `nomic-embed-text` is ~274 MB. After the initial download, models are cached locally and load in seconds. You can check what you have installed with `ollama list`.

### Install

```bash
pip install lilbee        # or: uv tool install lilbee
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

# Initialize a per-project knowledge base (like git init)
lilbee init

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

# Use a different chat model
lilbee ask "Explain this" --model qwen3

# Check what's indexed
lilbee status
```


## Agent integration

lilbee can serve as a local retrieval backend for AI coding agents via MCP or JSON CLI. See [docs/agent-integration.md](docs/agent-integration.md) for setup and usage.

## Interactive chat

Running `lilbee` or `lilbee chat` enters an interactive REPL with conversation history, streaming responses, and slash commands:

| Command | Description |
|---------|-------------|
| `/status` | Show indexed documents and config |
| `/add [path]` | Add a file or directory (tab-completes paths) |
| `/model [name]` | Switch chat model — no args opens an interactive picker; with a name, switches directly (tab-completes installed models) |
| `/version` | Show lilbee version |
| `/reset` | Delete all documents and data (asks for confirmation) |
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM.

## Supported formats

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | — |
| Office | `.docx`, `.xlsx`, `.pptx` | — |
| eBook | `.epub` | — |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | — |
| Structured | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml` | — |
| Text | `.md`, `.txt`, `.html`, `.rst` | — |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | — |

### Vision OCR (optional)

Scanned PDFs that produce no extractable text can be processed using a local vision model via Ollama. During `sync`, lilbee detects empty PDFs and:
- **Without a vision model configured:** skips the file and warns you to set one up
- **With a vision model configured:** rasterizes each page and sends it to the vision model for OCR

**Setup:**
```bash
# In chat, use the interactive picker:
/vision

# Or set directly:
/vision maternion/LightOnOCR-2

# Or via environment variable:
export LILBEE_VISION_MODEL=maternion/LightOnOCR-2
```

**Recommended models:**

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| maternion/LightOnOCR-2 | 1.5 GB | 11.9s/page | Best — clean markdown output |
| deepseek-ocr | 6.7 GB | 17.4s/page | Excellent accuracy, plain text |
| glm-ocr | 2.2 GB | 51.7s/page | Good accuracy |
| minicpm-v | 5.5 GB | 35.6s/page | Decent, slower |

> Benchmarks: Apple M1 Pro, 32 GB RAM, Ollama 0.17.7. See [benchmarks, test documents, and sample output](docs/benchmarks/vision-ocr.md).

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path |
| `LILBEE_CHAT_MODEL` | `qwen3:8b` | Ollama chat model |
| `LILBEE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding dimensions |
| `LILBEE_CHUNK_SIZE` | `512` | Tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk for embedding |
| `LILBEE_TOP_K` | `10` | Number of retrieval results |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | Custom system prompt for RAG answers |

CLI also accepts `--model` / `-m`, `--data-dir` / `-d`, and `--version` / `-V` flags.

## How it works

Documents are hashed and synced automatically — add, change, or delete files and lilbee keeps the index current. [Kreuzberg] extracts text from PDFs, Office docs, images (OCR), etc. [tree-sitter] chunks code by AST. Chunks are embedded via [Ollama] and stored in [LanceDB]. Queries embed the question, find the closest chunks by vector similarity, and pass them as context to the LLM.

### Data location

lilbee uses per-project databases when available, falling back to a global database:

1. **`--data-dir` / `LILBEE_DATA`** — explicit override (highest priority)
2. **`.lilbee/`** — found by walking up from the current directory (like `.git/`)
3. **Global** — platform-default location (see below)

Run `lilbee init` to create a `.lilbee/` directory in your project. It contains `documents/`, `data/`, and a `.gitignore` that excludes derived data. When active, all commands operate on the local database only.

| Platform | Global path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

## License

MIT

[Ollama]: https://ollama.com
[opencode]: https://opencode.ai
[Kreuzberg]: https://github.com/Goldziher/kreuzberg
[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
[LanceDB]: https://lancedb.com
[godot-docs]: https://github.com/godotengine/godot/tree/4.4-stable/doc/classes
[tml]: https://github.com/godotengine/godot/blob/4.4-stable/doc/classes/TileMapLayer.xml
[asg2d]: https://github.com/godotengine/godot/blob/4.4-stable/doc/classes/AStarGrid2D.xml
[nr2d]: https://github.com/godotengine/godot/blob/4.4-stable/doc/classes/NavigationRegion2D.xml
[ns2d]: https://github.com/godotengine/godot/blob/4.4-stable/doc/classes/NavigationServer2D.xml
[exa]: https://exa.ai
[lightonocr]: https://ollama.com/maternion/LightOnOCR-2
