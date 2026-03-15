# lilbee

> Beta — feedback and bug reports welcome. [Open an issue](https://github.com/tobocop2/lilbee/issues).

[![PyPI](https://img.shields.io/pypi/v/lilbee)](https://pypi.org/project/lilbee/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg)](https://github.com/tobocop2/lilbee/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://tobocop2.github.io/lilbee/)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/lilbee)](https://pypi.org/project/lilbee/)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Chat with your documents locally using Ollama — or plug into AI agents as a retrieval backend via MCP. Indexes PDFs (including scanned via vision OCR), Office docs, spreadsheets, images, and code with a git-like per-project model. Powered by [Kreuzberg] for text extraction, [Ollama] for embeddings and chat, and [LanceDB] for vector storage.

---

- [Why lilbee](#why-lilbee)
- [Demos](#demos)
- [Install](#install)
- [Quick start](#quick-start) · [Full usage guide](docs/usage.md)
- [Agent integration](#agent-integration)
- [Interactive chat](#interactive-chat)
- [Supported formats](#supported-formats)


---

## Why lilbee

lilbee indexes documents and code into a searchable local knowledge base. Use it standalone — search, ask questions, chat — or plug it into AI coding agents as a retrieval backend via MCP.

Most tools like this only handle code. lilbee handles PDFs, Word docs, spreadsheets, images (OCR) — and code too, with AST-aware chunking.

- **Standalone knowledge base** — add documents, search, ask questions, or chat interactively with model switching and slash commands
- **AI agent backend** — MCP server and JSON CLI so coding agents can search your indexed docs as context
- **Per-project databases** — `lilbee init` creates a `.lilbee/` directory (like `.git/`) so each project gets its own isolated index
- **Documents and code alike** — PDFs, Office docs, spreadsheets, images, ebooks, and [150+ code languages](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter
- **Open-source** — runs with [Ollama] and LanceDB, no cloud APIs or Docker required

Add files (`lilbee add`), then search or ask questions. Once indexed, `search` works without Ollama — agents use their own LLM to reason over the retrieved chunks.

## Demos

> Click the &#9654; arrows below to expand each demo.

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

<details>
<summary><b>One-shot question from OCR'd content</b></summary>

The scanned Star Wars: X-Wing Collector's Edition guide, queried with a single `lilbee ask` command — no interactive chat needed.

![Top speed question](demos/top-speed.gif)

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

## Hardware requirements

When used standalone, lilbee runs entirely on your machine — chat with your documents privately, no cloud required.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16–32 GB |
| **GPU / Accelerator** | — | Apple Metal (M-series), NVIDIA GPU (6+ GB VRAM) |
| **Disk** | 2 GB (models + data) | 10+ GB if using multiple models |
| **CPU** | Any modern x86_64 / ARM64 | — |

Ollama handles inference and uses Metal on macOS or CUDA on Linux/Windows. Without a GPU, models fall back to CPU — usable for embedding but slow for chat.

## Install

### Prerequisites

- Python 3.11+
- [Ollama] — the embedding model (`nomic-embed-text`) is auto-pulled on first sync. If no chat model is installed, lilbee prompts you to pick and download one.
- **Optional** (for scanned PDF/image OCR): [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or an Ollama vision model (recommended for better quality — see [vision OCR](docs/usage.md#vision-models))

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

See the [usage guide](docs/usage.md).


## Agent integration

lilbee can serve as a local retrieval backend for AI coding agents via MCP or JSON CLI. See [docs/agent-integration.md](docs/agent-integration.md) for setup and usage.

## Interactive chat

Running `lilbee` or `lilbee chat` enters an interactive REPL with conversation history, streaming responses, and slash commands:

| Command | Description |
|---------|-------------|
| `/status` | Show indexed documents and config |
| `/add [path]` | Add a file or directory (tab-completes paths) |
| `/model [name]` | Switch chat model — no args opens an interactive picker; with a name, switches directly (tab-completes installed models) |
| `/vision [name\|off]` | Switch vision OCR model — no args opens a picker, `off` disables (tab-completes catalog models) |
| `/version` | Show lilbee version |
| `/reset` | Delete all documents and data (asks for confirmation) |
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM.

## Supported formats

Text extraction powered by [Kreuzberg], code chunking by [tree-sitter]. Structured formats (XML, JSON, CSV) get embedding-friendly preprocessing. This list is not exhaustive — Kreuzberg supports additional formats beyond what's listed here.

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | — |
| Scanned PDF | `.pdf` (no extractable text) | [Tesseract](https://github.com/tesseract-ocr/tesseract) (auto, plain text) or [Ollama] vision model (recommended — preserves tables, headings, and layout as markdown) |
| Office | `.docx`, `.xlsx`, `.pptx` | — |
| eBook | `.epub` | — |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | — |
| Structured | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml` | — |
| Text | `.md`, `.txt`, `.html`, `.rst` | — |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | — |

See the [usage guide](docs/usage.md#ocr) for OCR setup and [model benchmarks](docs/benchmarks/vision-ocr.md).

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
