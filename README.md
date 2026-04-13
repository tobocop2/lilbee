# [lilbee](https://tobocop2.github.io/lilbee/)
> This is in active development. Cool things coming, bear with me please: https://github.com/tobocop2/lilbee/pull/15
> Feel free to use latest published versions but the entire project is actively being rebuilt and will be much more useful soon
> My motivation is a single executable that I can use for q&a, programming, and just having a locally curated encyclopedia like I used to have with Encarta 99, but this time I can talk to it instead and get responses without any need for reaching the Internet. Having a local search engine is awesome and being in full control of the inputs and outputs is even better. 
>
> Gain back some privacy while still having the awesome power of AI. Frontier AI's are awesome and local LLM's are no replacement but they certainly should be used much more than they currently are. Graphics cards not just for gamers and crypto miners, but now they have become very useful to my friends and I on a daily basis thanks to lilbee. 
>
> It's time the masses have something simple to use, fully local, and all in one process / install. Computers can be more than frontends for agents and web browsers and it's time to take advantage of our hardware. This is my attempt at a solution to this problem. I think existing solutions have too many moving pieces and require too many heavy dependencies and often use sidecar style solutions. 
>
> There's no simple way to make local AI immediately useful right in the terminal. That was also a big motivation for me. I needed something terminal first. A single executable that anyone can run is much more ideal and shareable and democratic. It's easier to install and use that way for everyone. 
>
> Local AI at this time is for nerds but it doesn't have to be and this approach I think is the right direction towards that goal. There's a lot to this project so check out the description below for what to expect For the GUI option, I'm releasing an obsidian plugin on top of lilbee with feature parity to the terminal UI here [https://github.com/tobocop2/obsidian-lilbee](https://github.com/tobocop2/obsidian-lilbee)

<p align="center">
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml"><img src="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://tobocop2.github.io/lilbee/coverage/"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-blue.svg" alt="Typed"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/dm/lilbee" alt="Downloads"></a>
</p>

> Interactively or programmatically chat with a database of documents using strictly your own hardware, completely offline. Augment any AI agent via MCP or shell — take a free model or even a frontier model and make it better. Talks to an incredible amount of data formats ([see supported formats](#supported-formats)). Integrate document search into your favorite GUI using the built-in REST API — no need for a separate web app when you already have a preferred GUI ([see Obsidian plugin](https://github.com/tobocop2/obsidian-lilbee)).

---

- [Why lilbee](#why-lilbee)
- [Demos](#demos)
- [Install](#install)
- [Quick start](#quick-start) · [Full usage guide](docs/usage.md)
- [Agent integration](#agent-integration)
- [HTTP Server](#http-server) · [API reference](https://tobocop2.github.io/lilbee/api/)
- [Interactive chat](#interactive-chat)
- [Supported formats](#supported-formats)


---

## Why lilbee

- **Your hardware, your data** — chat with your documents completely offline. No cloud, no telemetry, no API keys required
- **Make any model better** — augment any AI agent via MCP or shell with hybrid RAG search. Take a free model or even a frontier model and make it leagues better at your data
- **Talks to everything** — PDFs, Office docs, spreadsheets, images (OCR), ebooks, and [150+ code languages](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter
- **Bring your own GUI** — built-in REST API means you can integrate document search into whatever tool you already use. No extra app needed ([see Obsidian plugin](https://github.com/tobocop2/obsidian-lilbee))
- **Per-project databases** — `lilbee init` creates a `.lilbee/` directory (like `.git/`) so each project gets its own isolated index

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
- **Optional** (for scanned PDF/image OCR): [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a vision model (see [vision OCR](docs/usage.md#vision-models))

No external services needed. lilbee downloads and runs GGUF models locally via llama-cpp.

### Install

```bash
pip install lilbee        # or: uv tool install lilbee
```

### Optional extras

lilbee works out of the box. These extras unlock additional capabilities:

| Extra | Install | What it adds |
|-------|---------|-------------|
| **Concept graph** | `pip install lilbee[graph]` | Topic clustering + search boosting. Extracts concepts from your documents and uses their relationships to find results that pure text matching misses. Zero extra LLM calls. |
| **Cross-encoder reranking** | `pip install lilbee[reranker]` | Precision pass on search results. Re-scores every (query, chunk) pair with a cross-encoder model. Catches ranking errors the initial search missed. |
| **Web crawling** | `pip install lilbee[crawler]` | Index websites alongside local files. Recursive crawling with Playwright, hash-based change detection, SSRF protection. |
| **Remote providers** | `pip install lilbee[litellm]` | Connect to your favorite frontier model or any litellm-supported provider. Use remote models for chat while keeping embeddings local. |

Install multiple: `pip install lilbee[graph,reranker,crawler]`

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration and details.

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

## HTTP Server

lilbee includes a REST API server so you can integrate document search into any GUI or tool:

```bash
lilbee serve                          # start on a random port (written to <data_dir>/server.port)
lilbee serve --port 8080              # or pick a fixed port
```

Endpoints include `/api/search`, `/api/ask`, `/api/chat` (with streaming SSE variants), `/api/sync`, `/api/add`, and `/api/models`. When the server is running, interactive API docs are available at `/schema/redoc`. See the [API reference](https://tobocop2.github.io/lilbee/api/) for the full OpenAPI schema.

## Interactive chat

Running `lilbee` or `lilbee chat` enters an interactive REPL with conversation history, streaming responses, and slash commands:

| Command | Description |
|---------|-------------|
| `/status` | Show indexed documents and config |
| `/add [path]` | Add a file or directory (tab-completes paths) |
| `/model [name]` | Switch chat model — no args opens a curated picker; with a name, switches directly or prompts to download if not installed (tab-completes installed models) |
| `/vision [name\|off]` | Switch vision OCR model — no args opens a curated picker; with a name, prompts to download if not installed; `off` disables (tab-completes catalog models) |
| `/settings` | Show all current configuration values |
| `/set <key> <value>` | Change a setting (e.g. `/set temperature 0.7`) |
| `/version` | Show lilbee version |
| `/reset` | Delete all documents and data (asks for confirmation) |
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM. Background sync progress appears in the toolbar without interrupting the conversation.

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
