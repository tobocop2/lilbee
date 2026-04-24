# [lilbee](https://tobocop2.github.io/lilbee/)

A terminal-first local search engine for your own files, websites, and scanned documents. One install, no sidecar services, fully offline by default.

<p align="center">
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml"><img src="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://tobocop2.github.io/lilbee/coverage/"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-blue.svg" alt="Typed"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-ELv2-blue.svg" alt="License: Elastic License 2.0"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/dm/lilbee" alt="Downloads"></a>
</p>

> **In active development.** Moving fast toward a 0.6.66 final release. Interfaces may shift between beta versions. Feedback and issues are welcome.

---

- [Why lilbee](#why-lilbee)
- [Previews](#previews)
- [What you can do with it](#what-you-can-do-with-it)
- [TUI](#tui)
- [Hardware requirements](#hardware-requirements)
- [Install](#install)
- [Agent integration](#agent-integration)
- [HTTP Server](#http-server) · [API reference](https://tobocop2.github.io/lilbee/api/)
- [Interactive chat](#interactive-chat)
- [Supported formats](#supported-formats)
- [Experimental](#experimental)

---

## Why lilbee

Local AI tools have gotten great at getting you to a chat window fast. The first evening with a local model is genuinely fun. What makes it stick past the novelty is grounding: the model has to actually know your files, your notes, your codebase. Without that, the conversation runs out of places to go.

The interesting part of local AI isn't the chatbot alone. It's pairing a chatbot with a real search engine over your own documents. Index your stuff, retrieve what matters, let a local model reason over it, get answers with citations you can click back to the source. Now the model knows your world.

Historically that meant juggling a background daemon, a separate inference server, model files fetched by hand from the web, and a retrieval layer glued on top. lilbee bundles all of it into one install. Everything lives in one process, in the terminal, including a browsable GGUF model catalog.

The same executable ships a Textual TUI, a REST API, an MCP server for AI agents, and a Python library. It runs globally by default, or per-project by dropping a `.lilbee/` next to `.git/`, the same pattern git uses. Focused project vaults search better than one giant catch-all index.

An [Encarta 99](https://en.wikipedia.org/wiki/Encarta) you build for yourself, from your own files, shaped to your needs.

## Previews

> Real terminal recordings coming soon. Previews below give the shape of each screen. Written walkthroughs are under [`docs/benchmarks/`](docs/benchmarks/): [Godot level generator](docs/benchmarks/godot-level-generator.md) and [vision OCR model comparison](docs/benchmarks/vision-ocr.md).

**Chat.** The default screen. Streaming replies with clickable citations.

```
 ┌─ lilbee ──────────────────────────────────────────────────────┐
 │ [💬 qwen3:0.6b ▾] [🗄 nomic-embed ▾] [OCR] [All|Wiki|Raw]     │
 │───────────────────────────────────────────────────────────────│
 │                                                               │
 │ You:    what does the oil pressure warning mean?              │
 │                                                               │
 │ lilbee: The oil pressure warning indicates low oil            │
 │         pressure.[¹] When the light stays on, stop the        │
 │         engine immediately.[²]                                │
 │         ─────────────────────                                 │
 │         Sources                                               │
 │         [¹ owners-manual.pdf:42]   ← click to open            │
 │         [² owners-manual.pdf:43]                              │
 │                                                               │
 │───────────────────────────────────────────────────────────────│
 │ Ask anything...                                       [Send]  │
 │ SYNC vault   ████████░░░░░░  42%                              │
 └───────────────────────────────────────────────────────────────┘
```

**Task Center.** Every background job (sync, crawl, wiki build, model pull) in one place. Global concurrency cap; new tasks queue when full.

```
 ┌─ Task Center ─────────────── [cap 3/3] [Clear]┐
 │ ACTIVE (2)                                    │
 │   ████████████░░░░░░░░░  42%  PULL  qwen3:8b  │
 │   ██████░░░░░░░░░░░░░░░  18%  SYNC  vault     │
 │ QUEUED (1)                                    │
 │   CRAWL  https://docs.example.com             │
 │ COMPLETED                                     │
 │   ✓ SYNC  vault                      2 min ago│
 │   ✗ PULL  mistral                    5 min ago│
 │   ✓ ADD   cv-manual.pdf             12 min ago│
 └───────────────────────────────────────────────┘
```

**Wiki.** Auto-generated concept and entity pages, with drafts awaiting review.

```
 ┌─ Wiki ────────────────────────────────────────┐
 │ 🔍 Filter pages...                            │
 │                                               │
 │ Concepts (8)                                  │
 │   Braking Systems               5 src         │
 │   Cooling System                2 src         │
 │ Entities (12)                                 │
 │   Henry Ford                    3 src         │
 │ Drafts (2)                                    │
 │   Tire Pressure                 1 src         │
 │───────────────────────────────────────────────│
 │ ┌─ Braking Systems ────────────────────────┐  │
 │ │ 5 sources · faithfulness 0.92            │  │
 │ │                                          │  │
 │ │ Modern braking systems combine hydraulic │  │
 │ │ actuation with ABS to prevent wheel      │  │
 │ │ lockup under heavy deceleration.[¹]      │  │
 │ │                                          │  │
 │ │ [¹ brake-primer.pdf:8]  ← click          │  │
 │ └──────────────────────────────────────────┘  │
 └───────────────────────────────────────────────┘
```

**Model catalog.** Browse, install, and switch roles without leaving the terminal. `★` marks the featured pick for each role.

```
 ┌─ Model Catalog ───────────────────────────────┐
 │ [All tasks ▾] [All sizes ▾] [Featured ▾]      │
 │ 🔍 search...                  [Grid | List]   │
 │                                               │
 │ Our picks                                     │
 │ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
 │ │ Qwen3 0.6B★│ │ Nomic      │ │ BGE Rerank │  │
 │ │ ▌chat ▐    │ │ ▌embed▐    │ │ ▌rerank▐   │  │
 │ │ [GGUF]     │ │ [GGUF]     │ │ [GGUF]     │  │
 │ │ 450 MB ✓   │ │ 274 MB ✓   │ │ 1.2 GB     │  │
 │ │ [Use]      │ │ [Use]      │ │ [Pull]     │  │
 │ └────────────┘ └────────────┘ └────────────┘  │
 │                                               │
 │ Chat                                          │
 │ ┌────────────┐ ┌────────────┐                 │
 │ │ Qwen3 8B   │ │ Phi-4 14B  │                 │
 │ │ [GGUF]     │ │ [GGUF]     │                 │
 │ │ 4.9 GB     │ │ 9.1 GB     │                 │
 │ │ [Pull]     │ │ [Pull]     │                 │
 │ └────────────┘ └────────────┘                 │
 │               [Load more]                     │
 └───────────────────────────────────────────────┘
```

## What you can do with it

### A personal encyclopedia of what you've collected

Point lilbee at a folder of PDFs, notes, ebooks, or code and it indexes them into a searchable archive with citations that click back to the source line. The same pattern works for anything you have a lot of text about: a medical textbook collection, a guitar theory library, a field's research papers, a car's service manuals, your company's internal wiki. Whatever corpus you give it becomes a searchable, talkable version of exactly what you have.

### Grounding for AI agents

For programmers, lilbee plugs into whatever AI agent you already use (via MCP). Feed it your project's docs, your dependency source, the vendor SDK reference, your design notes, and the agent stops making up function names. It reads the actual code it's about to call, cites the file and line, and tells you when the answer isn't in the corpus instead of guessing. That matters: a lot of AI today produces confident-sounding guesses and charges per token for them. lilbee is built the other way. Answers should come from sources you can check, and the system should be willing to say it doesn't know.

### Offline copies of websites

Web crawling paired with local search and chat takes one command. Install the crawler extra, point lilbee at a docs site, a wiki, or a vendor's API reference, and the pages get fetched, converted to markdown, and indexed. From then on you can search or chat with that site completely offline, even if it changes or goes down.

### How it's built

Under the hood lilbee stands on established open-source projects: [Kreuzberg] handles document parsing, [LanceDB] is the embedded search layer, [llama-cpp][llama-cpp-python] runs models locally, [crawl4ai] and [Playwright] crawl the web, and [Textual] draws the terminal. The architectural bet is that everything stays embedded in one process. Most systems in this shape deploy a vector database and a model server separately, usually reaching for a cloud-hosted search service (Pinecone, managed Qdrant, managed Weaviate) to avoid operating them, which moves your data onto someone else's servers. lilbee skips that layer entirely. Copy the executable onto a laptop and you have a complete local search-and-chat stack with nothing to deploy.

### Documents, code, and scanned images

Document and code processing get treated as a first-class problem. Most retrieval libraries throw your files at a PDF extractor and call it done. lilbee splits the work along the grain of what's being indexed: prose and structured documents (90+ formats across PDFs, Office files, ebooks, HTML, and more) go through [Kreuzberg]'s Rust-based extraction pipeline with heading-aware chunking, so each chunk keeps its section context. Code goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to real functions, classes, and modules instead of arbitrary line ranges. Retrieval returns things that make sense on their own, not fragments that cut through an argument or a function signature.

Scanned PDFs and photographed notes go through an OCR pipeline with a choice of backends: Tesseract, a local GGUF vision model via llama-cpp's mtmd backend (which preserves tables and layout as markdown), or a remote vision model through the SDK backend.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a GGUF, pick a role. Retrieval and generation are deeply tunable. You can make chunks smaller for finer-grained matches, make search stricter to filter out loose results, skip automatic query rewriting for faster responses, turn on a second-pass re-scorer for precision over the top results, or lean more on topic relationships when your corpus has lots of interconnected ideas. All editable from the TUI, environment variables, or a project-local config file, with sensible defaults out of the box.

### Local-first, frontier-capable

lilbee is built as a local-first tool. The TUI shows a persistent warning whenever a cloud-hosted model is active so it's clear when chunks are leaving the machine. Popular frontier models are one `pip install --pre lilbee[litellm]` away when a local model isn't enough, so the power is there when you need it.

## TUI

`lilbee` with no args (or `lilbee chat`) launches a full Textual terminal app. Chat streams replies with clickable citations. A Task Center tracks every background job (sync, crawl, wiki build, model pull) and lets you cancel them with `/cancel`. Other screens cover the model catalog (`/models`), settings (`/settings`), first-time setup wizard (`/setup`), and the auto-built wiki (`/wiki`). Tab completion works for slash commands, file paths, model names, setting keys, and themes.

See [Previews](#previews) for a visual and the [slash-command reference](docs/usage.md#slash-commands) for the full list.

## Hardware requirements

Standalone mode runs entirely on your machine. No cloud required.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16 to 32 GB |
| **GPU / Accelerator** | none required | Apple Metal (M-series), NVIDIA GPU (6+ GB VRAM) |
| **Disk** | 2 GB (models + data) | 10+ GB if using multiple models |
| **CPU** | Any modern x86_64 / ARM64 | same as minimum |

lilbee uses llama-cpp-python for inference locally: Metal on macOS, CUDA on Linux/Windows when available, CPU otherwise (usable for embedding, slow for chat). Popular frontier models are optional; install with `pip install --pre lilbee[litellm]`.

## Install

### Prerequisites

- Python 3.11+
- **Optional** (for scanned PDF / image OCR): [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a GGUF vision model (see [vision OCR](docs/usage.md#vision-models))

No external services needed. lilbee downloads and runs GGUF models locally via llama-cpp.

### Install

```bash
pip install --pre lilbee                              # base install
pip install --pre lilbee[crawler]                     # + web crawling
pip install --pre lilbee[litellm]                     # + Ollama and frontier model support
pip install --pre lilbee[graph]                       # + concept-graph search boost
pip install --pre lilbee[graph,crawler,litellm]       # everything
```

> While 0.6.66 is in beta, the `--pre` flag is required. Once a stable release is cut, plain `pip install lilbee` will work.

### Optional extras

lilbee works out of the box. Extras unlock additional capabilities:

| Extra | Install | What it adds |
|-------|---------|-------------|
| **Web crawling** | `pip install --pre lilbee[crawler]` | Index websites alongside local files. Recursive crawling with Playwright, live progress, cancel, hash-based change detection, SSRF protection, rate limits. |
| **Ollama and frontier models** | `pip install --pre lilbee[litellm]` | Keep compatibility with existing Ollama setups, or use a popular frontier model (OpenAI, Anthropic, Gemini, etc.) for chat, vision, or embeddings while keeping other roles local. You provide the API key. Chunks sent to the provider leave your machine, and the TUI shows a persistent warning while a cloud model is active. |
| **Concept graph** | `pip install --pre lilbee[graph]` | Topic clustering and search boosting. Extracts concepts from your documents and uses their relationships to find results pure text matching misses. Zero extra LLM calls. |

Install multiple: `pip install --pre lilbee[graph,crawler,litellm]`

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration and details.

### Development (run from source)

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
uv run lilbee
```

## Agent integration

lilbee serves as a retrieval backend for AI coding agents via two entry points: an MCP server (`lilbee mcp`) and a JSON CLI (`lilbee --json ...`). MCP exposes search, document lifecycle, crawling, model management, and the full wiki surface as tools; `search` takes a `scope` argument so agents can target documents, wiki pages, or both.

See [docs/agent-integration.md](docs/agent-integration.md) for MCP client configuration, the full tool reference, and JSON CLI examples.

## HTTP Server

`lilbee serve` starts a REST API that any tool or GUI can hit. It covers search (with SSE streaming), document lifecycle, crawling, model management, configuration, and vault-aware source retrieval for GUI clients. Interactive API docs live at `/schema/redoc` when the server is running.

See the [API reference](https://tobocop2.github.io/lilbee/api/) for the full OpenAPI schema and the [usage guide](docs/usage.md) for `serve` options.

An Obsidian plugin that pairs with lilbee is coming soon. It has full feature parity with the TUI but is aimed at GUI users, especially for workflows where seeing the source matters: index a stack of PDFs, ask a question, and preview the exact page the citation points to without leaving the editor. The plugin runs `lilbee serve` as a managed sidecar (starting it, stopping it, and talking to it over the REST API), so there's no separate service for you to babysit. Track progress in [this PR](https://github.com/tobocop2/obsidian-lilbee/pull/7).

## Interactive chat

Running `lilbee` or `lilbee chat` enters the TUI. Type `/` to see the full slash-command list inline, or check the [slash-command reference in the usage guide](docs/usage.md#slash-commands). Slash commands and paths tab-complete; background jobs appear in the Task Center and are cancellable with `/cancel`.

## Supported formats

Text extraction powered by [Kreuzberg], code chunking by [tree-sitter]. Structured formats (XML, JSON, CSV) get embedding-friendly preprocessing. This list is not exhaustive; Kreuzberg supports additional formats beyond what's listed here.

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | none |
| Scanned PDF | `.pdf` (no extractable text) | [Tesseract](https://github.com/tesseract-ocr/tesseract) (auto, plain text), or a GGUF vision model via the native mtmd backend (recommended, preserves tables, headings, and layout as markdown) |
| Office | `.docx`, `.xlsx`, `.pptx` | none |
| eBook | `.epub` | none |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | none |
| Structured | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml` | none |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | none |

See the [usage guide](docs/usage.md#ocr) for OCR setup and [model benchmarks](docs/benchmarks/vision-ocr.md).

## Experimental

Two opt-in features that work but are still finding their final shape. Generation quality and retrieval behavior depend on corpus, models, and knobs; expect to iterate. Feedback is welcome.

### Wiki

lilbee analyzes the documents you've indexed and writes a wiki about them. Pages compound across sources instead of being one-per-document, so concepts and entities that show up repeatedly get their own page with citations from every source that mentions them. Pages live under `$LILBEE_DATA/wiki/`, grouped into `concepts/`, `entities/`, and a `drafts/` queue when confidence is low. An `index.md` tracks them all and `log.md` records every build, ingest, and prune.

Every section is citation-verified against the source chunks and scored for embedding faithfulness before publish. Plain-text concept slugs inside page bodies are rewritten to `[[wiki link]]` form so graph-style markdown viewers can render the connections. Some pages will land in `drafts/` for human review rather than publish direct.

See the [Wiki section of the usage guide](docs/usage.md#wiki) for the full command list and configuration.

### Semantic chunking

A semantic-chunking mode is available as an opt-in alternative to the default fixed-size chunker. It uses embedding similarity to find topic boundaries, so each chunk is one coherent thought instead of a fragment that cuts through an argument. The benefit shows up on prose-heavy corpora like novels, essays, long-form research papers, or interview transcripts. The trade-off is roughly 9x more embedding calls during indexing.

See the [Semantic chunking section of the usage guide](docs/usage.md#semantic-chunking) for trade-offs and how to enable it.

## License

Elastic License 2.0 (ELv2). See [LICENSE](LICENSE).

[Kreuzberg]: https://github.com/Goldziher/kreuzberg
[LanceDB]: https://lancedb.com
[llama-cpp-python]: https://github.com/abetlen/llama-cpp-python
[crawl4ai]: https://github.com/unclecode/crawl4ai
[Playwright]: https://playwright.dev
[Textual]: https://textual.textualize.io
[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
