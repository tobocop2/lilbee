# [lilbee](https://tobocop2.github.io/lilbee/)

> A terminal-first, fully local search engine for your own documents. The
> personal Encarta I wished still existed, but this time you can talk to it.
> Index files and websites, auto-build a wiki of the concepts and entities
> inside them, and augment any AI agent over MCP or a JSON CLI, all on your
> own hardware. Plug in popular frontier models when you want them; otherwise
> stay offline. A built-in REST API lets any GUI hit the same index.

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

> Interactively or programmatically chat with a database of documents using
> strictly your own hardware, completely offline. Augment any AI agent via MCP
> or shell. Talks to an incredible amount of data formats
> ([see supported formats](#supported-formats)). Integrate document search into
> any GUI using the built-in REST API, no separate web app needed.

---

- [Why lilbee](#why-lilbee)
- [What you can do with it](#what-you-can-do-with-it)
- [Wiki](#wiki)
- [TUI](#tui)
- [Demos](#demos)
- [Hardware requirements](#hardware-requirements)
- [Install](#install)
- [Quick start](#quick-start) · [Full usage guide](docs/usage.md)
- [Agent integration](#agent-integration)
- [HTTP Server](#http-server) · [API reference](https://tobocop2.github.io/lilbee/api/)
- [Interactive chat](#interactive-chat)
- [Supported formats](#supported-formats)

---

## Why lilbee

**One `pip install lilbee`. Five ways to use it, in one process:**

1. **One executable. Zero servers.** `pip install lilbee`, run `lilbee`,
   you're in. Indexing, search, chat, model management, crawling, and wiki
   generation all happen in the same process that drew the TUI. No external
   services to run alongside it.
2. **A full terminal UI.** `lilbee` launches a real
   [Textual](https://textual.textualize.io) app: streaming chat with
   clickable citations, a Task Center for every background job, a model
   catalog you can install from, a settings panel, a setup wizard, and a
   wiki browser. See [TUI](#tui).
3. **An AI agent augmenter.** Plug lilbee into any MCP-speaking agent to
   give it grounded retrieval over your own documents, or shell out via
   JSON CLI (`lilbee --json search "..."`). Agents stop hallucinating APIs
   they've never read.
4. **An HTTP server backend.** `lilbee serve` exposes a REST API with SSE
   streaming. Any external tool or GUI can hit the same index, including
   vault-aware source retrieval.
5. **A programming library.** `from lilbee import Lilbee`. The same
   `search`, `sync`, `add`, `remove`, `status` surface the CLI uses, as a
   Python class. Build your own pipelines on top without reverse-engineering
   shell output.

**Under the hood:**

- **Advanced search, entirely local.** lilbee runs multiple retrieval
  passes per query to narrow down the right chunks before you ever see
  them. Everything happens on your machine, no round trips to a hosted
  search service. See [docs/architecture.md](docs/architecture.md) for
  the mechanics.
- **Git-like per-project vaults, scaled to fit.** `lilbee init` drops a
  `.lilbee/` right next to your `.git/`. Each vault gets its own index,
  its own models, its own config. One for car manuals, one for a coding
  project, one for research papers. lilbee walks up from `cwd` the way
  git does, so you never have to think about which vault is active.
  Keeping several focused vaults is generally better than one giant
  kitchen-sink vault: a smaller, on-topic corpus returns sharper top-k
  results and stays out of the way of unrelated searches.
- **Self-contained model catalog and manager.** Browse, install, switch,
  and remove models (chat, vision, embedding, reranker) without leaving
  the terminal. Featured picks per role plus the full HuggingFace GGUF
  catalog.
- **Citations on every answer.** Chat replies and wiki sections link back
  to the exact source chunk. Click to jump to the line.
- **Vision OCR built in.** Scanned PDFs, photographed documents, and
  images go through a GGUF vision model (or Tesseract as a fallback) and
  come out as markdown with tables, headings, and layout preserved.
- **Your hardware, your data (by default).** No cloud, no telemetry, no
  API keys required to run lilbee locally. Popular frontier models are
  supported when you want them; see the next bullet for the privacy
  tradeoff.
- **Any model, any provider.** Native GGUF ships by default. Popular
  frontier models are one `pip install lilbee[litellm]` away (you provide
  the API key). **Using a frontier model means your document chunks leave
  your machine** and are sent to that provider on every query. lilbee
  shows a persistent warning in the TUI whenever a cloud-hosted model is
  active so this is never a surprise. Chat, vision, embeddings, and
  reranking are independent roles, so you can keep anything sensitive
  local while using a cloud model for the rest.
- **Auto-built wiki (experimental).** Concepts and entities extracted from
  your documents get their own linked pages that compound across sources.
  See [Wiki](#wiki).
- **Talks to everything.** PDFs (native and scanned), Office docs,
  spreadsheets, images (OCR), ebooks, and
  [150+ code languages](https://github.com/Goldziher/tree-sitter-language-pack)
  with AST-aware chunking. Semantic chunking is an opt-in for prose-heavy
  corpora; see [docs/usage.md](docs/usage.md).

Add files (`lilbee add`), crawl URLs (`lilbee add https://...`), then search or
ask questions. `search` returns chunks without calling the LLM, so agents use
their own model to reason over retrieved context.

## What you can do with it

- **Build a personal encyclopedia grounded in what you've collected.** Index
  a lifetime of PDFs, notes, ebooks, manuals, papers, and reference docs.
  Ask it questions, browse the auto-built wiki, chase citations back to the
  exact page. A private Encarta for whatever matters to you, updated every
  time you drop a file into it.
- **Back an AI agent doing knowledge research.** Point your agent at lilbee
  over MCP. It queries your document corpus instead of guessing from
  training data. Medical literature, legal filings, scientific papers,
  internal reports. The agent gets grounded answers and citations; you
  keep private documents on your own machine.
- **Back a coding agent across many languages and many document sets.**
  tree-sitter gives AST-aware chunking for
  [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack),
  so lilbee handles a Python monorepo, a Rust SDK, a TypeScript frontend,
  and a C firmware tree in one index. Add API references, vendor docs,
  and design docs alongside the code. Your coding agent stops hallucinating
  APIs it's never actually read.
- **Crawl websites and talk to them offline.** Install the crawler extra
  (`pip install lilbee[crawler]`) and point lilbee at a docs site, a wiki,
  a vendor's API reference. It fetches pages with a headless browser,
  follows links recursively with rate-limited retries, and indexes
  everything into the same store as your local files. From then on you can
  search or chat with that site offline, even if it goes down or changes.
  Hash-based change detection means re-crawls only touch what moved.
- **Digitize paper archives and scanned documents.** Point lilbee at a
  folder of scanned PDFs, photographed notes, or image-only documents.
  The vision OCR pipeline (native GGUF via mtmd, or Tesseract as a
  fallback) turns them into searchable markdown with tables, headings,
  and multi-column layout preserved. Family archives, old legal paperwork,
  photographed textbooks, your entire filing cabinet all become a corpus
  you can actually query.

## Wiki

> **Experimental.** The wiki layer works and has coverage, but generation
> quality depends heavily on your corpus and the chosen chat model. Expect
> some concepts to land in `drafts/` for human review rather than direct
> publish. Feedback on what's useful and what isn't is very welcome.

lilbee analyzes the documents you've indexed and writes a wiki about them.
Pages compound across sources instead of being one-per-document, so concepts
and entities that show up repeatedly get their own page with citations from
every source that mentions them:

- `concepts/`. One page per LLM-identified concept (e.g. `braking-systems.md`).
- `entities/`. One page per proper-noun entity extracted by NER (e.g.
  `henry-ford.md`).
- `index.md`. Auto-generated table of contents.
- `log.md`. Append-only audit trail of every build, ingest, and prune.

Every section is citation-verified against the source chunks, scored for
embedding faithfulness, and low-confidence output routes to `drafts/` for you to
accept or reject. Plain-text concept slugs in page bodies are rewritten to
`[[wiki link]]` form so graph-style markdown viewers can render the connections.
The wiki lives under `$LILBEE_DATA/wiki/` by default.

See the [Wiki section of the usage guide](docs/usage.md#wiki) for the full
command list and configuration.

## TUI

`lilbee` with no args (or `lilbee chat`) launches a full Textual terminal UI:

- **Chat.** Streaming responses, conversation history, and a sidebar of sources
  for every answer.
- **Task Center.** Every background job (sync, crawl, wiki build, model pull)
  shows live progress and is cancellable with `/cancel`.
- **Model catalog.** `/models` browses curated and HuggingFace models; install,
  remove, and switch roles (chat / vision / embedding / reranker) without
  leaving the terminal.
- **Settings.** `/settings` edits every `LILBEE_*` knob in-place, with
  per-setting reset and global reset-all.
- **Setup wizard.** `/setup` walks a first-time user through picking models
  and initializing the local index.
- **Wiki screen.** `/wiki` opens the auto-generated wiki for browsing, search,
  and draft review.
- **Autocomplete.** Tab completion for slash commands, paths, model names,
  setting keys, and themes.

The TUI is the default chat experience. Slash commands listed under
[Interactive chat](#interactive-chat) work the same from any screen.

## Demos

> Real terminal recordings coming soon. Previews below give the shape of each
> screen. Written walkthroughs are under [`docs/benchmarks/`](docs/benchmarks/):
> [Godot level generator](docs/benchmarks/godot-level-generator.md) and
> [vision OCR model comparison](docs/benchmarks/vision-ocr.md).

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

**Task Center.** Every background job (sync, crawl, wiki build, model pull) in
one place. Global concurrency cap; new tasks queue when full.

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

**Model catalog.** Browse, install, and switch roles without leaving the
terminal. `★` marks the featured pick for each role.

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

## Hardware requirements

Standalone mode runs entirely on your machine. No cloud required.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16 to 32 GB |
| **GPU / Accelerator** | none required | Apple Metal (M-series), NVIDIA GPU (6+ GB VRAM) |
| **Disk** | 2 GB (models + data) | 10+ GB if using multiple models |
| **CPU** | Any modern x86_64 / ARM64 | same as minimum |

lilbee uses llama-cpp-python for inference locally: Metal on macOS, CUDA on
Linux/Windows when available, CPU otherwise (usable for embedding, slow for
chat). Popular frontier models are optional; install with
`pip install lilbee[litellm]`.

## Install

### Prerequisites

- Python 3.11+
- **Optional** (for scanned PDF / image OCR): [Tesseract](https://github.com/tesseract-ocr/tesseract)
  (`brew install tesseract` / `apt install tesseract-ocr`) or a GGUF vision
  model (see [vision OCR](docs/usage.md#vision-models))

No external services needed. lilbee downloads and runs GGUF models locally via
llama-cpp.

### Install

```bash
pip install lilbee        # or: uv tool install lilbee
```

### Optional extras

lilbee works out of the box. Extras unlock additional capabilities:

| Extra | Install | What it adds |
|-------|---------|-------------|
| **Concept graph** | `pip install lilbee[graph]` | Topic clustering and search boosting. Extracts concepts from your documents and uses their relationships to find results pure text matching misses. Zero extra LLM calls. |
| **Web crawling** | `pip install lilbee[crawler]` | Index websites alongside local files. Recursive crawling with Playwright, live progress, cancel, hash-based change detection, SSRF protection, rate limits. |
| **Popular frontier models** | `pip install lilbee[litellm]` | Use a popular frontier model for chat, vision, or embeddings while keeping other roles local. You provide the API key. Chunks sent to the provider leave your machine, and the TUI shows a persistent warning while a cloud model is active. |

Install multiple: `pip install lilbee[graph,crawler,litellm]`

See the [full guide on optional extras](docs/usage.md#optional-extras) for
configuration and details.

### Development (run from source)

```bash
git clone https://github.com/tobocop2/lilbee && cd lilbee
uv sync
uv run lilbee
```

## Quick start

See the [usage guide](docs/usage.md).

## Agent integration

lilbee serves as a retrieval backend for AI coding agents via two entry points:
an MCP server (`lilbee mcp`) and a JSON CLI (`lilbee --json ...`). MCP exposes
search, document lifecycle, crawling, model management, and the full wiki
surface as tools; `search` takes a `scope` argument so agents can target
documents, wiki pages, or both.

See [docs/agent-integration.md](docs/agent-integration.md) for MCP client
configuration, the full tool reference, and JSON CLI examples.

## HTTP Server

`lilbee serve` starts a REST API that any tool or GUI can hit. It covers
search (with SSE streaming), document lifecycle, crawling, model management,
configuration, and vault-aware source retrieval for GUI clients. Interactive
API docs live at `/schema/redoc` when the server is running.

See the [API reference](https://tobocop2.github.io/lilbee/api/) for the full
OpenAPI schema and the [usage guide](docs/usage.md) for `serve` options.

## Interactive chat

Running `lilbee` or `lilbee chat` enters the TUI. Type `/` to see the full
slash-command list inline, or check the
[slash-command reference in the usage guide](docs/usage.md#slash-commands).
Slash commands and paths tab-complete; background jobs appear in the Task
Center and are cancellable with `/cancel`.

## Supported formats

Text extraction powered by [Kreuzberg], code chunking by [tree-sitter].
Structured formats (XML, JSON, CSV) get embedding-friendly preprocessing. This
list is not exhaustive; Kreuzberg supports additional formats beyond what's
listed here.

| Format | Extensions | Requires |
|--------|-----------|----------|
| PDF | `.pdf` | none |
| Scanned PDF | `.pdf` (no extractable text) | [Tesseract](https://github.com/tesseract-ocr/tesseract) (auto, plain text), or a GGUF vision model via the native mtmd backend (recommended, preserves tables, headings, and layout as markdown) |
| Office | `.docx`, `.xlsx`, `.pptx` | none |
| eBook | `.epub` | none |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Data | `.csv`, `.tsv` | none |
| Structured | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml` | none |
| Text | `.md`, `.txt`, `.html`, `.rst` | none |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | none |

See the [usage guide](docs/usage.md#ocr) for OCR setup and
[model benchmarks](docs/benchmarks/vision-ocr.md).

## License

MIT

[Kreuzberg]: https://github.com/Goldziher/kreuzberg
[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
