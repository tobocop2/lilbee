# [lilbee](https://tobocop2.github.io/lilbee/)

A batteries-included local search engine for your data and code that you can talk to.

[Project site](https://tobocop2.github.io/lilbee/) · [PyPI](https://pypi.org/project/lilbee/) · [Obsidian plugin](https://tobocop2.github.io/obsidian-lilbee/)

Point it at your files, notes, and code and ask questions in plain English; every answer links back to the file and line it came from. Point it at nothing and it's just a fast chatbot.

It's all one program: a full-screen terminal app, a command-line tool, a Model Context Protocol server, an HTTP API, and a Python library. Run it when you want, close it when you're done; nothing left running in the background, no container to keep alive. It runs on your computer; lilbee uses a cloud model only when you pick one.

<p align="center">
  <a href="https://github.com/tobocop2/lilbee/releases"><img src="https://img.shields.io/github/v/release/tobocop2/lilbee?include_prereleases&label=latest%20release" alt="Latest release (incl. pre-releases)"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee?include_prereleases&label=PyPI" alt="lilbee on PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml"><img src="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://tobocop2.github.io/lilbee/coverage/"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-blue.svg" alt="Typed"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-ELv2-blue.svg" alt="License: Elastic License 2.0"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/dm/lilbee" alt="Downloads"></a>
</p>

> ## ⚠️ Beta software
>
> lilbee is in **active beta** development. Every release on PyPI is a pre-release; you must use `--pre` (or uv's `--prerelease=allow`) when installing. Interfaces, command names, and on-disk formats may shift between betas. Feedback, bug reports, and issues are very welcome; that's the whole point of the beta.
>
> Latest pre-release (always): [lilbee on PyPI →](https://pypi.org/project/lilbee/)

---

- [Quick start](#quick-start)
- [Highlights](#highlights)
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

## Quick start

```bash
pip install --pre lilbee        # or: uv tool install --prerelease=allow lilbee
lilbee self-check               # verifies the build runs on your hardware
lilbee                          # launch the terminal app, then pick a chat + embedding model
```

Homebrew, AUR, Docker, Nix, prebuilt binaries (no Python), and CUDA wheels: see [Install](#install). Optional extras (`[crawler]`, `[litellm]`, `[graph]`): see [Optional extras](#optional-extras).

## Highlights

- **One program, one install.** A model catalog, a search over your own files and code, and a chat. The same executable is also a CLI, a Textual TUI, an MCP server, a REST API, and a Python library. No background daemon, no separate inference server, no vector database to stand up.
- **Answers cite the source line.** Ask a question; get a reply with clickable citations pointing back to the exact line they came from.
- **Bring your own files.** PDFs, Office files, ebooks, code in 150+ languages, scanned pages and photos (OCR), and crawled docs sites turned into searchable markdown.
- **A built-in model catalog.** Browse and pull models straight from Hugging Face Hub, from inside the app. lilbee is the model runtime; no hunting for files yourself.
- **Runs on your computer.** Models, index, and files all stay local. lilbee uses a cloud model only when you pick one, and flags it when it does.
- **Per-project libraries.** Run globally, or drop a `.lilbee/` next to `.git/` the way git does; each domain stays its own clean library.

## Why lilbee

The first evening with a local model is fun. What makes it more than a novelty is grounding: the model needs context from your notes, your files, your code, or it runs out of places to go. lilbee pairs the chat with a real search engine over a set of documents you choose, so a local model can reason over your world and answer with citations you can click back to the source.

Standing this up used to mean a background daemon, a separate inference server, model files fetched by hand, and a retrieval layer glued on top. lilbee bundles all of it into one install, in one process, in the terminal, including a built-in model catalog. The same executable is also a CLI, a REST API, an MCP server for AI agents, and a Python library. It runs globally by default, or per-project by dropping a `.lilbee/` next to `.git/`, the same pattern git uses; a focused library produces better answers than one catch-all pile of everything.

An [Encarta 99](https://en.wikipedia.org/wiki/Encarta) you build for yourself, from your own files, shaped to your needs.

## Previews

> ASCII sketches of each screen below: the shape, not the pixels. Written walkthroughs are under [`docs/benchmarks/`](docs/benchmarks/): [Godot level generator](docs/benchmarks/godot-level-generator.md) and [vision OCR model comparison](docs/benchmarks/vision-ocr.md).

**Chat.** The default screen. Streaming replies with clickable citations.

```
 ┌─ lilbee ──────────────────────────────────────────────────────┐
 │ [Chat]  Catalog  Status  Settings  Tasks         [INSERT]     │
 │                                                               │
 │ You:    what does the oil pressure warning mean?              │
 │                                                               │
 │ lilbee: The oil pressure warning indicates low oil            │
 │         pressure.[1] When the light stays on, stop the        │
 │         engine immediately.[2]                                │
 │         ─────────────────────                                 │
 │         Sources                                               │
 │         [1 owners-manual.pdf:42]   <- click to open           │
 │         [2 owners-manual.pdf:43]                              │
 │                                                               │
 │ ┌───────────────────────────────────────────────────────────┐ │
 │ │ Ask anything...                                           │ │
 │ │                                                           │ │
 │ │ Chat [Qwen3 0.6B]  Embed [Nomic v1.5]   [Search|Chat]     │ │
 │ └───────────────────────────────────────────────────────────┘ │
 │ SYNC vault   [============------------]  42%                  │
 └───────────────────────────────────────────────────────────────┘
```

**Task Center.** Every background job (sync, crawl, wiki build, model pull) in one place. Global concurrency cap; new tasks queue when full.

```
 ┌─ Task Center ─────────────── [cap 3/3] [Clear]┐
 │ ACTIVE (2)                                    │
 │   [============-----]  42%  PULL  Qwen3 8B    │
 │   [======-----------]  18%  SYNC  vault       │
 │ QUEUED (1)                                    │
 │   CRAWL  https://docs.example.com             │
 │ COMPLETED                                     │
 │   [v] SYNC  vault                  2 min ago  │
 │   [x] PULL  mistral                5 min ago  │
 │   [v] ADD   cv-manual.pdf         12 min ago  │
 └───────────────────────────────────────────────┘
```

**Wiki.** Auto-generated concept and entity pages, with drafts awaiting review.

```
 ┌─ Wiki ────────────────────────────────────────┐
 │ Filter pages...                               │
 │                                               │
 │ Concepts (8)                                  │
 │   Braking Systems               5 src         │
 │   Cooling System                2 src         │
 │ Entities (12)                                 │
 │   Henry Ford                    3 src         │
 │ Drafts (2)                                    │
 │   Tire Pressure                 1 src         │
 ├───────────────────────────────────────────────┤
 │ ┌─ Braking Systems ────────────────────────┐  │
 │ │ 5 sources | faithfulness 0.92            │  │
 │ │                                          │  │
 │ │ Modern braking systems combine hydraulic │  │
 │ │ actuation with ABS to prevent wheel      │  │
 │ │ lockup under heavy deceleration.[1]      │  │
 │ │                                          │  │
 │ │ [1 brake-primer.pdf:8]  <- click         │  │
 │ └──────────────────────────────────────────┘  │
 └───────────────────────────────────────────────┘
```

**Model catalog.** Browse models from Hugging Face Hub, pull one with a click, and switch roles without leaving the terminal. `*` marks the developer's pick for each role.

```
 ┌─ Model Catalog ───────────────────────────────────┐
 │ [ Local | Frontier ]                              │
 │ [All tasks v] [All sizes v]                       │
 │ search...                       [Grid | List]     │
 │                                                   │
 │ Our picks                                         │
 │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
 │ │ Qwen3 0.6B *│ │ Nomic v1.5  │ │ BGE Rerank  │   │
 │ │ [chat]      │ │ [embed]     │ │ [rerank]    │   │
 │ │ 450 MB  ok  │ │ 274 MB  ok  │ │ 1.2 GB      │   │
 │ │ [Use]       │ │ [Use]       │ │ [Pull]      │   │
 │ └─────────────┘ └─────────────┘ └─────────────┘   │
 │                                                   │
 │ Chat                                              │
 │ ┌─────────────┐ ┌─────────────┐                   │
 │ │ Qwen3 8B    │ │ Phi-4 14B   │                   │
 │ │ 4.9 GB      │ │ 9.1 GB      │                   │
 │ │ [Pull]      │ │ [Pull]      │                   │
 │ └─────────────┘ └─────────────┘                   │
 │                  [Load more]                      │
 └───────────────────────────────────────────────────┘
```

## What you can do with it

### A library of your own files

Point lilbee at a folder of PDFs, notes, ebooks, or code and it builds a searchable library, with citations that click back to the source line. The pattern works for anything you have a lot of text about: a medical-textbook collection, a field's research papers, a car's service manuals, your company's internal wiki. Whatever you give it becomes searchable, and you can talk to it.

### Grounding for AI agents

lilbee plugs into whatever AI agent you already use, over MCP. Feed it your project's docs, your dependency source, the vendor SDK reference, your design notes, and the agent stops making up function names: it reads the actual code it's about to call, cites the file and line, and says it doesn't know when the answer isn't in the corpus, instead of guessing.

### Offline copies of websites

Install the `[crawler]` extra, point lilbee at a docs site, a wiki, or a vendor's API reference, and the pages get fetched, converted to markdown, and added to your library. From then on you can search or chat with that copy of the site offline, even after it changes or goes down.

### How it's built

lilbee stands on established open-source projects, with everything embedded in one process:

- [Kreuzberg] parses documents
- [LanceDB] is the embedded search layer
- [llama-cpp][llama-cpp-python] runs models locally
- [crawl4ai] and [Playwright] crawl the web
- [Textual] draws the terminal

Most systems in this shape run a vector database and a model server separately, often reaching for a cloud-hosted search service, which moves your data onto someone else's machines. lilbee does the whole thing locally: pull it to your laptop, pull your models, curate your documents. A complete local search-and-chat stack with nothing to deploy.

### Documents, code, and scanned images

Most retrieval tools throw your files at a PDF extractor and call it done. lilbee splits the work by what's being indexed:

- **Prose and structured documents** (90+ formats: PDFs, Office files, ebooks, HTML, and more) go through [Kreuzberg]'s extraction pipeline with heading-aware chunking, so each chunk keeps its section context.
- **Code** goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to real functions, classes, and modules instead of arbitrary line ranges.
- **Scanned PDFs and photos** go through OCR: Tesseract for plain text, a local GGUF vision model that keeps tables and layout as markdown, or a remote vision model.

Retrieval returns things that make sense on their own, not fragments cut through an argument or a function signature.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a model, pick a role. Retrieval and generation are deeply tunable, 50+ settings in all: smaller chunks for finer matches, stricter search to filter loose results, a second-pass re-scorer for precision over the top results, more weight on topic relationships when your corpus is densely interconnected, and so on. Edit them from the TUI, environment variables, or a project-local config file. Sane defaults out of the box.

### Cloud models, when you want them

lilbee runs entirely on your machine by default. To point a role at a cloud-hosted model, install the `[litellm]` extra and add an API key; the TUI shows a persistent warning whenever a cloud model is active, so it's clear when chunks are leaving the machine.

## TUI

`lilbee` with no args (or `lilbee chat`) launches a full Textual terminal app. Chat streams replies with clickable citations. Above the prompt: searchable pickers for the active chat and embedding models, and a Search / Chat toggle (F3): Search runs document retrieval on every prompt, Chat answers directly, and Search falls back to a chat answer when nothing relevant is indexed. A Task Center tracks every background job (sync, crawl, wiki build, model pull) and cancels them with `/cancel`. Other screens cover the model catalog (`/models`), settings (`/settings`), the setup wizard (`/setup`), and the auto-built wiki (`/wiki`). Tab completion works everywhere: slash commands, file paths, model names, setting keys, themes.

See [Previews](#previews) for the shapes and the [slash-command reference](docs/usage.md#slash-commands) for the full list.

## Hardware requirements

Standalone mode runs entirely on your machine. No cloud required.

### Supported platforms

| Platform           | Minimum                                                                                                                                                                                                                                                                                                                | Recommended                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Linux x86_64**   | A 64-bit Intel or AMD CPU from **2013 or newer**: Intel Core i3/i5/i7 4th-gen (Haswell), Intel Xeon E3-12xx v3 / E5-26xx v3, AMD FX-95xx (Steamroller) or any AMD Zen-based chip. Anything corresponding to the [`x86-64-v3` microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels). | A modern Intel Core / Xeon / AMD Ryzen / EPYC + an NVIDIA, AMD, or Intel Arc GPU |
| **macOS arm64**    | Any Apple Silicon Mac (M1 or newer) running macOS 11+                                                                                                                                                                                                                                                                  | M-series Pro / Max / Ultra                                                       |
| **Windows x86_64** | A 64-bit Intel or AMD CPU from **2013 or newer** (same generations as Linux above), Windows 10/11                                                                                                                                                                                                                      | Modern desktop / workstation CPU + GPU                                           |
| **Linux ARM64**    | ARMv8 (NEON-capable): Raspberry Pi 4+, AWS Graviton, Ampere Altra, etc.                                                                                                                                                                                                                                                | Modern ARM server with 16+ GB RAM                                                |

### Resources

| Resource              | Minimum                                                  | Recommended                                                                                                                                                 |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RAM**               | 8 GB                                                     | 16 to 32 GB if you load several local models at once (chat + embed + rerank + vision); the actual footprint scales with the size and quantization you pick |
| **GPU / Accelerator** | none required (CPU-only inference works)                 | Apple Silicon (Metal) · any NVIDIA / AMD / Intel Arc GPU (Vulkan) · NVIDIA GPU + matching CUDA toolkit (opt-in CUDA-native wheels, see [Install](#install)) |
| **Disk**              | 2 GB (models + data)                                     | 10+ GB if you load multiple models                                                                                                                          |

Each active inference role (chat, embed, rerank, vision) runs in its own subprocess to keep the TUI responsive; the memory you need follows the size of the models you keep warm. Cloud models are optional, via the `[litellm]` extra (see [Install](#install)).

## Install

### Prerequisites

- Python 3.11, 3.12, 3.13, or 3.14
- **Optional** (for scanned PDF / image OCR): [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a GGUF vision model (see [vision OCR](docs/usage.md#vision-models))

No external services needed. lilbee downloads and runs models locally via llama-cpp-python.

### Default install (recommended for almost everyone)

The default wheel ships with **runtime CPU dispatch** (works on every x86_64 CPU from Sandy Bridge / 2011 forward) and **GPU acceleration via Vulkan on Linux/Windows** (covers NVIDIA / AMD / Intel Arc) or **Metal on macOS arm64**. If you have _any_ modern GPU, this gets you GPU-accelerated inference with zero opt-in.

**pip:**

```bash
pip install --pre lilbee
```

**uv:**

```bash
uv tool install --prerelease=allow lilbee
```

Verify the install works on your hardware:

```bash
lilbee self-check
```

This downloads a tiny model (~90 MB), runs an inference, and an embedding. Exits 0 with `SELF-CHECK PASSED` on success. It only verifies your hardware; it does not configure a working model. Launch the TUI (`lilbee`) and pick a chat + embedding model on the welcome screen, or from the CLI install an embedding model (required before `lilbee add`) and a chat model:

```bash
lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF   # embedding model; needed to index/search
lilbee model browse                                     # pick a chat model interactively
```

### Other ways to install

If you'd rather not install Python, the prebuilt binary ships through several package managers. They all wrap the same release artifact (its own Python interpreter and llama-cpp backend), so pick whichever fits your setup.

| Method | Command | Notes |
| --- | --- | --- |
| **Homebrew** (macOS arm64, Linux x86_64) | `brew tap tobocop2/lilbee && brew install lilbee` | Clears the macOS quarantine attribute on install, so Gatekeeper won't block the first launch. (The macOS binary is unsigned; the formula handles it.) |
| **AUR** (Arch Linux) | `paru -S lilbee` | Package `lilbee`, works with `yay` / `pacaur` / any helper. Wraps the Linux x86_64 binary, no compilation. |
| **Docker** | `docker run --rm -v lilbee-data:/home/lilbee/data ghcr.io/tobocop2/lilbee:latest --help` | Image on the GitHub Container Registry, tagged with the version and `latest`. Data lives at `/home/lilbee/data`; mount a volume there. |
| **Nix** (NixOS, nix-darwin, any host with nix) | `nix run github:tobocop2/lilbee` | On Linux the flake bundles `glibc`, `libgomp`, and `vulkan-loader` so it runs on bare NixOS. |

Or grab the [standalone binary](#pre-built-executables-no-python-required) directly (no package manager, no Python).

<a id="linux-runtime-requirements"></a>

### Linux runtime requirements

The Linux x86_64 wheel links against the Vulkan loader at runtime so it can fall back from GPU to CPU on a single binary. Most desktop distros (Ubuntu 22.04+, Pop!_OS, Mint) ship `libvulkan1` by default. Bare Arch / Fedora / Alpine images do not, and `lilbee self-check` will fail with `cannot open shared object file: libvulkan.so.1`. Install the loader once.

```bash
# Arch / Manjaro
sudo pacman -S vulkan-icd-loader

# Fedora / RHEL
sudo dnf install vulkan-loader

# Debian / Ubuntu (only if missing)
sudo apt-get install libvulkan1
```

### NVIDIA users wanting CUDA-native (5-15% faster than Vulkan)

The default wheel already uses your NVIDIA GPU through Vulkan. **You only need a CUDA wheel if you want the absolute last bit of performance** out of CUDA-native kernels.

CUDA wheels live on a per-CUDA-version index (because each is linked against a specific CUDA runtime). Pick the index that matches your installed CUDA toolkit:

```bash
# CUDA 12.4 (pip)
pip install --pre lilbee --extra-index-url https://tobocop2.github.io/lilbee/cu124/

# CUDA 12.4 (uv tool)
uv tool install --prerelease=allow lilbee \
  --extra-index-url https://tobocop2.github.io/lilbee/cu124/

# CUDA 12.5 (pip)
pip install --pre lilbee --extra-index-url https://tobocop2.github.io/lilbee/cu125/

# CUDA 12.5 (uv tool)
uv tool install --prerelease=allow lilbee \
  --extra-index-url https://tobocop2.github.io/lilbee/cu125/
```

Don't know your CUDA version? `nvidia-smi` (look at the top-right corner). NVIDIA driver 555+ supports CUDA 12.5; older drivers may need cu124.

### Intel Mac

Intel Mac wheels ship from a separate index because they're cross-compiled and the lane is best-effort:

```bash
pip install --pre lilbee --extra-index-url https://tobocop2.github.io/lilbee/cpu/
```

If a wheel isn't available for your Python version, pip falls back to the sdist and builds locally (you'll need a working Xcode toolchain).

### Pre-built executables (no Python required)

A single binary with its own Python runtime, always the latest pre-release. `chmod +x` it and run (e.g. `./lilbee-linux-x86_64 self-check`).

| Platform           | Download                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **Linux x86_64**   | [lilbee-linux-x86_64](https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-linux-x86_64)             |
| **macOS arm64**    | [lilbee-macos-arm64](https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-macos-arm64)               |
| **Windows x86_64** | [lilbee-windows-x86_64.exe](https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-windows-x86_64.exe) |

The Linux binary needs **glibc 2.28+** (Fedora 28+, RHEL/Rocky/Alma 8+, Debian 10+, Ubuntu 18.10+, Amazon Linux 2023, current Arch); on older systems use `uv tool install lilbee` or [Docker](#docker).

### Optional extras

**lilbee works out of the box.** Three optional extras add more: `[crawler]` indexes websites, `[litellm]` bridges to popular hosted model providers, `[graph]` adds concept-graph search. Append the name in brackets to a `pip` or `uv tool install`:

| Extra            | What it does, and how to add it                                                                                                                                                                                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`[crawler]`**  | **Index websites alongside your files.** Recursive crawling with Playwright, live progress, cancel, hash-based change detection, SSRF protection, rate limits. `pip install --pre 'lilbee[crawler]'`                                                                                       |
| **`[litellm]`**  | **Bridge to popular hosted model providers** for chat, vision, or embeddings while other roles stay local. You provide the API key; the TUI shows a persistent warning whenever a hosted model is active, and chunks sent to that provider leave your machine. `pip install --pre 'lilbee[litellm]'` |
| **`[graph]`**    | **Concept-graph search.** Extracts the ideas in your documents and uses how they relate to surface results plain keyword matching misses. No extra model calls. `pip install --pre 'lilbee[graph]'`                                                                                        |

Install multiple at once:

```bash
pip install --pre 'lilbee[graph,crawler,litellm]'
# or
uv tool install --prerelease=allow 'lilbee[graph,crawler,litellm]'
```

Combine with `--extra-index-url` for CUDA:

```bash
uv tool install --prerelease=allow 'lilbee[graph,crawler]' \
  --extra-index-url https://tobocop2.github.io/lilbee/cu125/
```

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration and details.

### Upgrading to the latest pre-release

```bash
# pip
pip install --upgrade --pre lilbee

# uv (force reinstall)
uv tool install --reinstall --prerelease=allow lilbee
```

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

An [Obsidian plugin](https://tobocop2.github.io/obsidian-lilbee/) pairs with lilbee for users who want a GUI alongside the terminal. It runs `lilbee serve` in the background, so there's no separate service to manage, and every citation in chat or wiki opens a Source Preview that scrolls to the exact passage in the original document. Install via [BRAT](https://github.com/TfTHacker/obsidian42-brat); see the [plugin README](https://github.com/tobocop2/obsidian-lilbee#quick-start) for setup.

## Interactive chat

Running `lilbee` or `lilbee chat` enters the TUI. Type `/` to see the full slash-command list inline, or check the [slash-command reference in the usage guide](docs/usage.md#slash-commands). Slash commands and paths tab-complete; background jobs appear in the Task Center and are cancellable with `/cancel`.

## Supported formats

Text extraction powered by [Kreuzberg], code chunking by [tree-sitter]. Structured formats (XML, JSON, CSV) get embedding-friendly preprocessing. This list is not exhaustive; Kreuzberg supports additional formats beyond what's listed here.

| Format       | Extensions                                                                                                                                              | Requires                                                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PDF          | `.pdf`                                                                                                                                                  | none                                                                                                                                                                                             |
| Scanned PDF  | `.pdf` (no extractable text)                                                                                                                            | [Tesseract](https://github.com/tesseract-ocr/tesseract) (auto, plain text), or a GGUF vision model via the native mtmd backend (recommended, preserves tables, headings, and layout as markdown) |
| Office       | `.docx`, `.xlsx`, `.pptx`                                                                                                                               | none                                                                                                                                                                                             |
| eBook        | `.epub`                                                                                                                                                 | none                                                                                                                                                                                             |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`                                                                                                       | [Tesseract](https://github.com/tesseract-ocr/tesseract)                                                                                                                                          |
| Data         | `.csv`, `.tsv`                                                                                                                                          | none                                                                                                                                                                                             |
| Structured   | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml`                                                                                                              | none                                                                                                                                                                                             |
| Code         | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | none                                                                                                                                                                                             |

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
