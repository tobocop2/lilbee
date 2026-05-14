<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/lilbee-logo-dark.svg">
    <img alt="lilbee" src="docs/lilbee-logo-light.svg" width="340">
  </picture>
</p>

<p align="center"><strong>A batteries-included local search engine for your data and code that you can talk to.</strong></p>

<p align="center"><a href="https://tobocop2.github.io/lilbee/">Project site</a> &nbsp;·&nbsp; <a href="https://pypi.org/project/lilbee/">PyPI</a> &nbsp;·&nbsp; <a href="https://tobocop2.github.io/obsidian-lilbee/">Obsidian plugin</a> &nbsp;·&nbsp; <a href="https://tobocop2.github.io/lilbee/api/">API docs</a></p>

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

Point it at your files, notes, and code and ask questions in plain English; every answer links back to the file and line it came from. Point it at nothing and it's just a fast chatbot.

<p align="center">
  <img alt="lilbee chat, owner's manual, cited answers"
       src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.png"
       width="720">
</p>

<details><summary>▶ Watch the chat (90s)</summary>

<p align="center">
  <img alt="lilbee chat demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif" width="720">
</p>
</details>

It's all one program: a full-screen terminal app, a command-line tool, a Model Context Protocol server, an HTTP API, and a Python library. Run it when you want, close it when you're done; nothing left running in the background, no container to keep alive. It runs on your computer; lilbee uses a cloud model only when you pick one.

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
- [Supported formats](#supported-formats)
- [Experimental](#experimental)

---

## Quick start

All the install options are in [Install](#install) below: pip, uv, Homebrew, AUR, Docker, Nix, a standalone binary (no Python), CUDA wheels, or from source. Optional extras (`[crawler]`, `[litellm]`, `[graph]`) are there too.

## Highlights

- **One program, one install.** A model catalog, a search over your own files and code, and a chat. The same executable is also a CLI, a Textual TUI, an MCP server, a REST API, and a Python library. No background daemon, no separate inference server, no vector database to stand up.
- **Answers cite the source line.** Ask a question; get a reply with clickable citations pointing back to the exact line they came from.
- **Bring your own files.** PDFs, Office files, ebooks, code in 150+ languages, scanned pages and photos (OCR), and crawled docs sites turned into searchable markdown.
- **A built-in model catalog.** Browse and pull models straight from Hugging Face Hub, from inside the app. lilbee is the model runtime; no hunting for files yourself.
- **Runs on your computer.** Models, index, and files all stay local. lilbee uses a cloud model only when you pick one, and flags it when it does.
- **Per-project libraries.** Run globally, or drop a `.lilbee/` next to `.git/` the way git does; each domain stays its own clean library.

## Why lilbee

The first evening with a local model is fun. What makes it more than a novelty is grounding: the model needs context from your notes, your files, your code, or it runs out of places to go. lilbee pairs the chat with a real search engine over a set of documents you choose, so a local model can reason over your world and answer with citations you can click back to the source.

Standing this up used to mean a background daemon, a separate inference server, model files fetched by hand, and a retrieval layer glued on top. lilbee folds all of it into one install, in one process, in the terminal. Run it globally, or scope a library per project by dropping a `.lilbee/` next to `.git/`, the same pattern git uses; a focused library answers better than one catch-all pile of everything.

> An [Encarta 99](https://en.wikipedia.org/wiki/Encarta) you build for yourself, from your own files, shaped to your needs.

## Previews

> A still for each screen and flow, with the animated walkthrough one click away under it. The full reel with captions lives in [`docs/demos.md`](docs/demos.md); the tape sources are in [`demos/`](demos).

**Chat.** The default screen. Streaming replies with clickable citations.

<p align="center"><img alt="lilbee chat with cited answers" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.png" width="800"></p>

<details><summary>▶ Watch: ask the owner's manual, get cited answers</summary>

<p align="center"><img alt="lilbee chat demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif" width="800"></p>
</details>

**Setup wizard.** Pick a chat model and an embedding model from the catalog on first run; both pull in the background while you keep working.

<p align="center"><img alt="lilbee setup wizard" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.png" width="800"></p>

<details><summary>▶ Watch: pick chat and embedding models</summary>

<p align="center"><img alt="lilbee setup wizard demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.gif" width="800"></p>
</details>

**Add documents.** `/add <path>` copies a file or folder into the corpus and indexes it; the Task Center shows live progress, and you can keep asking questions while it runs.

<p align="center"><img alt="lilbee add documents and Task Center" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.png" width="800"></p>

<details><summary>▶ Watch: /add a PDF, see Tasks, ask a question</summary>

<p align="center"><img alt="lilbee add demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif" width="800"></p>
</details>

**Model catalog.** Browse models from Hugging Face Hub, pull one with a click, and switch roles without leaving the terminal. `*` marks the developer's pick for each role.

<p align="center"><img alt="lilbee model catalog" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.png" width="800"></p>

<details><summary>▶ Watch: catalog grid, filters, model info, live pull</summary>

<p align="center"><img alt="lilbee model catalog demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif" width="800"></p>
</details>

**Crawl a URL.** `/crawl` opens a modal: paste a URL, pick depth and a page cap, and the page (or a small site) is fetched, converted to markdown, and added to your library. Then ask against it with the same cited-answer flow.

<p align="center"><img alt="lilbee crawl modal" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.png" width="800"></p>

<details><summary>▶ Watch: crawl a Wikipedia page, then ask about it</summary>

<p align="center"><img alt="lilbee crawl demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.gif" width="800"></p>
</details>

**Settings.** ~50 knobs grouped into Models, Ingest, Generation, Retrieval, Display, Crawling, API-Keys, System. Edit in the TUI, environment variables, or a project-local config file.

<p align="center"><img alt="lilbee settings screen" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.png" width="800"></p>

<details><summary>▶ Watch: cycle every settings pane</summary>

<p align="center"><img alt="lilbee settings demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.gif" width="800"></p>
</details>

**Command surface.** `Ctrl+P` opens the Textual command palette; `?` toggles a keybinding cheat sheet; `/help` opens the searchable slash-command catalog.

<p align="center"><img alt="lilbee command palette" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.png" width="800"></p>

<details><summary>▶ Watch: palette + cheat sheet + slash catalog</summary>

<p align="center"><img alt="lilbee command palette demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.gif" width="800"></p>
</details>

**Tour.** Every screen at a glance.

<details><summary>▶ Watch the 60-second tour</summary>

<p align="center"><img alt="lilbee tour" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif" width="800"></p>
</details>

**CLI.** Ollama-style commands for scripts and one-off jobs: `init`, `model pull`, `model list`, `add`, `status`, `sync`, `search`.

<p align="center"><img alt="lilbee CLI tour" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/cli.png" width="800"></p>

<details><summary>▶ Watch: CLI tour</summary>

<p align="center"><img alt="lilbee CLI demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/cli.gif" width="800"></p>
</details>

**Wiki.** Auto-generated concept and entity pages, with drafts awaiting review. (Experimental; still ascii-sketched while the redesign settles.)

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

## What you can do with it

### A library of your own files

Point lilbee at a folder of PDFs, notes, ebooks, or code and it builds a searchable library, with citations that click back to the source line. The pattern works for anything you have a lot of text about: a medical-textbook collection, a field's research papers, a car's service manuals, your company's internal wiki. Whatever you give it becomes searchable, and you can talk to it.

### Grounding for AI agents

lilbee plugs into whatever AI agent you already use, over MCP. Feed it your project's docs, your dependency source, the vendor SDK reference, your design notes, and the agent stops making up function names: it reads the actual code it's about to call, cites the file and line, and says it doesn't know when the answer isn't in the corpus, instead of guessing.

### Offline copies of websites

Install the `[crawler]` extra, point lilbee at a docs site, a wiki, or a vendor's API reference, and the pages get fetched, converted to markdown, and added to your library. From then on you can search or chat with that copy of the site offline, even after it changes or goes down.

### How it's built

lilbee stands on established open-source projects, all embedded in one process:

- [Kreuzberg] parses documents
- [LanceDB] is the embedded search layer
- [tree-sitter] chunks code
- [llama-cpp][llama-cpp-python] runs models locally
- [crawl4ai] and [Playwright] crawl the web
- [Textual] draws the terminal

### Documents, code, and scanned images

Most retrieval tools throw your files at a PDF extractor and call it done. lilbee splits the work by what's being indexed:

- **Prose and structured documents** (90+ formats: PDFs, Office files, ebooks, HTML, and more) go through [Kreuzberg]'s extraction pipeline with heading-aware chunking, so each chunk keeps its section context.
- **Code** goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to real functions, classes, and modules instead of arbitrary line ranges.
- **Scanned PDFs and photos** go through OCR: Tesseract for plain text, a local GGUF vision model that keeps tables and layout as markdown, or a remote vision model.

Retrieval returns things that make sense on their own, not fragments cut through an argument or a function signature.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a model, pick a role. Retrieval and generation expose 50+ settings (chunk size, search strictness, a second-pass re-scorer, how much weight topic relationships carry), edited from the TUI, environment variables, or a project-local config file. Sane defaults out of the box.

### Cloud models, when you want them

lilbee runs entirely on your machine by default. To point a role at a cloud-hosted model, install the `[litellm]` extra and add an API key; the TUI shows a persistent warning whenever a cloud model is active, so it's clear when chunks are leaving the machine.

## TUI

`lilbee` (no args) launches a full Textual terminal app: streaming chat with clickable citations, a model bar with searchable pickers and a Search/Chat toggle, a Task Center for background jobs, and screens for the model catalog, settings, the setup wizard, and the auto-built wiki. Type `/` for the command list; tab completion works everywhere.

See [Previews](#previews) for the shapes and the [usage guide](docs/usage.md) for commands and settings.

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

Each active inference role (chat, embed, rerank, vision) runs in its own subprocess to keep the TUI responsive, so the memory you need follows the size of the models you keep warm.

## Install

**Two routes, and the difference matters:**

- **Into your own Python** with `pip` or `uv` (Python 3.11 to 3.14). Smaller install, picks the fastest CPU code path for your machine at runtime, managed with the tools you already use. Recommended if you have Python.
- **A self-contained bundle**: the standalone binary, or the Homebrew / AUR / Nix / Docker builds that wrap it. Nothing else to install, but a large file on a fixed CPU baseline (a 2013-or-newer x86_64 chip), a touch slower on newer hardware than the `pip` / `uv` wheel. Recommended if you'd rather not deal with Python.

No external services either way; lilbee downloads and runs models locally. Optional, for scanned-PDF / image OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a [GGUF vision model](docs/usage.md#vision-models).

| How | Command | Notes |
| --- | --- | --- |
| **pip** | `pip install --pre lilbee` | Recommended. The default wheel runs on any x86_64 CPU and uses your GPU via Vulkan / Metal automatically. Intel Mac: add `--extra-index-url https://tobocop2.github.io/lilbee/cpu/`. |
| **uv** | `uv tool install --prerelease=allow lilbee` | Same wheel as pip; fetches a Python for you if you need one. |
| **Homebrew** | `brew tap tobocop2/lilbee && brew install lilbee` | macOS arm64 / Linux x86_64. Bundled build; clears the macOS quarantine flag for you. |
| **AUR** | `paru -S lilbee` | Arch Linux. Wraps the Linux x86_64 binary; works with `yay` / `pacaur` / any helper. |
| **Docker** | `docker run --rm -v lilbee-data:/home/lilbee/data ghcr.io/tobocop2/lilbee:latest --help` | GHCR image, tagged by version and `latest`. Data lives at `/home/lilbee/data` — mount a volume there. |
| **Nix** | `nix run github:tobocop2/lilbee` | NixOS, nix-darwin, or any host with nix. On Linux the flake bundles `glibc`, `libgomp`, and `vulkan-loader` so it runs on bare NixOS. |
| **Standalone binary** | [download for your platform &rarr;](https://github.com/tobocop2/lilbee/releases/latest) | One file, own Python runtime, no `pip` needed. Linux needs glibc 2.28+; the macOS / Windows builds are unsigned (`xattr -d com.apple.quarantine ./lilbee-macos-arm64` if Gatekeeper blocks it). |
| **CUDA-native** | `pip install --pre lilbee --extra-index-url https://tobocop2.github.io/lilbee/cu125/` | Only for the last bit of NVIDIA speed; the default wheel already uses your GPU via Vulkan. `cu121` / `cu124` / `cu125` — match `nvidia-smi`. |
| **From source** | `git clone https://github.com/tobocop2/lilbee && cd lilbee && uv sync && uv run lilbee` | For hacking on it. Needs `git` and `uv`. |

Then check it runs and pick a model:

```bash
lilbee self-check    # ~90 MB download; runs an inference + an embedding; "SELF-CHECK PASSED" on success
lilbee               # launch the terminal app; pick a chat + embedding model on the welcome screen
```

Everything past that (commands, slash commands, settings, the API) lives in the [usage guide](docs/usage.md).

### Linux runtime requirements

The Linux x86_64 wheel and binary link the Vulkan loader at runtime. Most desktop distros (Ubuntu 22.04+, Pop!_OS, Mint) ship `libvulkan1`; bare Arch / Fedora / Alpine images don't, and `lilbee self-check` fails with `cannot open shared object file: libvulkan.so.1`. Install it once: `sudo pacman -S vulkan-icd-loader` (Arch / Manjaro), `sudo dnf install vulkan-loader` (Fedora, RHEL), or `sudo apt-get install libvulkan1` (Debian, Ubuntu).

### Optional extras

These only matter for a `pip` or `uv` install: add the name in brackets, e.g. `pip install --pre 'lilbee[crawler,litellm]'` (combine multiple, and `--extra-index-url` still works for CUDA). The standalone binary and the Homebrew / AUR / Nix / Docker builds already include all three. lilbee works without them either way.

| Extra | What it adds |
| --- | --- |
| `[crawler]` | Index websites alongside your files: crawl a docs site or wiki to markdown, then search it offline. Recursive crawl with Playwright, live progress, cancel, change detection, SSRF guards, rate limits. |
| `[litellm]` | Bridge to popular hosted model providers for chat, vision, or embeddings while other roles stay local. You provide the key; the TUI flags whenever a hosted model is active, and chunks sent to it leave your machine. |
| `[graph]` | Concept-graph search: extracts the ideas in your documents and uses how they relate to surface matches plain keyword search misses. No extra model calls. |

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration.

### Upgrading

```bash
pip install --upgrade --pre lilbee
# or
uv tool install --reinstall --prerelease=allow lilbee
```

## Agent integration

lilbee is a retrieval backend for AI coding agents, over MCP or a JSON CLI: search, document lifecycle, crawling, model management, and the wiki, all exposed as tools, scoped to documents, wiki pages, or both. The repo ships a drop-in [`AGENTS.md`](demos/AGENTS.md), a [`lilbee-worker` subagent](demos/.opencode/agents/lilbee-worker.md) for the long ops, and a reusable [`lilbee-mcp` skill](docs/agent-skills/lilbee-mcp/SKILL.md) (opencode / Claude Skill format) that documents the full MCP surface. See [docs/agent-integration.md](docs/agent-integration.md) for how to wire it up.

**Writing Godot 4 code against an indexed class reference.** opencode indexes Godot 4's class XMLs (810 files, 3449 chunks) via the `lilbee-worker` subagent, then `lilbee_search`-es for `AStarGrid2D`, `TileMap`, `RandomNumberGenerator`, and friends as it writes a procedural level generator. Every API call is backed by a `godot-classes/<Class>.xml:line` citation. (See [the benchmark](docs/benchmarks/godot-level-generator.md): 4 hallucinated APIs without lilbee, 0 with.)

<p align="center"><img alt="opencode + lilbee MCP writing Godot 4 code with file:line citations" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.png" width="800"></p>

<details><summary>▶ Watch: index the Godot class reference, then write cited code</summary>

<p align="center"><img alt="opencode + lilbee MCP, Godot class reference demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.gif" width="800"></p>
</details>

**Querying an owner's manual.** The same shape, smaller corpus: index `cv-manual.pdf`, ask for the oil capacity, get a page-cited answer.

<p align="center"><img alt="opencode + lilbee MCP over a manual" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-manual.png" width="800"></p>

<details><summary>▶ Watch: index a PDF via lilbee MCP, then answer with a citation</summary>

<p align="center"><img alt="opencode + lilbee MCP, manual demo" src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-manual.gif" width="800"></p>
</details>

## HTTP Server

`lilbee serve` starts a REST API any tool or GUI can hit: search (with SSE streaming), document lifecycle, crawling, model management, configuration. See the [API reference](https://tobocop2.github.io/lilbee/api/) for the OpenAPI schema and the [usage guide](docs/usage.md) for options.

The [Obsidian plugin](https://tobocop2.github.io/obsidian-lilbee/) is a GUI built on it: it runs `lilbee serve` in the background, and every citation opens a Source Preview scrolled to the exact passage. Install via [BRAT](https://github.com/TfTHacker/obsidian42-brat); the [plugin README](https://github.com/tobocop2/obsidian-lilbee#quick-start) has setup.

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

[Kreuzberg]: https://github.com/kreuzberg-dev/kreuzberg
[LanceDB]: https://lancedb.com
[llama-cpp-python]: https://github.com/abetlen/llama-cpp-python
[crawl4ai]: https://github.com/unclecode/crawl4ai
[Playwright]: https://playwright.dev
[Textual]: https://textual.textualize.io
[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
