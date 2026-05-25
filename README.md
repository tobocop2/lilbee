<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/lilbee-logo-dark.svg">
    <img alt="lilbee" src="docs/lilbee-logo-light.svg" width="340">
  </picture>
</p>

<p align="center"><strong>Run local AI models, search your own files and code, and crawl the web, all in one program.</strong></p>

<p align="center"><a href="https://lilbee.sh/">Project site</a> &nbsp;·&nbsp; <a href="https://pypi.org/project/lilbee/">PyPI</a> &nbsp;·&nbsp; <a href="https://obsidian.lilbee.sh/">Obsidian plugin</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/api/">REST API</a></p>

<p align="center">
  <a href="https://github.com/tobocop2/lilbee/releases"><img src="https://img.shields.io/github/v/release/tobocop2/lilbee?include_prereleases&label=latest%20release" alt="Latest release (incl. pre-releases)"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee?include_prereleases&label=PyPI" alt="lilbee on PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml"><img src="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://lilbee.sh/coverage/"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-blue.svg" alt="Typed"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-ELv2-blue.svg" alt="License: Elastic License 2.0"></a>
  <a href="https://pepy.tech/project/lilbee"><img src="https://static.pepy.tech/badge/lilbee/month" alt="PyPI downloads / month"></a>
  <a href="https://github.com/tobocop2/lilbee/releases"><img src="https://img.shields.io/github/downloads/tobocop2/lilbee/total" alt="GitHub release downloads"></a>
</p>

A batteries-included local search engine you can talk to: it runs the AI models, indexes your files and code, crawls the web, and plugs into your coding agent, so there's nothing else to install or set up. Ask in plain English; every answer cites the file and line.

![lilbee chat with cited answers from a Crown Victoria owner's manual](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

It's all one program: you never stand up a separate model server, a vector database, or a container — lilbee runs the models and keeps the index itself. Reach it as a full-screen terminal app, a command-line tool, a Model Context Protocol server, an HTTP API, or a Python library. Close it and it's gone, or run it as a service if you'd rather keep it warm. It runs on your computer; lilbee uses a cloud model only when you pick one.

> **Tutorial reel:** every demo on this page (and the extras) as a real video player at [**lilbee.sh/tutorial.html**](https://lilbee.sh/tutorial.html).

> ## ⚠️ Beta software
>
> lilbee is in **active beta** development. Every release on PyPI is a pre-release; you must use `--pre` (or uv's `--prerelease=allow`) when installing. Interfaces, command names, and on-disk formats may shift between betas. Feedback, bug reports, and issues are very welcome; that's the whole point of the beta.
>
> Latest pre-release (always): [lilbee on PyPI →](https://pypi.org/project/lilbee/)

---

- [Quick start](#quick-start)
- [Tutorial reel](https://lilbee.sh/tutorial.html) (long-form videos)
- [Highlights](#highlights)
- [Why lilbee](#why-lilbee)
- [What you can do with it](#what-you-can-do-with-it)
- [TUI](#tui)
- [Hardware requirements](#hardware-requirements)
- [Install](#install)
- [Agent integration](#agent-integration)
- [HTTP Server](#http-server) · [REST API reference](https://lilbee.sh/api/)
- [Supported formats](#supported-formats)
- [Experimental](#experimental)
- [Built on](#built-on)

---

## Quick start

Two recommended ways to use lilbee, depending on whether you're the one driving:

- **Run `lilbee`** for the full-screen terminal app. A welcome wizard picks a chat and embedding model, then you index files, search, and chat without leaving the TUI. The Settings screen exposes every retrieval knob (search depth, distance threshold, reranker, chunking) so you can tune lilbee to your library shape.
- **Wire it into your agent over MCP.** Any MCP-aware coding agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets it can quote. Agents can also *fine-tune lilbee on the fly* via `lilbee_settings_set`. Drop in the [lilbee-mcp skill](docs/agent-skills/lilbee-mcp/SKILL.md) and the agent reads the full surface: every tool, every retrieval knob, and when to widen for prose vs narrow for code. See [Agent integration](#agent-integration).

Defaults are sane for chatting with code, documentation, crawled sites, and long PDFs. Every retrieval setting is writable from the TUI Settings screen, the `/set` slash command, MCP `lilbee_settings_set`, or `config.toml`. When answers feel thin or noisy, the usual knobs are `top_k`, `max_distance`, or `diversity_max_per_source`.

CLI, the HTTP API, env vars, and `config.toml` are there for scripting, headless runs, and custom integrations. See the [usage guide](docs/usage.md).

## Highlights

- **Answers cite the source line.** Click a citation, jump to the file at the exact line. When the answer isn't in your library, lilbee says so instead of inventing one.
- **A real retrieval pipeline, not keyword search.** Hybrid keyword + vector, a concept graph, optional reranking, all from published research. [50+ settings](docs/usage.md) to tune.
- **It brings and runs the models itself.** Browse Hugging Face, pull a model, give it a role (chat, embedding, vision, reranking); lilbee runs it on Metal, Vulkan, or CUDA. You never point it at a server you set up.
- **Indexes anything textual.** PDFs, Office, ebooks, code in 150+ languages, scanned pages (OCR), crawled sites.
- **Per-project libraries.** Keep one library for everything, or give each project its own.
- **One install, many surfaces.** TUI, CLI, [MCP server](#agent-integration), [REST API](https://lilbee.sh/api/), and Python library. Nothing to stand up: it loads on demand and runs as a service only if you want it warm.
- **One file instead of an Electron app.** The standalone binary is 250-365 MB and bundles the whole thing: search engine, web crawler, MCP server, HTTP server, and terminal UI, with Python and llama.cpp included. Comparable desktop AI apps are multi-hundred-MB to multi-GB Electron bundles that do less.
- **Works with your coding agent.** Wire lilbee into any MCP-aware agent and it searches your library and answers from your actual files, citations and all; it can even widen retrieval or swap models mid-conversation via `lilbee_settings_set`.
- **Your hardware, put to work.** Your machine can do a lot more than you're using it for. lilbee runs local models on hardware you already own, no cloud account required.
- **It works, and the demos prove it.** Every GIF and tutorial reel here is recorded live on real hardware, nothing staged. Backed by 100% test coverage, full typing, and CI on macOS, Linux, and Windows.
- **Up and running in one command.** Install, run `lilbee`, and a first-run wizard pulls a model and drops you into chat.

## Why lilbee

The first evening with a local model is fun. What makes it more than a novelty is grounding: the model needs context from your notes, your files, your code. lilbee pairs the chat with a real search engine over documents you choose, so a local model can reason over your world and answer with citations you can click back to the source.

Standing this up used to mean a background daemon, a separate inference server, model files fetched by hand, and a retrieval layer glued on top. lilbee folds all of it into one install, driven from the terminal. Run it globally, or scope a library per project by dropping a `.lilbee/` next to `.git/`, the same pattern git uses; a focused library answers better than one catch-all pile of everything.

> **The long-term goal:** make local AI practical and useful, for questions and for code, on hardware you already own. No token budgets to ration, no provider to depend on; the cloud's there when you want it. An [Encarta 99](https://en.wikipedia.org/wiki/Encarta) you build for yourself, over your files, your code, even the web pages you save: read it yourself, or have your agent read it for you.

## What you can do with it

### A library of your own files

Point lilbee at a folder of PDFs, notes, ebooks, or code and it builds a searchable library, with citations that click back to the source line. The pattern works for anything you have a lot of text about: a medical-textbook collection, a field's research papers, a car's service manuals, your company's internal wiki. Whatever you give it becomes searchable, and you can talk to it.

![/add a PDF, watch the Task Center, ask a cited question](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

### Already using an MCP-aware agent? Hand setup to it.

If you've already got an MCP-aware coding agent running, it can do the setup for you: browse the model catalog, pull picks, wire them into the embedding / reranker / vision roles, and tune retrieval for your library and question style. No TUI, no config file, no restart. Agents already understand search engines, so the right knobs to move are obvious to them. See the [`lilbee-mcp` skill](docs/agent-skills/lilbee-mcp/SKILL.md) for the workflow and example prompts.

### Opencode integration (coming)

Local-model [opencode](https://opencode.ai) support is coming in [#267](https://github.com/tobocop2/lilbee/pull/267), with tool-calling working across many GGUF families.

The demo shows a small local model (Qwen) given a specific instruction: when its first search comes back thin, widen lilbee's search settings and search again. The second pass returns the full function bodies with file:line citations. A more capable model would do the same from a higher-level prompt like "improve your search results." Read the [lilbee-mcp skill](docs/agent-skills/lilbee-mcp/SKILL.md) to teach your own model the pattern.

![agent fine-tunes lilbee mid-conversation: outline → widened retrieval → source with file:line citations](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code-self-tune.gif)

### Grounding for AI agents

Once configured, lilbee plugs into whatever agent you use, over MCP. Feed it your project's docs, your dependency source, your API docs, your design notes; the agent stops making up function names and instead reads the actual code, cites file and line, and says it doesn't know when the answer isn't in your library.

Your files, the search index, and the embeddings stay on your computer. The agent calls `lilbee_search` and gets back cited snippets. The demo below is lilbee talking to lilbee: an agent indexes lilbee's own source, then answers questions about how lilbee works with file:line citations.

![an agent indexes lilbee's own source through lilbee's MCP server, then answers questions about how lilbee works with file:line citations](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code.gif)

### Offline copies of websites

Install the `[crawler]` extra, point lilbee at a docs site, a wiki, or a vendor's API reference, and the pages get fetched, converted to markdown, and added to your library. From then on you can search or chat with that copy of the site offline, even after it changes or goes down.

![/crawl a Wikipedia page, then ask a cited question against it](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.gif)

### Documents, code, and scanned images

lilbee splits indexing by what's being read:

- **Prose and structured documents** (PDFs, Office files, ebooks, HTML, 90+ formats) go through [Kreuzberg] with heading-aware chunking, so each chunk keeps its section context.
- **Code** goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to functions, classes, and modules instead of arbitrary line ranges.
- **Scanned PDFs and photos** go through OCR: Tesseract for plain text, or a local / remote vision model that keeps tables and layout as markdown.

Retrieval returns things that make sense on their own, not fragments cut through an argument or a function signature.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a model, pick a role. Retrieval and generation expose 50+ settings (chunk size, search strictness, reranker depth, and more), editable from the TUI, env vars, or a project-local config file. Sane defaults.

![browse the model catalog, search Hugging Face Hub, pull a model live](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

### See when a model won't load before you download it

Hugging Face has thousands of GGUFs, but the bundled llama.cpp only supports a subset of architectures and brand-new ones take time to reach the pinned runtime. lilbee tags incompatible models in the catalog and refuses the download (with an override confirm), so you don't wait through a multi-GB pull only to hit "unsupported architecture" at load.

![search HF Hub for deepseek-v4, see the unsupported pill in grid and list view](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-unsupported.gif)

### Cloud models, when you want them

lilbee runs entirely on your machine by default. Two ways to use a cloud model when you want one:

- **Bring your own key.** Install the `[litellm]` extra, add an API key, then point any role (chat, embedding, vision, rerank) at a cloud model from the same catalog. The TUI shows a warning the whole time a cloud model is on.
- **Pair lilbee with a cloud agent over MCP.** Your files, the embeddings, and the index stay local. Any MCP-aware agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets.

Either way, your files and the index stay on your computer. Only what you ask and the snippets needed to answer it get sent to the cloud model.

## TUI

`lilbee` (no args) launches a full Textual terminal app: streaming chat with clickable citations, a model bar with searchable pickers and a Search/Chat toggle, a Task Center for background jobs, and screens for the model catalog, settings, the setup wizard, and the auto-built wiki. Type `/` for the command list; tab completion works everywhere.

![sweep through every TUI screen](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif)

`Ctrl+P` opens the Textual command palette, `?` toggles the keybinding cheat sheet, `/help` opens the slash-command catalog. Every action lilbee can take is reachable from one of those three.

![command palette, keybinding cheat sheet, slash-command catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.gif)

Every GIF on this page (plus the extras that don't fit here) is at [**lilbee.sh/tutorial.html**](https://lilbee.sh/tutorial.html) as an embedded video with long-form captions. Tape sources are in [`demos/`](demos). For commands and settings, see the [usage guide](docs/usage.md).

## Hardware requirements

Standalone mode runs entirely on your machine. No cloud required. **Minimum:** Apple Silicon Mac, or a 64-bit Intel/AMD CPU from 2013+, or an ARMv8 Linux box; 8 GB RAM, 2 GB disk.

<details>
<summary>Full platform and resource breakdown</summary>

| Platform | Minimum | Recommended |
| --- | --- | --- |
| **macOS arm64** | Apple Silicon (M1 or newer), macOS 11+ | M-series Pro / Max / Ultra |
| **Linux x86_64** | 64-bit Intel/AMD from 2013+ ([`x86-64-v3`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels)) | Modern Intel/AMD CPU + an NVIDIA, AMD, or Intel Arc GPU |
| **Windows x86_64** | 64-bit Intel/AMD from 2013+ (`x86-64-v3`), Windows 10/11 | Modern desktop / workstation CPU + GPU |
| **Linux ARM64** | ARMv8 NEON-capable (Raspberry Pi 4+, AWS Graviton, Ampere Altra) | Modern ARM server with 16+ GB RAM |

| Resource | Minimum | Recommended |
| --- | --- | --- |
| **RAM** | 8 GB | 16 to 32 GB to keep several local models warm at once (chat + embed + rerank + vision); actual footprint scales with the sizes and quantizations you pick |
| **GPU / Accelerator** | none required (CPU-only works) | Apple Silicon (Metal) · NVIDIA / AMD / Intel Arc (Vulkan) · NVIDIA + CUDA toolkit (opt-in CUDA wheels, see [Install](#install)) |
| **Disk** | 2 GB | 10+ GB for multiple models |

</details>

## Install

**Two routes, and the difference matters:**

- **Into your own Python** with `pip` or `uv` (Python 3.11 to 3.14). Uses the Python and tooling you already have, picks the fastest CPU code path for your machine at runtime, and upgrades like any other package. Recommended if you have Python.
- **A self-contained bundle**: the standalone binary, or the Homebrew / AUR / Nix / Docker builds that wrap it. Nothing else to install. The trade-off is a single large download — it bundles its own Python runtime, `llama.cpp`, and the optional extras — and a small cold-start cost the first time it self-extracts. Recommended if you'd rather not deal with Python.

Have an NVIDIA GPU? Both routes have a CUDA build that's faster than the default Vulkan path. Skip to [On NVIDIA hardware](#on-nvidia-hardware).

No external services either way; lilbee downloads and runs models locally. Optional, for scanned-PDF / image OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a [GGUF vision model](docs/usage.md#vision-models).

| How | Command | Notes |
| --- | --- | --- |
| **pip** | `pip install --pre lilbee` | Recommended. The default wheel runs on any x86_64 CPU and uses your GPU via Vulkan / Metal automatically. Intel Mac: add `--extra-index-url https://lilbee.sh/cpu/` ([browse wheels](https://lilbee.sh/cpu/lilbee/)). |
| **uv** | `uv tool install --prerelease=allow lilbee` | Same wheel as pip; fetches a Python for you if you need one. |
| **Homebrew** | `brew tap tobocop2/lilbee && brew install lilbee` | macOS arm64 / Linux x86_64. Bundled build; clears the macOS quarantine flag for you. |
| **AUR** | `paru -S lilbee` | Arch Linux. Wraps the Linux x86_64 binary; works with `yay` / `pacaur` / any helper. |
| **Docker** | `docker run --rm -v lilbee-data:/home/lilbee/data ghcr.io/tobocop2/lilbee:latest --help` | GHCR image, tagged by version and `latest`. Data lives at `/home/lilbee/data`. Mount a volume there. |
| **Nix** | `nix run github:tobocop2/lilbee` | NixOS, nix-darwin, or any host with nix. On Linux the flake bundles `glibc`, `libgomp`, and `vulkan-loader` so it runs on bare NixOS. |
| **Standalone binary** | [download for your platform &rarr;](https://github.com/tobocop2/lilbee/releases/latest) | One file, own Python runtime, no `pip` needed. Linux needs glibc 2.28+; the macOS / Windows builds are unsigned (`xattr -d com.apple.quarantine ./lilbee-macos-arm64` if Gatekeeper blocks it). |
| **From source** | `git clone https://github.com/tobocop2/lilbee && cd lilbee && uv sync && uv run lilbee` | For hacking on it. Needs `git` and `uv`. |

### On NVIDIA hardware

The default Vulkan build works on NVIDIA cards, but there's a dedicated CUDA build that's faster on NVIDIA hardware and sidesteps the iGPU + dGPU Vulkan-loader crash on Windows.

| | Command |
| --- | --- |
| **pip** | `pip install --pre lilbee --extra-index-url https://lilbee.sh/cu125/` |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-cuda` |
| **AUR** | `paru -S lilbee-cuda` |
| **Nix** | `nix run github:tobocop2/lilbee#lilbee-cuda` |
| **Binary** | [`lilbee-linux-x86_64-cu125`](https://github.com/tobocop2/lilbee/releases/latest) or [`lilbee-windows-x86_64-cu125.exe`](https://github.com/tobocop2/lilbee/releases/latest) |

Same `lilbee` command after install. The CUDA runtime is bundled; you only need the NVIDIA driver. Already have the regular `lilbee` installed? On AUR `paru -S lilbee-cuda` swaps it automatically; on Homebrew run `brew uninstall lilbee` first. Older driver? `cu124` and `cu121` ship via the matching wheel indexes and as direct-download Linux binaries on the release page.

Then check it runs and pick a model:

```bash
lilbee self-check    # ~90 MB download; runs an inference + an embedding; "SELF-CHECK PASSED" on success
lilbee               # launch the terminal app; pick a chat + embedding model on the welcome screen
```

The [usage guide](docs/usage.md) covers the rest: TUI screens, slash commands, CLI, HTTP server, MCP, env vars, and `config.toml`.

### Linux runtime requirements

The Linux x86_64 wheel and binary link the Vulkan loader at runtime. Most desktop distros (Ubuntu 22.04+, Pop!_OS, Mint) ship `libvulkan1`; bare Arch / Fedora / Alpine images don't, and `lilbee self-check` fails with `cannot open shared object file: libvulkan.so.1`. Install it once: `sudo pacman -S vulkan-icd-loader` (Arch / Manjaro), `sudo dnf install vulkan-loader` (Fedora, RHEL), or `sudo apt-get install libvulkan1` (Debian, Ubuntu).

### Optional extras

These only matter for a `pip` or `uv` install: add the name in brackets, e.g. `pip install --pre 'lilbee[crawler,litellm]'` (combine multiple, and `--extra-index-url` still works for CUDA). The standalone binary and the Homebrew / AUR / Nix / Docker builds already include all three. lilbee works without them either way.

| Extra | What it adds |
| --- | --- |
| `[crawler]` | Index websites alongside your files: crawl a docs site or wiki to markdown, then search it offline. |
| `[litellm]` | Bridge to hosted model providers for chat, vision, or embeddings while other roles stay local. The TUI flags when a hosted role is active. |
| `[graph]` | Concept-graph search: extracts the ideas in your documents and uses how they relate to surface matches plain keyword search misses. No extra model calls. |

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration.

### Upgrading

```bash
pip install --upgrade --pre lilbee
# or
uv tool install --reinstall --prerelease=allow lilbee
```

## Agent integration

Drop the [`lilbee-mcp` skill](docs/agent-skills/lilbee-mcp/SKILL.md) into `.opencode/skills/` or `.claude/skills/`, register lilbee as an MCP server, and any MCP-aware coding agent can search your library, swap models, and tune retrieval. The skill is the single entry point: it documents every tool, the workflows the agent should follow, and points to drop-in `AGENTS.md` and worker-subagent starters under [`examples/agent-integration/`](examples/agent-integration/).

**The demos below use opencode driving a cloud model. lilbee stays local; only the queries and the returned chunks cross the wire to the cloud model.** Local-model opencode integration is on the way across many GGUF families: see [Opencode integration (coming)](#opencode-integration-coming) above.

Live-indexing example: opencode (cloud model) indexes a Godot 4 pathfinding subset (~3s), then `lilbee_search`-es for `AStarGrid2D` and answers method-by-method against your *local* files.

![an MCP-driven coding agent indexes a small local godot subset and answers with cited methods](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot-search.gif)

The same shape scales up. Pre-index Godot 4's full class reference (810 XMLs, 3449 chunks) and the same opencode + cloud setup can write a procedural level generator with every API call backed by a `godot-classes/<Class>.xml:line` citation; the [side-by-side benchmark](docs/benchmarks/godot-level-generator.md) measured 4 hallucinated APIs without lilbee, 0 with.

![cited codegen against the full Godot class reference](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.gif)

## HTTP Server

The HTTP server exposes a REST API any tool or GUI can hit: search (with SSE streaming), document lifecycle, crawling, model management, configuration. See the [REST API reference](https://lilbee.sh/api/) and the [usage guide](docs/usage.md#http-server) for setup.

The [Obsidian plugin](https://obsidian.lilbee.sh/) is a GUI built on it: it starts the HTTP server in the background, and every citation opens a Source Preview scrolled to the exact passage. Install via [BRAT](https://github.com/TfTHacker/obsidian42-brat); the [plugin README](https://github.com/tobocop2/obsidian-lilbee#quick-start) has setup.

### Running as a service (optional)

For tools that talk to lilbee's HTTP REST API (the Obsidian plugin, custom GUIs, anything hitting `/api/*`), your OS launcher can keep the HTTP server warm so requests skip the cold-start.

This is the only lilbee surface that benefits from a system daemon. The TUI, `lilbee chat`, the MCP server, and the rest of the CLI are designed to load on demand and exit when you close them. There's no always-on process to babysit, which is uncommon in this corner of the local-AI ecosystem.

Pull a chat and embedding model first; all recipes pin the server to `127.0.0.1:42697`.

| Platform | Command |
| --- | --- |
| **macOS (Homebrew)** | `brew services start lilbee` |
| **Linux (Arch / AUR)** | `systemctl --user enable --now lilbee` (add `loginctl enable-linger $USER` on headless servers) |
| **NixOS** | Import `lilbee.nixosModules.lilbee`, set `services.lilbee.enable = true;` |

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

Two opt-in features that work but are still finding their final shape. Generation quality and retrieval behavior depend on your library, models, and knobs; expect to iterate. Feedback is welcome.

### Wiki

lilbee analyzes the documents you've indexed and writes a wiki about them. Pages compound across sources: concepts and entities that show up repeatedly get their own page with citations from every source that mentions them. Sections are citation-verified before publish, and plain-text concept references are rewritten to `[[wiki link]]` form so graph-style markdown viewers can render the connections. Lower-confidence pages land in a `drafts/` queue for review rather than publishing direct.

See the [Wiki section of the usage guide](docs/usage.md#wiki) for the full command list and configuration.

### Semantic chunking

A semantic-chunking mode is available as an opt-in alternative to the default fixed-size chunker. It uses embedding similarity to find topic boundaries, so each chunk is one coherent thought instead of a fragment that cuts through an argument. The benefit shows up on prose-heavy collections like novels, essays, long-form research papers, or interview transcripts. The trade-off is roughly 9x more embedding calls during indexing.

See the [Semantic chunking section of the usage guide](docs/usage.md#semantic-chunking) for trade-offs and how to enable it.

## Built on

lilbee stands on a stack of established open-source projects, all bundled into one install:

- [llama.cpp] (via [llama-cpp-python]) is the local model runtime. Every chat, embedding, vision, and reranker call goes through it. Without llama.cpp there is no lilbee.
- [Hugging Face Hub] (via [huggingface_hub]) hosts the model catalog and handles every download. Search, browse, and pull all route through it.
- [Kreuzberg] parses 90+ document formats with heading-aware chunking.
- [LanceDB] is the embedded vector store.
- [tree-sitter] (via [tree-sitter-language-pack]) chunks code across 150+ languages.
- [crawl4ai] and [Playwright] crawl the web; [Tesseract] is the OCR fallback when no vision model is set.
- [LiteLLM] bridges cloud model providers (the `[litellm]` optional extra).
- [Textual] draws the terminal; [Litestar] runs the HTTP server.
- [MCP Python SDK] is the agent surface; [Typer] is the CLI; [Pydantic] is the config + validation backbone.

## License

Elastic License 2.0 (ELv2). See [LICENSE](LICENSE).

[Kreuzberg]: https://github.com/kreuzberg-dev/kreuzberg
[LanceDB]: https://lancedb.com
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[llama-cpp-python]: https://github.com/abetlen/llama-cpp-python
[Hugging Face Hub]: https://huggingface.co
[huggingface_hub]: https://github.com/huggingface/huggingface_hub
[crawl4ai]: https://github.com/unclecode/crawl4ai
[Playwright]: https://playwright.dev
[Textual]: https://textual.textualize.io
[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
[tree-sitter-language-pack]: https://github.com/Goldziher/tree-sitter-language-pack
[Tesseract]: https://github.com/tesseract-ocr/tesseract
[Litestar]: https://litestar.dev
[LiteLLM]: https://github.com/BerriAI/litellm
[MCP Python SDK]: https://github.com/modelcontextprotocol/python-sdk
[Typer]: https://typer.tiangolo.com
[Pydantic]: https://docs.pydantic.dev
