<p align="center">
  <a href="https://lilbee.sh/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-dark.svg">
      <img alt="lilbee" src="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-light.svg" width="340">
    </picture>
  </a>
</p>

<p align="center"><strong>Run and manage local AI models, and search everything you own with them, all in one program.</strong></p>

<p align="center"><a href="https://lilbee.sh/">Project site</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/tutorial">Tutorial reels</a> &nbsp;·&nbsp; <a href="https://pypi.org/project/lilbee/">PyPI</a> &nbsp;·&nbsp; <a href="https://obsidian.lilbee.sh/">Obsidian plugin</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/api/">REST API</a></p>

<p align="center">
  <a href="https://github.com/tobocop2/lilbee/releases"><img src="https://img.shields.io/github/v/release/tobocop2/lilbee?include_prereleases&label=release&logo=github&logoColor=white" alt="Latest release"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee?include_prereleases&label=PyPI&logo=pypi&logoColor=white" alt="lilbee on PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2C3E50" alt="License: MIT"></a>
  <a href="https://community.obsidian.md/plugins/lilbee"><img src="https://img.shields.io/badge/Obsidian-Community%20plugin-7c3aed?logo=obsidian&logoColor=white" alt="Obsidian community plugin"></a>
  <a href="https://glama.ai/mcp/servers/tobocop2/lilbee"><img src="https://glama.ai/mcp/servers/tobocop2/lilbee/badges/score.svg" alt="Glama MCP server score"></a>
</p>

<p align="center">
  <a href="https://github.com/tobocop2/lilbee/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/tobocop2/lilbee/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://lilbee.sh/coverage/"><img src="https://img.shields.io/badge/coverage-100%25-2EA043" alt="Coverage"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-2A6DB2" alt="Typed"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://pepy.tech/project/lilbee"><img src="https://img.shields.io/pepy/dt/lilbee?label=downloads&logo=python&logoColor=white&color=3776AB" alt="PyPI downloads"></a>
  <a href="https://github.com/tobocop2/lilbee/releases"><img src="https://img.shields.io/github/downloads/tobocop2/lilbee/total?label=release%20downloads&logo=github&logoColor=white&color=2C3E50" alt="GitHub release downloads"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/badge/PyPI-pip%20%7C%20uv-3775A9?logo=pypi&logoColor=white" alt="Install from PyPI"></a>
  <a href="https://github.com/tobocop2/homebrew-lilbee"><img src="https://img.shields.io/badge/Homebrew-tap-FBB040?logo=homebrew&logoColor=white" alt="Homebrew tap"></a>
  <a href="https://aur.archlinux.org/packages/lilbee"><img src="https://img.shields.io/aur/version/lilbee?logo=archlinux&logoColor=white&label=AUR" alt="lilbee on the AUR"></a>
  <a href="https://github.com/tobocop2/lilbee/pkgs/container/lilbee"><img src="https://img.shields.io/badge/Docker-ghcr.io-2496ED?logo=docker&logoColor=white" alt="Docker image on GHCR"></a>
</p>

<p align="center">
  <a href="https://github.com/tobocop2/lilbee#install"><img src="https://img.shields.io/badge/Nix-flake-5277C3?logo=nixos&logoColor=white" alt="Nix flake"></a>
  <a href="https://tobocop2.github.io/flatpak-lilbee/"><img src="https://img.shields.io/badge/Flatpak-repo-4A90D9?logo=flatpak&logoColor=white" alt="Flatpak repo"></a>
  <a href="https://github.com/tobocop2/lilbee/releases/latest"><img src="https://img.shields.io/badge/Snap-sideload-82BEA0?logo=snapcraft&logoColor=white" alt="Snap package"></a>
  <a href="https://github.com/tobocop2/lilbee#install"><img src="https://img.shields.io/badge/Scoop-bucket-555555?logo=windows&logoColor=white" alt="Scoop bucket"></a>
</p>

A batteries-included local search engine you can talk to: it runs the AI models, indexes your files and code, crawls the web, and plugs into your coding agent, so there's nothing else to install or set up. Ask in plain English; every answer cites the file and line.

![ask lilbee "what is lilbee in one sentence?" and get a cited answer drawn from its own README](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/what_is_lilbee.gif)

It's all one program: no separate model server, [vector database](#built-on), or container to stand up. lilbee runs the models and keeps the index itself. Reach it as a terminal app, CLI, Model Context Protocol server, HTTP API, or Python library. Close it and it's gone, or run it as a service to keep it warm. Everything runs on your computer; it uses a cloud model only when you pick one.

Models are no different: lilbee has its own model manager and multi-GPU fleet, built on llama.cpp, so one executable does everything (browse Hugging Face, download a model, give it a role, run it on Metal / Vulkan / CUDA). Battle-tested managers are always supported too. If you already use [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai), point lilbee at your existing setup and skip its native model support if you prefer.

> **Tutorial reel:** every demo on this page (and the extras) as a real video player at [**lilbee.sh/tutorial**](https://lilbee.sh/tutorial).

> ## ⚠️ Beta software
>
> lilbee is in **active beta** development. Every release on PyPI is a pre-release; you must use `--pre` (or uv's `--prerelease=allow`) when installing. Interfaces, command names, and on-disk formats may shift between betas. Feedback, bug reports, and issues are very welcome; that's the whole point of the beta.
>
> Latest pre-release (always): [lilbee on PyPI →](https://pypi.org/project/lilbee/)

---

- [Quick start](#quick-start)
- [Tutorial reel](https://lilbee.sh/tutorial) (long-form videos)
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
- **Connect it to your agent over MCP.** Any MCP-aware coding agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets it can quote. Agents can also _fine-tune lilbee on the fly_ via `lilbee_settings_set`. Drop in the [lilbee-mcp skill](src/lilbee/skills/lilbee_mcp/SKILL.md) and the agent reads the full surface: every tool, every retrieval knob, and when to widen for prose vs narrow for code. See [Agent integration](#agent-integration).

Defaults are sane for chatting with code, documentation, crawled sites, and long PDFs. Every retrieval setting is writable from the TUI Settings screen, the `/set` slash command, MCP `lilbee_settings_set`, or `config.toml`. When answers feel thin or noisy, the usual knobs are `top_k`, `max_distance`, or `diversity_max_per_source`.

CLI, the HTTP API, env vars, and `config.toml` are there for scripting, headless runs, and custom integrations. See the [usage guide](docs/usage.md).

## Highlights

- **Answers cite the source line.** Click a citation, jump to the file at the exact line. When the answer isn't in your library, lilbee says so instead of inventing one.
- **It works, and the demos prove it.** Every GIF and tutorial reel here is recorded live on real hardware, nothing staged. Backed by 100% test coverage, full typing, and CI on macOS, Linux, and Windows.
- **Up and running in one command.** Install, run `lilbee`, and a first-run wizard pulls a model and drops you straight into chat.
- **Reads almost anything you point it at.** Documents, scanned pages, spreadsheets, ebooks, web pages, and source code: [90+ formats and 150+ languages](#supported-formats) in all. Whatever you give it becomes searchable.
- **Splits it into pieces that stand on their own.** [Prose and code are chunked differently](#documents-code-and-scanned-images), so each piece keeps its meaning instead of getting cut mid-thought. A search engine is only as good as the chunks underneath it, and this is where most of the quality lives.
- **A sophisticated [search engine](docs/architecture.md#search-pipeline) on top, built on published research.** It ranks every result by how well it answers you, so the best match comes back first. 50+ knobs to [tune from the Settings screen](docs/usage.md#settings-screen) or hand to your agent, with sane defaults if you'd rather not.
- **It brings and runs the models itself.** Browse Hugging Face, pull a model, give it a role (chat, embedding, vision, reranking); lilbee runs it on Metal, Vulkan, or CUDA. You never point it at a server you set up.
- **Already on Ollama or LM Studio? Keep them.** Managing models for you is the default, but lilbee also works with both, so you never have to switch model managers. Their models show up in the same catalog and role pickers, alongside lilbee's own.
- **Your hardware, put to work.** Your machine can do a lot more than you're using it for. lilbee runs local models on hardware you already own, no cloud account required.
- **Per-project libraries.** Keep one library for everything, or give each project its own.
- **One install, many surfaces.** TUI, CLI, [MCP server](#agent-integration), [REST API](https://lilbee.sh/api/), and Python library. Nothing to stand up.
- **Everything in one file, nothing to operate.** The standalone binary bundles the whole thing (search engine, web crawler, MCP server, HTTP server, terminal UI, Python, and llama.cpp) in 250-365 MB, or 600 MB+ with CUDA. No Docker, no vector database, no model server, nothing to keep running; it loads on demand. Comparable desktop AI apps (often Electron) ship hundreds of MB to several GB and do less.
- **Works with your coding agent.** Connect lilbee to your AI coding assistant and it answers from your actual files and code, with citations, instead of guessing. It can even adjust its own search as it works.

## Why lilbee

A small local model is fun, but limited on its own. Give it properly processed documents and a search engine over them, and it becomes genuinely powerful. Without those, it never gets past novelty.

lilbee does all of it in one install: it runs the models, processes your [documents](#built-on), crawls the web pages you point it at, and searches the lot with a real engine. Use it in the terminal, or connect it to your coding agent so it answers from your files with citations instead of guessing.

> **The long-term goal:** make local AI genuinely useful on hardware you already own, with no token budgets and no provider to depend on; the cloud's there only when you want it. The same engine works two ways. It's an [Encarta 99](https://en.wikipedia.org/wiki/Encarta) you build over your files and saved web pages, that you read and ask questions of. And it's a reference layer for code: point it at your project, dependencies, and API docs, and your coding agent answers from what's there instead of guessing function names. Read it yourself, or have your agent read it for you.

## How lilbee compares

lilbee is built for consumer hardware and for people who don't want to babysit infrastructure. One install gives you a model manager, a search engine over your files, a web crawler, and an MCP server for coding agents (with native [opencode](https://opencode.ai) and [hermes](https://github.com/NousResearch/hermes-agent) support), all in a single executable. The goal is to make local AI something anyone can use to get the most out of the hardware they already own, not a setup reserved for people who enjoy wiring tools together.

It isn't another model server you point an app at. It's a local search engine with the model runner built in, so it sits between two worlds: the desktop runners that get a model chatting on your machine, and [vLLM](https://github.com/vllm-project/vllm), the server you stand up to push one model to a cluster of users. lilbee runs models to do retrieval over your files, and scales that whole stack across every GPU in the machine, from one small file.

<details>
<summary><b>Full comparison table</b></summary>

| | lilbee | [LM Studio](https://lmstudio.ai/) | [Ollama](https://ollama.com/) | [vLLM](https://github.com/vllm-project/vllm) |
|---|---|---|---|---|
| **Primary focus** | search engine over your files | desktop app to run and chat with models | local model runner with a growing ecosystem | high-throughput GPU serving |
| Runs local models | ✓ | ✓ | ✓ | ✓ |
| Search your own files, with citations | ✓ full RAG pipeline | per-session doc attachment (RAG, no citations) | — | — |
| Chat, embedding, vision, rerank as one managed fleet | ✓ | — | — | — |
| Multi-GPU model placement | ✓ VRAM-aware tensor-split | ✓ GPU selection + tensor parallelism (CUDA) | ✓ auto layer-split | ✓ tensor + pipeline parallel |
| Scales the whole stack, not just one model | ✓ per-GPU replicas + load-balancing router | — | — | one model per server |
| Built for many-user throughput at scale | single-user focus | — | limited | ✓ this is its job |
| Web crawler built in | ✓ | — | — | — |
| Long-term memory (opt-in) | ✓ | — | — | — |
| Single-file footprint, GPU build (excludes models) | ✓ ~675 MB, whole stack | desktop app + llama.cpp/MLX runtimes | ~1.3 GB, runner only | multi-GB Python + CUDA stack |
| Interfaces | TUI, CLI, MCP, REST, Python, Obsidian GUI | desktop GUI, lms CLI, Python + TS SDKs, REST API, MCP client | desktop GUI, CLI, REST API, Python/JS libs | API server |
| Use your existing Ollama / LM Studio / cloud as a backend | ✓ | — | — | — |

</details>

Ollama and LM Studio are great at running a model and chatting with it; vLLM is what you reach for to serve one model to many users at maximum throughput. lilbee is the only one of the four built around retrieval, and the only one that scales the whole stack, chat, embedding, vision, and reranking, across every GPU in the machine behind a load-balancing router. On a CUDA box that entire stack is a single ~675 MB binary, smaller than Ollama's ~1.3 GB runner-only build and a fraction of a vLLM and PyTorch environment. Already on Ollama or LM Studio? lilbee runs on top of them. Prefer a GUI to the terminal? The [Obsidian plugin](https://obsidian.lilbee.sh/) maps lilbee's model manager and search to a visual interface inside your vault.

## What you can do with it

### A library of your own files

Point lilbee at a folder of PDFs, notes, ebooks, or code and it builds a searchable library, with citations that click back to the source line. The pattern works for anything you have a lot of text about: a shelf of appliance manuals, a field's research papers, a car's service manuals, your company's internal wiki. Whatever you give it becomes searchable, and you can talk to it.

![/add a PDF, watch the Task Center, ask a cited question](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

![chat with an indexed PDF manual: a cited markdown table extracted from the source](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

### Already using an MCP-aware agent? Hand setup to it.

If you've already got an MCP-aware coding agent running, it can do the setup: browse the catalog, pull picks, assign them to the embedding / reranker / vision roles, and tune retrieval. No TUI, no config file, no restart. Agents already understand search engines, so the right knobs are obvious to them. See the [`lilbee-mcp` skill](src/lilbee/skills/lilbee_mcp/SKILL.md) for the workflow and example prompts.

### Coding agents: opencode and hermes

lilbee launches straight into [opencode](https://opencode.ai) and [hermes](https://github.com/NousResearch/hermes-agent) with your local fleet wired in, so a coding agent runs entirely on your machine. It calls lilbee's tools to search your library, swap models, and tune retrieval, and answers from your own files with file:line citations. Tool-calling works across many GGUF families; [docs/agent-models.md](docs/agent-models.md) lists the verified models and how the QA harness measures them.

![opencode running a local model against lilbee's tools](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/agent-launcher-opencode.gif)

![hermes running a local model against lilbee's tools](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/agent-launcher-hermes.gif)

In this demo a small local model is told: when its first search comes back thin, widen lilbee's search settings and search again. The second pass returns the full function bodies with file:line citations. A more capable model would do the same from a higher-level prompt like "improve your search results." Read the [lilbee-mcp skill](src/lilbee/skills/lilbee_mcp/SKILL.md) to teach your own model the pattern.

![agent fine-tunes lilbee mid-conversation: outline, then widened retrieval, then source with file:line citations](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code-self-tune.gif)

### A reference for AI agents

Once configured, lilbee plugs into whatever agent you use, over MCP. Feed it your project's docs, your dependency source, your API docs, your design notes; the agent stops making up function names and instead reads the actual code, cites file and line, and says it doesn't know when the answer isn't in your library.

Your files, the search index, and the embeddings stay on your computer. The agent calls `lilbee_search` and gets back cited snippets. The demo below is lilbee talking to lilbee: an agent indexes lilbee's own source, then answers questions about how lilbee works with file:line citations.

![an agent indexes lilbee's own source through lilbee's MCP server, then answers questions about how lilbee works with file:line citations](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code.gif)

### Offline copies of websites

Install the `[crawler]` extra, point lilbee at a docs site, a wiki, or a vendor's API reference, and the pages get fetched, converted to markdown, and added to your library. From then on you can search or chat with that copy of the site offline, even after it changes or goes down.

![/crawl a Wikipedia page, then ask a cited question against it](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.gif)

Or crawl a whole site, not just one page. With recursive crawling on, lilbee follows the links and indexes the lot; watch the page count climb in the Task Center, then ask one question that synthesizes across the whole site.

![crawl a whole site at depth 1, then ask a multipart question that spans it: a cited answer drawn from across the pages, with Qwen3-8B and a reranker](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl-site.gif)

### Documents, code, and scanned images

lilbee splits indexing by what's being read:

- **Prose and structured documents** (PDFs, Office files, ebooks, HTML, 90+ formats) go through [Xberg] with heading-aware chunking, so each chunk keeps its section context.
- **Code** goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to functions, classes, and modules instead of arbitrary line ranges.
- **Scanned PDFs and photos** go through OCR in 100+ languages: Tesseract for plain text (set `LILBEE_OCR_LANGUAGE`, e.g. `eng+deu`), or a local / remote vision model that keeps tables and layout as markdown.

Retrieval returns things that make sense on their own, not fragments cut through an argument or a function signature.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a model, pick a role. Retrieval and generation expose 50+ settings (chunk size, search strictness, reranker depth, and more), editable from the TUI, env vars, or a project-local config file. Sane defaults.

![browse the model catalog, search Hugging Face Hub, pull a model live](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

### Already running Ollama or LM Studio? Keep them.

> **Watch it:** [Ollama as the model manager](https://lilbee.sh/tutorial/#ollama) and [LM Studio as the model manager](https://lilbee.sh/tutorial/#lm-studio). Point lilbee at a running manager, index a PDF on camera, and get a cited answer back.

**lilbee works with [Ollama](https://ollama.com/) and [LM Studio](https://lmstudio.ai/).** Finding and running models for you is the default and the simplest path: lilbee pulls them, runs them on Metal / Vulkan / CUDA, and you never stand up a server. But you don't have to adopt a new model manager to use lilbee.

If your models already live in Ollama or LM Studio, point lilbee at the running endpoint and they appear in the same catalog and role pickers (chat, embedding, vision, rerank), labeled by where they run, alongside lilbee's own and any cloud models. They're read-only: lilbee lists and runs them but never pulls or deletes them, so their lifecycle stays in the app you already use. Mix freely.

On a `pip` or `uv` install, talking to Ollama or LM Studio needs the `[litellm]` extra (`pip install --pre 'lilbee[litellm]'`); the Homebrew, AUR, Nix, Docker, Flatpak, and Snap builds already include it. See [Install](#install).

### See when a model won't load before you download it

Hugging Face has thousands of GGUFs, but the bundled llama.cpp only supports a subset of architectures and brand-new ones take time to reach the pinned runtime. lilbee tags incompatible models in the catalog and refuses the download (with an override confirm), so you don't wait through a multi-GB pull only to hit "unsupported architecture" at load.

![search HF Hub for deepseek-v4, see the unsupported pill in grid and list view](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-unsupported.gif)

### Cloud models, when you want them

lilbee runs entirely on your machine by default. Two ways to use a cloud model when you want one:

- **Bring your own key.** Install the `[litellm]` extra, add an API key, then point any role (chat, embedding, vision, rerank) at a cloud model from the same catalog. The TUI shows a warning the whole time a cloud model is on.
- **Pair lilbee with a cloud agent over MCP.** Your files, the embeddings, and the index stay local. Any MCP-aware agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets.

Either way, your files and the index stay on your computer. Only what you ask and the snippets needed to answer it get sent to the cloud model.

## Run a model bigger than one card

When a chat model won't fit on a single GPU, lilbee spreads it across the ones you have. It sizes each role's memory with gguf-parser, keeps headroom on every card, and tensor-splits the chat model across the fewest GPUs that fit, with the embedder, reranker, and vision models placed alongside it behind a load-balancing router. This is automatic: ask a question and the model loads split across your cards, answering from your own indexed source.

![a model too big for one card auto-split across the GPUs, answering from lilbee's own indexed source](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-multi-gpu-self-index.gif)

You can also place it by hand. The placement editor pins each role to the cards you choose, previews the fit before anything loads, and applies it live. Ask for a layout that can't fit and it tells you the exact shortfall instead of failing at load time.

![the placement editor: a too-small layout refused with the exact shortfall, then spread across the cards and applied](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-manual-placement.gif)

## TUI

`lilbee` (no args) launches a full Textual terminal app: streaming chat with clickable citations, a model bar with searchable pickers and a Search/Chat toggle, a Task Center for background jobs, and screens for the model catalog, settings, the setup wizard, and the auto-built wiki. Type `/` for the command list; tab completion works everywhere.

![sweep through every TUI screen](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif)

`Ctrl+P` opens the Textual command palette, `?` toggles the keybinding cheat sheet, `/help` opens the slash-command catalog. Every action lilbee can take is reachable from one of those three.

![command palette, keybinding cheat sheet, slash-command catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.gif)

Every GIF on this page (plus the extras that don't fit here) is at [**lilbee.sh/tutorial**](https://lilbee.sh/tutorial) as an embedded video with long-form captions. Tape sources are in [`demos/`](demos). For commands and settings, see the [usage guide](docs/usage.md).

## Hardware requirements

Standalone mode runs entirely on your machine. No cloud required. **Minimum:** Apple Silicon Mac, or a 64-bit Intel/AMD CPU from 2013+ (older CPUs: [On older CPUs](#on-older-cpus-pre-avx2)), or an ARMv8 Linux box; 8 GB RAM, 2 GB disk.

<details>
<summary>Full platform and resource breakdown</summary>

| Platform           | Minimum                                                                                                    | Recommended                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **macOS arm64**    | Apple Silicon (M1 or newer), macOS 11+                                                                     | M-series Pro / Max / Ultra                              |
| **Linux x86_64**   | 64-bit Intel/AMD from 2013+ ([`x86-64-v3`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels)) | Modern Intel/AMD CPU + an NVIDIA, AMD, or Intel Arc GPU |
| **Windows x86_64** | 64-bit Intel/AMD from 2013+ (`x86-64-v3`), Windows 10/11                                                   | Modern desktop / workstation CPU + GPU                  |
| **Linux ARM64**    | ARMv8 NEON-capable (Raspberry Pi 4+, AWS Graviton, Ampere Altra)                                           | Modern ARM server with 16+ GB RAM                       |

| Resource              | Minimum                        | Recommended                                                                                                                                               |
| --------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RAM**               | 8 GB                           | 16 to 32 GB to keep several local models warm at once (chat + embed + rerank + vision); actual footprint scales with the sizes and quantizations you pick |
| **GPU / Accelerator** | none required (CPU-only works) | Apple Silicon (Metal) · NVIDIA / AMD / Intel Arc (Vulkan) · NVIDIA + CUDA toolkit (opt-in CUDA wheels, see [Install](#install))                           |
| **Disk**              | 2 GB                           | 10+ GB for multiple models                                                                                                                                |

</details>

## Install

**Two routes, and the difference matters:**

- **Into your own Python** with `pip` or `uv` (Python 3.11 to 3.14). Uses the Python and tooling you already have, picks the fastest CPU code path for your machine at runtime, and upgrades like any other package. Recommended if you have Python.
- **A self-contained bundle**: the standalone binary, or the Homebrew / AUR / Nix / Docker / Flatpak / Snap builds that wrap it. Nothing else to install. The trade-off is a single large download (it bundles its own Python runtime, `llama.cpp`, and the optional extras) and a small cold-start cost the first time it self-extracts. Recommended if you'd rather not deal with Python.

Have an NVIDIA GPU? Both routes have a CUDA build that's faster than the default Vulkan path. Skip to [On NVIDIA hardware](#on-nvidia-hardware).

No external services either way; lilbee downloads and runs models locally. Optional, for scanned-PDF / image OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) or a [GGUF vision model](docs/usage.md#vision-models).

| How                   | Command                                                                                  | Notes                                                                                                                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pip**               | `pip install --pre lilbee`                                                               | Recommended. The default wheel runs on any x86_64 CPU with AVX2 (2013+; older CPUs: [On older CPUs](#on-older-cpus-pre-avx2)) and uses your GPU via Vulkan / Metal automatically. Intel Mac: add `--extra-index-url https://lilbee.sh/cpu/` ([browse wheels](https://lilbee.sh/cpu/lilbee/)). |
| **uv**                | `uv tool install --prerelease=allow lilbee`                                              | Same wheel as pip; fetches a Python for you if you need one.                                                                                                                                                          |
| **Homebrew**          | `brew tap tobocop2/lilbee && brew install lilbee`                                        | macOS arm64 / Linux x86_64. Bundled build; clears the macOS quarantine flag for you.                                                                                                                                  |
| **AUR**               | `paru -S lilbee`                                                                         | Arch Linux. Wraps the Linux x86_64 binary; works with `yay` / `pacaur` / any helper.                                                                                                                                  |
| **Docker**            | `docker run --rm -v lilbee-data:/home/lilbee/data ghcr.io/tobocop2/lilbee:latest --help` | GHCR image, tagged by version and `latest`. Data lives at `/home/lilbee/data`. Mount a volume there.                                                                                                                  |
| **Nix**               | `nix run github:tobocop2/lilbee`                                                         | NixOS, nix-darwin, or any host with nix. On Linux the flake bundles `glibc`, `libgomp`, and `vulkan-loader` so it runs on bare NixOS.                                                                                 |
| **Flatpak**           | `flatpak remote-add --if-not-exists lilbee https://tobocop2.github.io/flatpak-lilbee/lilbee.flatpakrepo && flatpak install lilbee io.github.tobocop2.lilbee` | Linux x86_64, any distro with flatpak. Needs the [Flathub remote](https://flathub.org/setup) for the runtime. Run with `flatpak run io.github.tobocop2.lilbee` (worth an alias); `flatpak update` picks up new releases. Data lives under `~/.var/app/io.github.tobocop2.lilbee/`. |
| **Snap**              | `curl -LO https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-linux-x86_64.snap && sudo snap install ./lilbee-linux-x86_64.snap --dangerous --classic` | Linux x86_64. Sideloaded, so snapd flags it `--dangerous` (it just means unsigned) and it won't auto-update; rerun the same command to upgrade. |
| **Scoop**             | `scoop bucket add lilbee https://github.com/tobocop2/lilbee && scoop install lilbee` | Windows x86_64. Installs the CUDA build on machines with a recent NVIDIA driver, otherwise the CPU build. `scoop update lilbee` upgrades. |
| **Standalone binary** | [download for your platform &rarr;](https://github.com/tobocop2/lilbee/releases/latest)  | One file, own Python runtime, no `pip` needed. Linux needs glibc 2.28+; the macOS / Windows builds are unsigned (`xattr -d com.apple.quarantine ./lilbee-macos-arm64` if Gatekeeper blocks it).                       |
| **From source**       | `git clone https://github.com/tobocop2/lilbee && cd lilbee && uv sync && uv run lilbee`  | For hacking on it. Needs `git` and `uv`.                                                                                                                                                                              |

### On NVIDIA hardware

The default Vulkan build works on NVIDIA cards, but there's a dedicated CUDA build that's faster on NVIDIA hardware and sidesteps the iGPU + dGPU Vulkan-loader crash on Windows.

<details>
<summary>CUDA install commands</summary>

|              | Command                                                                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pip**      | `pip install --pre lilbee --extra-index-url https://lilbee.sh/cu125/`                                                                                                        |
| **uv**       | `uv tool install --prerelease=allow lilbee --extra-index-url https://lilbee.sh/cu125/`                                                                                       |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-cuda`                                                                                                                                   |
| **AUR**      | `paru -S lilbee-cuda`                                                                                                                                                        |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-cuda`                                                                                                                                 |
| **Scoop**    | `scoop install lilbee` (auto-picks the CUDA build when an NVIDIA driver is present)                                                                                          |
| **Binary**   | [`lilbee-linux-x86_64-cu125`](https://github.com/tobocop2/lilbee/releases/latest) or [`lilbee-windows-x86_64-cu125.exe`](https://github.com/tobocop2/lilbee/releases/latest) |

Same `lilbee` command after install. The CUDA runtime is bundled; you only need the NVIDIA driver. Already have the regular `lilbee` installed? On AUR `paru -S lilbee-cuda` swaps it automatically; on Homebrew run `brew uninstall lilbee` first. Older driver? `cu124` and `cu121` ship via the matching wheel indexes and as direct-download Linux binaries on the release page.

</details>

Then check it runs and pick a model:

```bash
lilbee self-check    # ~90 MB download; runs an inference + an embedding; "SELF-CHECK PASSED" on success
lilbee               # launch the terminal app; pick a chat + embedding model on the welcome screen
```

The [usage guide](docs/usage.md) covers the rest: TUI screens, slash commands, CLI, HTTP server, MCP, env vars, and `config.toml`.

### On older CPUs (pre-AVX2)

Pre-2013 Intel or pre-Zen AMD CPUs lack [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), so the normal build crashes on launch. The `lilbee-compat` build runs on any x86-64 chip back to ~2008.

<details>
<summary>Install commands for every channel, and why it's needed</summary>

|              | Command                                                                                                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pip**      | `pip install --pre lilbee 'lancedb==0.33.0+compat' --extra-index-url https://lilbee.sh/compat/`                                                                          |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-compat`                                                                                                                                            |
| **AUR**      | `paru -S lilbee-compat`                                                                                                                                                                 |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-compat`                                                                                                                                          |
| **Scoop**    | `scoop install lilbee-compat`                                                                                                                                                           |
| **Flatpak**  | `flatpak install lilbee io.github.tobocop2.lilbee.compat`                                                                                                                               |
| **Snap**     | `curl -LO https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-compat-linux-x86_64.snap && sudo snap install ./lilbee-compat-linux-x86_64.snap --dangerous --classic`    |
| **Binary**   | [`lilbee-compat-linux-x86_64`](https://github.com/tobocop2/lilbee/releases/latest), [`lilbee-compat-windows-x86_64.exe`](https://github.com/tobocop2/lilbee/releases/latest), or [`lilbee-compat-macos-x86_64`](https://github.com/tobocop2/lilbee/releases/latest) (pre-AVX2 Intel Macs, e.g. the 2013 Mac Pro) |

Same `lilbee` command after install. The crash is from [lancedb](https://lancedb.github.io/lancedb/)'s AVX2-compiled wheels; this build swaps in a [lancedb fork](https://github.com/tobocop2/lance) that picks instructions at runtime. A 👍 or comment on the upstream [lance PR](https://github.com/lance-format/lance/pull/6630) helps it land.

</details>

### Linux runtime requirements

The Linux x86_64 wheel and binary link the Vulkan loader at runtime. Most desktop distros (Ubuntu 22.04+, Pop!\_OS, Mint) ship `libvulkan1`; bare Arch / Fedora / Alpine images don't, and `lilbee self-check` fails with `cannot open shared object file: libvulkan.so.1`. Install it once: `sudo pacman -S vulkan-icd-loader` (Arch / Manjaro), `sudo dnf install vulkan-loader` (Fedora, RHEL), or `sudo apt-get install libvulkan1` (Debian, Ubuntu).

### Optional extras

These only matter for a `pip` or `uv` install: add the name in brackets, e.g. `pip install --pre 'lilbee[crawler,litellm]'` (combine multiple, and `--extra-index-url` still works for CUDA). The standalone binary and the Homebrew / AUR / Nix / Docker / Flatpak / Snap builds already include all three. lilbee works without them either way.

| Extra       | What it adds                                                                                                                                              |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[crawler]` | Index websites alongside your files: crawl a docs site or wiki to markdown, then search it offline.                                                       |
| `[litellm]` | Bridge to hosted model providers for chat, vision, or embeddings while other roles stay local. The TUI flags when a hosted role is active.                |
| `[graph]`   | Concept-graph search: extracts the ideas in your documents and uses how they relate to surface matches plain keyword search misses. No extra model calls. |

See the [full guide on optional extras](docs/usage.md#optional-extras) for configuration.

### Upgrading

```bash
pip install --upgrade --pre lilbee
# or
uv tool install --reinstall --prerelease=allow lilbee
```

## Agent integration

Drop the [`lilbee-mcp` skill](src/lilbee/skills/lilbee_mcp/SKILL.md) into `.opencode/skills/` or `.claude/skills/`, register lilbee as an MCP server, and any MCP-aware coding agent can search your library, swap models, and tune retrieval. The skill is the single entry point: it documents every tool, the workflows the agent should follow, and points to drop-in `AGENTS.md` and worker-subagent starters under [`examples/agent-integration/`](examples/agent-integration/).

**The demos below use opencode with a cloud model. lilbee stays local; only the queries and the returned chunks go to the cloud model.** Local-model opencode and hermes integration works across many GGUF families: see [Coding agents: opencode and hermes](#coding-agents-opencode-and-hermes) above.

Live-indexing example: opencode (cloud model) indexes a Godot 4 pathfinding subset (~3s), then `lilbee_search`-es for `AStarGrid2D` and answers method-by-method against your _local_ files.

![an MCP-driven coding agent indexes a small local godot subset and answers with cited methods](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot-search.gif)

It scales up. Pre-index Godot 4's full class reference (810 XMLs, 3449 chunks) and the same opencode + cloud setup writes a procedural level generator, every API call backed by a `godot-classes/<Class>.xml:line` citation; the [side-by-side benchmark](docs/benchmarks/godot-level-generator.md) measured 4 hallucinated APIs without lilbee, 0 with.

![cited codegen against the full Godot class reference](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.gif)

## HTTP Server

The HTTP server exposes a REST API any tool or GUI can hit: search (with SSE streaming), document lifecycle, crawling, model management, configuration. See the [REST API reference](https://lilbee.sh/api/) and the [usage guide](docs/usage.md#http-server) for setup.

The [Obsidian plugin](https://obsidian.lilbee.sh/) is a GUI built on it: it starts the HTTP server in the background, and every citation opens a Source Preview scrolled to the exact spot. It is an official [Obsidian community plugin](https://community.obsidian.md/plugins/lilbee): install it from Settings then Community plugins inside Obsidian. The [plugin README](https://github.com/tobocop2/obsidian-lilbee#quick-start) has setup.

### Running as a service (optional)

For tools that talk to lilbee's HTTP REST API (the Obsidian plugin, custom GUIs, anything hitting `/api/*`), your OS launcher can keep the HTTP server warm so requests skip the cold-start.

<details>
<summary>Daemon setup per platform</summary>

This is the only lilbee surface that benefits from a daemon. The TUI, `lilbee chat`, the MCP server, and the rest of the CLI load on demand and exit when you close them. No always-on process to babysit.

Pull a chat and embedding model first; all recipes pin the server to `127.0.0.1:42697`.

| Platform               | Command                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **macOS (Homebrew)**   | `brew services start lilbee`                                                                    |
| **Linux (Arch / AUR)** | `systemctl --user enable --now lilbee` (add `loginctl enable-linger $USER` on headless servers) |
| **NixOS**              | Import `lilbee.nixosModules.lilbee`, set `services.lilbee.enable = true;`                       |

</details>

## Supported formats

Document extraction powered by [Xberg], code chunking by [tree-sitter]. lilbee handles every format Xberg can extract (100+) and tracks its list directly, so support grows as Xberg adds formats. The table below covers the common ones.

<details>
<summary>Format table</summary>

| Format       | Extensions                                                                                                                                              | Requires                                                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PDF          | `.pdf`                                                                                                                                                  | none                                                                                                                                                                                             |
| Scanned PDF  | `.pdf` (no extractable text)                                                                                                                            | [Tesseract](https://github.com/tesseract-ocr/tesseract) (auto, plain text, 100+ languages via `LILBEE_OCR_LANGUAGE`), or a GGUF vision model via the native mtmd backend (recommended, preserves tables, headings, and layout as markdown) |
| Office       | `.docx`, `.xlsx`, `.pptx`, `.doc`, `.xls`, `.ppt`, `.odt`, `.ods`, `.rtf`                                                                               | none                                                                                                                                                                                             |
| eBook        | `.epub`, `.fb2`                                                                                                                                         | none                                                                                                                                                                                             |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`, `.gif`, `.heic`, `.avif`, `.jp2`/`.j2k`/`.jpx` (JPEG-2000)                                           | [Tesseract](https://github.com/tesseract-ocr/tesseract) or a vision model                                                                                                                        |
| Text/markup  | `.md`, `.txt`, `.html`, `.rst`, `.org`, `.tex`, `.typst`                                                                                                | none                                                                                                                                                                                             |
| Data         | `.csv`, `.tsv`, `.json`, `.jsonl`, `.xml`, `.yaml`, `.toml`                                                                                             | none                                                                                                                                                                                             |
| Email        | `.eml`, `.msg`                                                                                                                                          | none                                                                                                                                                                                             |
| Archives     | `.zip`, `.tar`, `.gz`, `.7z`, `.pst` (combined contents extracted)                                                                                      | none                                                                                                                                                                                             |
| Code         | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java` and [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) via tree-sitter (AST-aware chunking) | none                                                                                                                                                                                             |

Plus notebooks, bibliographies, iWork, and audio/video, among others. See the [usage guide](docs/usage.md#ocr) for OCR setup and [model benchmarks](docs/benchmarks/vision-ocr.md).

</details>

## Experimental

<details>
<summary>Two opt-in features that work but are still finding their final shape: <strong>Wiki</strong> and <strong>semantic chunking</strong>. Click to expand.</summary>

<br>

Generation quality and retrieval behavior depend on your library, models, and knobs; expect to iterate. Feedback is welcome.

### Wiki

lilbee analyzes the documents you've indexed and writes a wiki about them. Pages compound across sources: concepts and entities that show up repeatedly get their own page with citations from every source that mentions them. Sections are citation-verified before publish, and plain-text concept references are rewritten to `[[wiki link]]` form so graph-style markdown viewers can render the connections. Lower-confidence pages land in a `drafts/` queue for review rather than publishing direct.

See the [Wiki section of the usage guide](docs/usage.md#wiki) for the full command list and configuration.

### Semantic chunking

A semantic-chunking mode is available as an opt-in alternative to the default fixed-size chunker. It uses embedding similarity to find topic boundaries, so each chunk is one coherent thought instead of a fragment that cuts through an argument. The benefit shows up on prose-heavy collections like novels, essays, long-form research papers, or interview transcripts. The trade-off is roughly 9x more embedding calls during indexing.

See the [Semantic chunking section of the usage guide](docs/usage.md#semantic-chunking) for trade-offs and how to enable it.

</details>

## Built on

lilbee stands on a stack of established open-source projects, all bundled into one install:

- [Xberg] parses 90+ document formats with heading-aware chunking.
- [llama.cpp] is the local model runtime: lilbee bundles its `llama-server` and starts it for you, so every chat, embedding, vision, and reranker call goes through it. [llama-swap] keeps a server per role resident together behind one endpoint, and [gguf-parser] estimates each model's memory footprint so lilbee loads what fits. Without llama.cpp there is no lilbee.
- [Hugging Face Hub] (via [huggingface_hub]) hosts the model catalog and handles every download. Search, browse, and pull all route through it.
- [LanceDB] is the embedded vector store.
- [tree-sitter] (via [tree-sitter-language-pack]) chunks code across 150+ languages.
- [crawl4ai] and [Playwright] crawl the web; [Tesseract] is the OCR fallback when no vision model is set.
- [LiteLLM] bridges cloud model providers (the `[litellm]` optional extra).
- [Textual] draws the terminal; [Litestar] runs the HTTP server.
- [MCP Python SDK] is the agent surface; [Typer] is the CLI; [Pydantic] is the config + validation backbone.
- [Nuitka] compiles the whole thing into the standalone single-file binary, bundling its own Python runtime so there is nothing to install and nothing to compile.

## Support

Having trouble? See [TROUBLESHOOTING.md](https://github.com/tobocop2/lilbee/blob/main/TROUBLESHOOTING.md) for log locations and common failures.

lilbee is built and maintained by one person. If it is useful to you, you can chip in via [PayPal](https://paypal.me/lilbeedotsh). Bug reports and pull requests help just as much.

## License

MIT. See [LICENSE](LICENSE).

[Xberg]: https://github.com/xberg-io/xberg
[LanceDB]: https://lancedb.com
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[llama-swap]: https://github.com/mostlygeek/llama-swap
[gguf-parser]: https://github.com/gpustack/gguf-parser-go
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
[Nuitka]: https://nuitka.net
