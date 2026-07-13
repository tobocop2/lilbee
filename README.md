<p align="center">
  <a href="https://lilbee.sh/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-dark.svg">
      <img alt="lilbee" src="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-light.svg" width="340">
    </picture>
  </a>
</p>

<p align="center"><strong>The whole local AI stack in one executable: it runs and manages the models, and searches everything you own with them.</strong></p>

<p align="center"><a href="https://lilbee.sh/">Project site</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/tutorial">Tutorial reels</a> &nbsp;·&nbsp; <a href="https://pypi.org/project/lilbee/">PyPI</a> &nbsp;·&nbsp; <a href="https://obsidian.lilbee.sh/">Obsidian plugin</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/api/">REST API</a> &nbsp;·&nbsp; <a href="https://web.libera.chat/#lilbee">Chat (#lilbee)</a></p>

<p align="center">
  <a href="https://github.com/tobocop2/lilbee/releases/latest"><img src="https://img.shields.io/github/v/release/tobocop2/lilbee?label=release&logo=github&logoColor=white" alt="Latest release"></a>
  <a href="https://pypi.org/project/lilbee/"><img src="https://img.shields.io/pypi/v/lilbee?include_prereleases&label=PyPI&logo=pypi&logoColor=white" alt="lilbee on PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2C3E50" alt="License: MIT"></a>
  <a href="https://community.obsidian.md/plugins/lilbee"><img src="https://img.shields.io/badge/Obsidian-Community%20plugin-7c3aed?logo=obsidian&logoColor=white" alt="Obsidian community plugin"></a>
  <a href="https://glama.ai/mcp/servers/tobocop2/lilbee"><img src="https://glama.ai/mcp/servers/tobocop2/lilbee/badges/score.svg" alt="Glama MCP server score"></a>
  <a href="https://web.libera.chat/#lilbee"><img src="https://img.shields.io/badge/IRC-%23lilbee%20on%20Libera.Chat-5865F2?logo=liberadotchat&logoColor=white" alt="#lilbee on Libera.Chat"></a>
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

lilbee runs and manages your models: chat, embedding, vision, and rerank, placed across every GPU you have. It puts them to work as a search engine you can talk to, over your files, notes, code, and the web, where every answer cites the exact file and line. It crawls websites into your library, [launches your coding agents on local models](#launch-your-coding-agent-on-local-models), and hands any [MCP-aware agent](#a-reference-for-ai-agents) cited answers from everything you've indexed. The same engine backs the [Obsidian community plugin](https://obsidian.lilbee.sh/), so your vault gets all of it without a terminal. Ask in plain English. No containers, no networking, nothing else to install or set up.

![ask lilbee "what is lilbee in one sentence?" and get a cited answer drawn from its own README](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/what_is_lilbee.gif)

It's all one program: no separate model server, [vector database](#built-on), or container to stand up. lilbee runs the models and keeps the index itself. Reach it as a terminal app, CLI, Model Context Protocol server, HTTP API, or Python library. Close it and it's gone, or run it as a service to keep it warm. Everything runs on your computer; it uses a cloud model only when you pick one.

Models are no different: lilbee has its own model manager and multi-GPU fleet, built on llama.cpp, so one executable does everything (browse Hugging Face, download a model, give it a role, run it on Metal / Vulkan / CUDA). You don't need [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) at all: the [model families lilbee runs](#tested-model-families) are the architectures behind most of the 190,000+ GGUF repos on Hugging Face, verified per family on real GPUs. If you already use them, point lilbee at your existing setup and keep your models.

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
- [First start](#first-start)
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
- **A model too big for one card runs across all of them.** lilbee sizes each role with gguf-parser and tensor-splits your chat model across the fewest GPUs that fit, placing the embedder, reranker, and vision models alongside it behind a load-balancing router. It happens automatically, or you can pin each role to the cards you choose. [Run a model bigger than one card](#run-a-model-bigger-than-one-card).
- **Already on Ollama or LM Studio? Keep them.** You don't need either one: lilbee's own manager handles everything, across the same [model families](#tested-model-families) they run. But lilbee also works with both, so you never have to switch. Their models show up in the same catalog and role pickers, alongside lilbee's own.
- **Your hardware, put to work.** Your machine can do a lot more than you're using it for. lilbee runs local models on hardware you already own, no cloud account required.
- **Per-project libraries.** Keep one library for everything, or give each project its own.
- **One install, many surfaces.** TUI, CLI, [MCP server](#agent-integration), [REST API](https://lilbee.sh/api/), and Python library. Nothing to stand up.
- **Everything in one file, nothing to operate.** The standalone binary bundles the whole thing (search engine, web crawler, MCP server, HTTP server, terminal UI, Python, and llama.cpp) in ~290-420 MB, or ~0.6-1.2 GB with CUDA. No Docker, no vector database, no model server, nothing to keep running; it loads on demand. Comparable desktop AI apps (often Electron) ship hundreds of MB to several GB and do less.
- **Works with your coding agent.** Connect lilbee to your AI coding assistant and it answers from your actual files and code, with citations, instead of guessing. It can even adjust its own search as it works.

## Why lilbee

A small local model is fun, but limited on its own. Give it properly processed documents and a search engine over them, and it becomes genuinely powerful. Without those, it never gets past novelty.

lilbee does all of it in one install: it runs the models, processes your [documents](#built-on), crawls the web pages you point it at, and searches the lot with a real engine. The same engine works two ways:

- **An [Encarta 99](https://en.wikipedia.org/wiki/Encarta) over your own files.** Build a library from your documents and saved web pages, then read it and ask questions of it in the terminal.
- **A reference layer for code.** Point it at your project, dependencies, and API docs, and your coding agent answers from what's actually there, with file:line citations, instead of guessing function names.

> **The long-term goal:** make local AI genuinely useful on hardware you already own, with no token budgets and no provider to depend on; the cloud's there only when you want it.

## How lilbee compares

lilbee is built for consumer hardware and for people who don't want to babysit infrastructure. It isn't another model server you point an app at; it's a local search engine with the model runner built in. One install gives you the whole stack in a single executable:

- **A search engine over your files**, with answers that cite the source line, not just a model to chat with.
- **A managed fleet**, chat, embedding, vision, and reranking, spread across every GPU in the machine behind a load-balancing router, not one model loaded at a time.
- **Everything bundled**: model manager, search engine, web crawler, MCP server for coding agents (native [opencode](https://opencode.ai) and [hermes](https://github.com/NousResearch/hermes-agent)), HTTP server, TUI, and Python, in one file.

It sits between two worlds: the desktop runners that get a model chatting on your machine ([Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai)), and [vLLM](https://github.com/vllm-project/vllm), the server you stand up to push one model to a cluster of users. lilbee runs models to do retrieval over your files, and scales that whole stack across every GPU in the machine, from one small file.

<details>
<summary><b>Full comparison table: lilbee vs Ollama, LM Studio, and vLLM. Click to expand.</b></summary>

### Full comparison table

| | lilbee | [LM Studio](https://lmstudio.ai/) | [Ollama](https://ollama.com/) | [vLLM](https://github.com/vllm-project/vllm) |
|---|---|---|---|---|
| **Primary focus** | local search, chat, and serving across your GPUs | desktop app to run and chat with models | local model runner with a growing ecosystem | high-throughput GPU serving |
| Runs local models | ✓ | ✓ | ✓ | ✓ |
| Search your own files, with citations | ✓ [full RAG pipeline](docs/architecture.md#search-pipeline), inline per-line citations | per-session doc attachment ([RAG, document-level citation](https://lmstudio.ai/docs/app/basics/rag)) | — | — |
| Chat, embedding, vision, rerank as one managed fleet | ✓ [all four, coordinated](docs/architecture.md#local-inference-engine) | chat, embed, vision (no rerank), [loaded individually](https://lmstudio.ai/docs/developer/core/server) | chat, embed, vision (no rerank), [loaded individually](https://docs.ollama.com/faq) | each supported, but [one model per server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |
| Multi-GPU model placement | ✓ [VRAM-aware tensor-split](docs/architecture.md#local-inference-engine) | ✓ GPU selection + [tensor parallelism (CUDA)](https://lmstudio.ai/changelog/lmstudio-v0.4.15) | ✓ [auto multi-GPU offload](https://docs.ollama.com/faq) | ✓ [tensor + pipeline parallel](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) |
| Scales the whole stack, not just one model | ✓ per-GPU replicas + [load-balancing router](docs/architecture.md#local-inference-engine) | — | — | one model per server |
| Built for many-user throughput at scale | ✓ [a data-parallel replica per GPU, requests load-balanced](docs/architecture.md#local-inference-engine) | — | [limited](https://docs.ollama.com/faq) | ✓ [this is its job](https://github.com/vllm-project/vllm) |
| Web crawler built in | ✓ [built in](#offline-copies-of-websites) | — | — | — |
| Long-term memory (opt-in) | ✓ [opt-in](docs/usage.md#memory) | — | — | — |
| Interfaces | [TUI, CLI, MCP, REST, Python](docs/architecture.md#interfaces), Obsidian GUI | [desktop GUI, lms CLI, Python + TS SDKs, REST API, MCP client](https://lmstudio.ai/docs) | [desktop GUI, CLI, REST API, Python/JS libs](https://docs.ollama.com/) | [API server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |
| Use your existing Ollama / LM Studio / cloud as a backend | ✓ [how](#already-running-ollama-or-lm-studio-keep-them) | — | — | — |

</details>

Of the four, lilbee is the only one built around retrieval, and the only one that scales the whole stack, chat, embedding, vision, and reranking, across every GPU in the machine behind a load-balancing router.

<details>
<summary><b>Install size by platform: one file that undercuts the others while doing more. Click to expand.</b></summary>

### Install size (single-file download, models excluded)

Download sizes in decimal GB/MB (bytes ÷ 1000), measured from each project's own release artifacts, linked.

| | macOS | Windows | Linux | What you get |
|---|---|---|---|---|
| **[lilbee](https://github.com/tobocop2/lilbee/releases)** (Metal / Vulkan, default) | 286 MB | 303 MB | 422 MB | the whole stack: search engine, crawler, servers, TUI, model runner, fleet manager |
| **[lilbee](https://github.com/tobocop2/lilbee/releases)** (CUDA, opt-in for NVIDIA) | n/a | 633 MB | 1.20 GB | the same whole stack, with the faster CUDA runtime |
| [Ollama](https://github.com/ollama/ollama/releases) | 164 MB | 1.43 GB (CUDA bundled) | 1.44 GB (CUDA bundled) | a model runner, fetches its runtimes separately |
| [LM Studio](https://lmstudio.ai/download) | 569 MB | 617 MB | 1.10 GB | a desktop app (Electron) |
| [vLLM](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) | n/a | n/a | multi-GB | a Python + CUDA serving engine |

Even lilbee's CUDA build stays under Ollama's, and it's the whole stack, not just a model runner.

</details>

Already on Ollama or LM Studio? lilbee runs on top of them. Prefer a GUI to the terminal? The [Obsidian plugin](https://obsidian.lilbee.sh/) maps lilbee's model manager and search to a visual interface inside your vault.

## What you can do with it

### A library of your own files

Point lilbee at a folder of PDFs, notes, ebooks, or code and it builds a searchable library, with citations that click back to the source line. The pattern works for anything you have a lot of text about: a shelf of appliance manuals, a field's research papers, a car's service manuals, your company's internal wiki. Whatever you give it becomes searchable, and you can talk to it.

![/add a PDF, watch the Task Center, ask a cited question](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

![chat with an indexed PDF manual: a cited markdown table extracted from the source](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

### Already using an MCP-aware agent? Hand setup to it.

If you've already got an MCP-aware coding agent running, it can do the setup: browse the catalog, pull picks, assign them to the embedding / reranker / vision roles, and tune retrieval. No TUI, no config file, no restart. Agents already understand search engines, so the right knobs are obvious to them. See the [`lilbee-mcp` skill](src/lilbee/skills/lilbee_mcp/SKILL.md) for the workflow and example prompts.

### Launch your coding agent on local models

`lilbee launch opencode` and `lilbee launch hermes` set up lilbee's local models in your agent in one command. lilbee registers itself as a provider and an MCP server in the agent's own config, leaves your existing setup intact, warms a model, and opens the agent pointed at it. No API keys, no provider setup, and nothing leaves your machine. Tool-calling works across many GGUF families; [docs/agent-models.md](docs/agent-models.md) has the verified list and how the QA harness measures it.

These reels show each agent, launched on a local model, doing real work on lilbee's own source. opencode adds a `lilbee launch status` subcommand, runs it, and writes a test that passes; hermes does the same with `lilbee launch list`.

![opencode, launched on a local lilbee model, adds a launch-status subcommand and lands a passing test](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/agent-launcher-opencode.gif)

![hermes, launched on a local lilbee model, adds a launch-list subcommand and lands a passing test](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/agent-launcher-hermes.gif)

It tunes itself, too. Give a small local model a thin first result and tell it to widen lilbee's search, and the second pass returns full function bodies with file:line citations. A more capable model would do the same from a higher-level prompt like "improve your search results." Read the [lilbee-mcp skill](docs/agent-skills/lilbee-mcp/SKILL.md) to teach your own model the pattern.

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

- **Prose and structured documents** (PDFs, Office files, ebooks, HTML, 90+ formats) go through [Kreuzberg] with heading-aware chunking, so each chunk keeps its section context.
- **Code** goes through [tree-sitter]'s AST-aware splitter across [150+ languages](https://github.com/Goldziher/tree-sitter-language-pack), so chunks map to functions, classes, and modules instead of arbitrary line ranges.
- **Scanned PDFs and photos** go through OCR: Tesseract for plain text, or a local / remote vision model that keeps tables and layout as markdown.

Retrieval returns things that make sense on their own, not fragments cut through an argument or a function signature.

### Pick and tune your models

Chat, embedding, vision, and reranking models are installed and switched from inside the terminal: browse the catalog, pull a model, pick a role. Retrieval and generation expose 50+ settings (chunk size, search strictness, reranker depth, and more), editable from the TUI, env vars, or a project-local config file. Sane defaults.

### Tested model families

One representative per architecture family, pulled with `lilbee model pull` and run through the full pipeline (index, search, answer; OCR for vision) on consumer hardware. [docs/tested-models.md](docs/tested-models.md) has the details and method. Between them, these families are the architectures behind most of the 190,000+ GGUF model repos on Hugging Face: if a model's family is listed, its variants and quants are expected to work.

<details>
<summary><b>The family tables, per role. Click to expand.</b></summary>

**Vision** (all on a single 12 GB card, projector fetched by the pull itself):

| Family | Projector type |
|--------|----------------|
| LightOnOCR | lightonocr |
| Qwen2.5-VL | qwen2.5vl merger |
| Qwen3-VL | qwen3vl |
| Gemma 3 | gemma3 |
| SmolVLM2 | idefics3 |
| MiniCPM-V | resampler |
| InternVL3 | internvl |
| LLaVA 1.6 | mlp |
| Gemma 4 | mixed vision+audio |
| dots.ocr | dots.ocr |

**Chat** (one per memory-architecture class):

| Class | Representative |
|-------|----------------|
| Dense GQA | Llama 3.2 |
| Dense | Qwen3 |
| Sliding-window attention | Gemma 3 |
| Multi-head latent attention | DeepSeek V2 Lite |
| Mixture of experts | OLMoE |
| Hybrid SSM | LFM2 |

**Embedding:**

| Class | Representative |
|-------|----------------|
| BERT encoder | all-MiniLM-L6-v2 |
| nomic-bert | nomic-embed-text v1.5 |
| Decoder-pooled | Qwen3-Embedding 0.6B |
| Decoder-pooled, large | Qwen3-Embedding 8B |
| XLM-RoBERTa | bge-m3 |

**Rerank:**

| Class | Representative |
|-------|----------------|
| Cross-encoder | bge-reranker-v2-m3 |
| LLM reranker | Qwen3-Reranker 0.6B |

</details>


![browse the model catalog, search Hugging Face Hub, pull a model live](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

### Already running Ollama or LM Studio? Keep them.

> **Watch it:** [Ollama as the model manager](https://lilbee.sh/tutorial/#ollama) and [LM Studio as the model manager](https://lilbee.sh/tutorial/#lm-studio). Point lilbee at a running manager, index a PDF on camera, and get a cited answer back.

**lilbee works with [Ollama](https://ollama.com/) and [LM Studio](https://lmstudio.ai/).** Finding and running models for you is the default and simplest path (lilbee pulls them and runs them on Metal / Vulkan / CUDA, no server to stand up), but you don't have to adopt a new model manager to use lilbee.

- **Point it at a running manager.** Your models in Ollama or LM Studio show up in the same catalog and role pickers (chat, embedding, vision, rerank), labeled by where they run, alongside lilbee's own and any cloud models. Mix freely.
- **They stay read-only.** lilbee lists and runs them but never pulls or deletes them, so their lifecycle stays in the app you already use.
- **On `pip` / `uv`,** this needs the `[litellm]` extra (`pip install --pre 'lilbee[litellm]'`); the Homebrew, AUR, Nix, Docker, Flatpak, and Snap builds already include it. See [Install](#install).

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

`Ctrl+P` opens the Textual command palette, `?` on an empty prompt (or `F1` anywhere) toggles the keybinding cheat sheet, `/help` opens the slash-command catalog. Every action lilbee can take is reachable from one of those three.

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
| **GPU / Accelerator** | none required (CPU-only works) | Apple Silicon (Metal) · NVIDIA / AMD / Intel Arc (Vulkan) · NVIDIA driver (opt-in CUDA wheels, runtime bundled, see [Install](#install))                           |
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
| **Binary**   | [`lilbee-compat-linux-x86_64`](https://github.com/tobocop2/lilbee/releases/latest) or [`lilbee-compat-windows-x86_64.exe`](https://github.com/tobocop2/lilbee/releases/latest)         |

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

## First start

The very first launch does one-time work: the executable unpacks itself behind
a progress bar, and the model loads before your first answer, shown live inside
the answer bubble. Every launch after that opens straight to chat in a couple
of seconds.

<table>
  <tr>
    <th align="center">Very first launch</th>
    <th align="center">Every launch after</th>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/first-start.gif" alt="First launch: a one-time unpack bar, chat opens, the first answer shows the engine loading in its bubble, then streams" width="380"></td>
    <td><img src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/later-start.gif" alt="Every later launch: chat opens in about two seconds and the answer follows" width="380"></td>
  </tr>
</table>

Measured with a small chat model (Qwen3 0.6B):

| | Very first launch | Every launch after |
|---|---|---|
| Chat on screen | 15 to 20s, one-time unpack | 2 to 3s |
| First answer of the session | 10 to 20s, the engine load plays in the bubble | the same, or instant with **Keep engine warm** |
| Answers after that | model speed | model speed |

Bigger models load longer; the bubble shows real progress while weights are read.
How the engine lifecycle works, and how to keep it warm so relaunches skip the
load entirely, lives in the [usage guide](docs/usage.md#the-engine-lifecycle).

## Agent integration

Drop the [`lilbee-mcp` skill](src/lilbee/skills/lilbee_mcp/SKILL.md) into `.opencode/skills/` or `.claude/skills/`, register lilbee as an MCP server, and any MCP-aware coding agent can search your library, swap models, and tune retrieval. The skill is the single entry point: it documents every tool, the workflows the agent should follow, and points to drop-in `AGENTS.md` and worker-subagent starters under [`examples/agent-integration/`](examples/agent-integration/).

**The demos below use opencode with a cloud model. lilbee stays local; only the queries and the returned chunks go to the cloud model.** To run the agent itself on a local model instead, see [Launch your coding agent on local models](#launch-your-coding-agent-on-local-models) above.

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

Text extraction powered by [Kreuzberg], code chunking by [tree-sitter]. Structured formats (XML, JSON, CSV) get embedding-friendly preprocessing. This list is not exhaustive; Kreuzberg supports additional formats beyond what's listed here.

<details>
<summary>Format table</summary>

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

- [Kreuzberg] parses 90+ document formats with heading-aware chunking.
- [llama.cpp] is the local model runtime: lilbee bundles its `llama-server` and starts it for you, so every chat, embedding, vision, and reranker call goes through it. [llama-swap] keeps a server per role resident together behind one endpoint, and [gguf-parser] estimates each model's memory footprint so lilbee loads what fits. Without llama.cpp there is no lilbee.
- [Hugging Face Hub] (via [huggingface_hub]) hosts the model catalog and handles every download. Search, browse, and pull all route through it.
- [LanceDB] is the embedded vector store.
- [tree-sitter] (via [tree-sitter-language-pack]) chunks code across 150+ languages.
- [crawl4ai] and [Playwright] crawl the web; [Tesseract] is the OCR fallback when no vision model is set.
- [LiteLLM] bridges cloud model providers (the `[litellm]` optional extra).
- [Textual] draws the terminal; [Litestar] runs the HTTP server.
- [MCP Python SDK] is the agent surface; [Typer] is the CLI; [Pydantic] is the config + validation backbone.
- [Nuitka] compiles the whole thing into the standalone single-file binary, bundling its own Python runtime so there is nothing to install and nothing to compile.

## FAQ

**What is lilbee, in one line?** A local AI search engine: it runs the models and searches your files, code, and the web, with answers that cite the source.

**Is it really one program?** Yes. The model runtime (llama.cpp) and the index (LanceDB) run inside lilbee. No separate model server, no vector database, no container.

**Is lilbee a model manager?** Yes, a complete one. It browses Hugging Face, downloads models, assigns roles, runs them on Metal, Vulkan, or CUDA, and places large models across multiple GPUs, so you do not need a separate model runner. Already use Ollama or LM Studio? Point lilbee at them instead.

**Does my data leave my machine?** No. Your files stay on disk and search runs locally. A cloud model is used only when you pick one.

**Will a model fit my GPU?** lilbee reads the GGUF file and your devices and estimates fit before you download, and splits large models across multiple GPUs. More at [lilbee.sh/gpu](https://lilbee.sh/gpu/).

**Can my coding agent use it?** Yes, over MCP. The agent reads your real code and docs before answering, cited to the file and line. More at [lilbee.sh/mcp](https://lilbee.sh/mcp/).

## Support

Having trouble? See [TROUBLESHOOTING.md](https://github.com/tobocop2/lilbee/blob/main/TROUBLESHOOTING.md) for log locations and common failures.

lilbee is built and maintained by one person. If it is useful to you, you can chip in via [PayPal](https://paypal.me/lilbeedotsh). Bug reports and pull requests help just as much.

## License

MIT. See [LICENSE](LICENSE).

[Kreuzberg]: https://github.com/kreuzberg-dev/kreuzberg
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
