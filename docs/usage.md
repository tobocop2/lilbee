# Usage Guide

> **There are two recommended ways to use lilbee.** Run `lilbee` for the TUI if
> you're the one driving; wire it into your agent over MCP if an AI agent is.
> Both cover the everyday workflow: pick models, index files, search, chat,
> manage the wiki. You should not normally need to touch a CLI flag, an
> environment variable, or `config.toml` by hand for either path.
>
> Everything in [Reference for advanced users](#reference-for-advanced-users)
> further down (CLI commands, the HTTP server, env vars, the config file) is
> for users who need to drive lilbee from outside those two paths: CI, scripts,
> headless boxes, custom integrations. Skip past it unless that's you.

- [The TUI](#the-tui)
  - [First run](#first-run)
  - [Adding documents](#adding-documents)
  - [Search vs Chat mode](#search-vs-chat-mode)
  - [Model bar](#model-bar)
  - [Catalog screen](#catalog-screen)
  - [Settings screen](#settings-screen)
  - [Slash commands](#slash-commands)
  - [Task Center](#task-center)
  - [Wiki](#wiki)
- [Agent integration (MCP)](#agent-integration)
- [Per-project libraries](#per-project-libraries)
- [Cloud models](#cloud-models)
- [Reference for advanced users](#reference-for-advanced-users)
  - [CLI commands](#cli-commands)
  - [HTTP server](#http-server)
  - [Data locations](#data-locations)
  - [Config file (config.toml)](#config-file)
  - [Environment variables](#environment-variables)
  - [Optional extras](#optional-extras)
  - [Cross-encoder reranking](#cross-encoder-reranking)
  - [Semantic chunking](#semantic-chunking)
  - [OCR](#ocr)

---

## The TUI

`lilbee` (no args) launches the full Textual app: streaming chat with clickable
citations, a model bar, a Task Center for background jobs, and screens for the
catalog, settings, the setup wizard, and the auto-built wiki.

Press `?` at any time for the keybinding cheat sheet, `Ctrl+P` for the Textual
command palette, and `/help` for the slash-command catalog. Every action lilbee
can take is reachable from one of those three.

### First run

The welcome screen walks you through:

1. Picking an **embedding model** (required for indexing your documents).
2. Picking a **chat model** (optional; needed for cited answers and the
   conversational REPL).

Both pickers show every model the role can run: native GGUFs from the built-in
catalog and, when an API key is configured, anything your provider exposes.
Pulling a model installs it; selecting it assigns it to the matching role.

After that you're in chat. `/add` indexes documents, `/crawl` indexes a website
(crawler extra required), `/settings` opens the settings tabs.

### Adding documents

Press `/add` to add files, directories, or web pages. Paths tab-complete; the
job runs in the Task Center, so you can keep chatting while indexing happens.

If a file with the same name is already indexed, `add` skips it. To re-index in
place, remove the document first with `/delete name`, or pass `--force` from
the CLI.

### Search vs Chat mode

The bar above the prompt has a two-state pill that toggles between **Search**
and **Chat**. F3 flips it.

- **Search** (default). Every prompt goes through document retrieval first.
  Relevant chunks are passed to the chat model as context, and the
  reply ends with a Sources block of clickable citations. If retrieval finds
  nothing, lilbee falls through to a chat-only answer for that single prompt
  and shows a one-time toast so you know the answer wasn't backed by your files.
- **Chat.** Retrieval is skipped entirely; the model answers directly from
  whatever it already knows. Useful for conversational follow-ups that don't
  need your library, or for talking to a model when you haven't indexed
  anything yet.

The toggle is disabled and forced to Chat when no embedding model is
configured (Search has nothing to search against).

### Model bar

The bar above the prompt shows what's active for chat and embedding, plus the
mode toggle:

```
Chat [Qwen3 0.6B]   Embed [Nomic v1.5]   [Search | Chat]
```

The chat and embedding labels are searchable pickers. Click one (or focus it
with Tab and press Enter) to open a modal with a search input above a
virtualized list of every model that role can use. Type to filter; Enter picks
the highlighted row; Escape cancels. The active model's display label sits on
the button at all times so you can see what's loaded without opening the
picker.

The pickers list everything the role can run: native GGUFs you already have
installed plus, when the `litellm` extra is installed and an API key is set,
whatever the SDK backend exposes for that provider. There is no separate
"local-only" picker; routing happens automatically once the model is selected.

### Catalog screen

`/models` (or `/m`, `/catalog`) opens the catalog. The top of the screen has
two sub-tabs:

- **Local.** Everything you can run on this machine. Native GGUF models from
  the developer's picks, the full HuggingFace catalog browsable by task and
  size, and any locally-running OpenAI-compatible backend that exposes models
  on its REST endpoint. Toggle between a card grid and a dense list view;
  both share the same search box. Pulling a model installs it; selecting an
  installed model assigns it to the matching role.
- **Frontier.** Cloud chat models grouped by provider (Anthropic, Gemini,
  OpenAI, and so on). Only appears when at least one provider API key is
  configured, either via the Settings screen's API-Keys tab or the
  provider's standard environment variable. The list shows whatever the
  provider's SDK exposes for that key with no curation on lilbee's side.
  Selecting a row makes that model the active chat model and routes
  subsequent prompts through the cloud provider; a persistent warning
  appears in the model bar so it's clear when chunks are leaving the
  machine.

### Settings screen

`/settings` opens a tabbed settings editor. Every value persists to
`config.toml` (see [Config file](#config-file)); the equivalent env vars
override individual values at runtime without touching the file.

Tabs for features that aren't installed are hidden, not greyed out:

- **API-Keys** appears only when the `litellm` extra is installed.
- **Crawling** appears only when the `crawler` extra is installed.
- **Wiki** appears only when the experimental wiki layer is enabled
  (`cfg.wiki = true` in `config.toml` or `LILBEE_WIKI=1`).

Install the relevant extra (or flip the wiki flag) and the tab shows up on
the next visit.

### Slash commands

All slash commands available from the TUI. Slash commands and paths
tab-complete; `/help` opens the same catalog live.

| Command | Aliases | Description |
|---------|---------|-------------|
| `/model [name]` | | Switch chat model. No args opens the catalog picker; with a name, switches directly or prompts to download |
| `/models` | `/m`, `/catalog` | Browse the full model catalog |
| `/add <path>` | | Add a file or directory to the index (tab-completes paths) |
| `/crawl [url]` | | Crawl a URL. No args opens a dialog |
| `/delete <name>` | | Remove a document from the index |
| `/remove <name>` | | Remove an installed model |
| `/wiki` | | Open the auto-generated wiki |
| `/setup` | | Run the first-time setup wizard |
| `/settings` | | View or change settings |
| `/set <key> <val>` | | Change a setting (e.g. `/set temperature 0.7`) |
| `/theme <name>` | | Switch theme |
| `/status` | | Show indexed documents and config |
| `/login <token>` | | Log in to HuggingFace |
| `/clear` | | Clear chat history |
| `/cancel` | | Cancel active operations |
| `/reset` | | Factory reset (asks for confirmation) |
| `/version` | | Show lilbee version |
| `/help` | `/h` | Show available commands |
| `/quit` | `/q`, `/exit` | Exit |

A spinner shows while waiting for the first token from the LLM.

### Task Center

Background jobs (sync, crawl, wiki build, model pull) appear in the Task
Center with live progress. `/cancel` cancels the active operation. The TUI
stays responsive throughout; each inference role (chat, embed, rerank,
vision) runs in its own subprocess so a stuck model doesn't lock the chat.

### Wiki

lilbee analyzes the documents you've indexed and writes a wiki about them,
inspired by Andrej Karpathy's [LLM Wiki](https://karpathy.ai/llmwiki/). Pages
compound across sources instead of being one-per-document, so concepts and
entities that show up repeatedly in your library get their own page with
citations from every source that mentions them.

Open it with `/wiki`. Pages live under `$LILBEE_DATA/wiki/`:

| Directory | Contents |
|-----------|----------|
| `concepts/` | One page per LLM-identified concept (e.g. `braking-systems.md`) |
| `entities/` | One page per proper-noun entity extracted by NER (e.g. `henry-ford.md`) |
| `drafts/` | Low-faithfulness or parse-failure pages awaiting your accept/reject |
| `archive/` | Pages retired by `lilbee wiki prune` |
| `synthesis/` | Cross-source pages produced by `lilbee wiki synthesize` |
| `index.md` | Auto-generated table of contents, grouped by page type |
| `log.md` | Append-only audit trail of every build, ingest, lint, and prune |

Every section is citation-verified against the source chunks and scored for
embedding faithfulness; low-confidence output routes to `drafts/`. Plain-text
concept slugs inside page bodies are rewritten to Obsidian `[[wiki links]]` so
the graph view shows how ideas connect. The directory is Obsidian-compatible
out of the box.

The wiki is built incrementally during sync (default cap of 20 changed sources
per sync) so day-to-day re-ingest never churns existing concept slugs. Rebuild
from scratch, lint, drafts review, and prune are also available as CLI
commands (see [Wiki commands](#wiki-1)) and as MCP tools.

## Agent integration

lilbee is also the retrieval backend for AI coding agents. Wire it into any
agent that speaks MCP (Claude Code, opencode, Cursor, anything else) and the
agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets it can
quote back. Your files, the embeddings, and the index stay on your computer.
See the [`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) for the full MCP
tool list and workflows. Non-MCP agents can use the [JSON CLI
fallback](#json-cli-fallback) below.

### JSON CLI fallback

Every lilbee CLI command accepts `--json` (or `-j`) before the subcommand for
structured output. Use this as the shell-out path for agents that can't speak
MCP. The shape mirrors the MCP tools: one JSON object per stdout line, errors
return non-zero exit with `{"error": "..."}`, and `distance` scores are
lower-is-more-relevant. Vectors are stripped from output.

**Read (inline, no LLM):**

```bash
lilbee --json status                           # indexed sources, models, totals
lilbee --json search "query" --top-k 12        # cited chunks (no LLM at query time)
lilbee --json chunks manual.pdf                # inspect how one source was chunked
lilbee --json topics "auth"                    # concept-graph view of a query
lilbee --json model list                       # installed models
lilbee --json model show <ref>                 # catalog + installed metadata for a model
lilbee --json version
lilbee --json self-check                       # runtime + model self-check
```

**Write (LLM calls or long ops):**

```bash
lilbee --json add ~/docs ~/notes               # copy files / dirs in, indexes in one call
lilbee --json add https://example.com/page     # URL becomes a markdown source
lilbee --json sync                             # re-index after edits to the documents directory
lilbee --json rebuild                          # nuke the index and re-ingest everything
lilbee --json remove manual.pdf                # drop chunks (keeps the file on disk)
lilbee --json remove manual.pdf --delete       # drop chunks and delete the source file
lilbee --json ask "question"                   # full local RAG (llama-cpp or SDK backend)
lilbee --json model pull <ref>                 # download a model, streams JSON progress events
lilbee --json model pull <ref> --allow-unsupported  # override the architecture-compat check
lilbee --json model rm <ref>                   # delete an installed model
lilbee --json reset --yes                      # factory reset (destructive, requires --yes)
lilbee --json init [path]                      # create a .lilbee/ in a directory
```

`add` is the most common entry point: files, directories, and URLs all go
through it, and indexing happens in the same call. Long ops take seconds to
minutes; the final JSON includes per-file outcomes and counts.

**Wiki (experimental, opt-in):**

```bash
lilbee --json wiki status                      # page counts + wiki_enabled flag
lilbee --json wiki build                       # generate the topic / entity wiki
lilbee --json wiki update                      # refresh after a sync (full rebuild today)
lilbee --json wiki synthesize                  # cross-source synthesis pages
lilbee --json wiki lint                        # orphans, stale citations, pending drafts
lilbee --json wiki citations <source>          # per-section citation coverage for one source
lilbee --json wiki drafts list                 # pending drafts with drift + faithfulness
lilbee --json wiki drafts diff <slug>          # unified diff between a draft and the live page
lilbee --json wiki drafts accept <slug>        # promote a draft to concepts/ or entities/
lilbee --json wiki drafts reject <slug>        # discard a draft
lilbee --json wiki prune                       # archive stale pages
```

**Two patterns worth knowing:**

- **`search` vs `ask`.** `search` returns raw chunks without an LLM call. Use it
  when your agent has its own LLM and just needs context from your files. `ask` runs
  lilbee's local RAG end-to-end and returns an answer with sources. Most
  non-MCP agents want `search`.
- **Citation rule still applies.** Every fact stated from `search` results must
  trace back to a chunk's `source` + line range, exactly as returned. Don't
  invent.

**Output shape:**

```json
// lilbee --json search "oil change interval" --top-k 3
{"command": "search", "query": "oil change interval", "results": [
  {"source": "manual.pdf", "chunk": "Change oil every 5,000 miles...", "distance": 0.23, "chunk_type": "raw"}
]}

// lilbee --json status
{"config": {...}, "sources": [{"filename": "manual.pdf", "chunk_count": 42}], "total_chunks": 42}

// lilbee --json model pull <ref>  (streams events, then a final "done" line)
{"event": "progress", "model": "...", "bytes": 12345678, "total": 999999999}
{"event": "done", "model": "...", "installed": true}
```

**Gaps vs MCP.** The CLI doesn't expose `crawl` (non-blocking URL crawling) or
per-key settings management. Use `add <url>` for one-shot URL ingest. For
continuous crawling or programmatic settings, the HTTP server exposes both: see
the [REST API reference](https://lilbee.sh/api/).

> [!CAUTION]
> **Private data and cloud agents**
>
> When an agent queries lilbee, retrieved chunks are sent to whatever LLM the
> agent uses, including cloud-hosted models. If your index contains private,
> confidential, or sensitive documents, verify two things before connecting an
> agent:
>
> 1. **Check which database is active.** Run `lilbee status` and confirm the
>    data directory is the one you intend the agent to access. lilbee walks up
>    the directory tree to find `.lilbee/`, so you may be exposing a different
>    project's data than you expect.
> 2. **Know where your agent sends data.** If the agent uses a cloud-hosted
>    model, your document chunks will leave your machine. Use a local model
>    (native GGUF via llama-cpp or a local SDK backend) if your documents must
>    stay private.

## Per-project libraries

lilbee uses a git-like per-project model. Running `lilbee init` from a project
directory creates a `.lilbee/` folder there, just like `git init` creates
`.git/`. Once initialized, every lilbee invocation from that directory (or any
subdirectory) automatically uses the local database, both in the TUI and from
the CLI:

```bash
cd ~/projects/my-engine
lilbee init                  # creates .lilbee/ here
lilbee                       # launches the TUI scoped to this library
```

If there's no `.lilbee/` in the current directory, lilbee walks up the tree
looking for one; if none is found, it falls back to the global database at the
platform default location (see [Data locations](#data-locations)). `/status`
in the TUI (or `lilbee status` from the shell) shows which database is active.

## Cloud models

lilbee runs entirely on your machine by default. There are two ways to use
cloud models when you want to:

- **Bring your own key, inside lilbee.** Install the `[litellm]` extra and add
  an API key in `/settings` → API-Keys, then pick a cloud model from the model
  bar picker or the Frontier tab in `/catalog`. The TUI shows a persistent
  warning whenever a cloud role is active.
- **Pair lilbee with a cloud agent over MCP.** lilbee stays the local part:
  your files, the embeddings, the search index. The agent (Claude Code,
  opencode, anything that speaks MCP) calls `lilbee_search` / `lilbee_add` and
  gets back cited snippets. See [Agent integration](#agent-integration).

Either way your files and the index never leave the machine; only the queries
and the snippets the model needs to answer cross the wire when you opt in.

---

## Reference for advanced users

Everything from here on is for users who want to script lilbee, run it
headless, integrate it with an agent over MCP, or override individual settings
without opening the TUI. None of it is required for everyday use; the TUI
exposes all of these through pickers, `/settings`, and slash commands.

## CLI commands

Every TUI action is also a CLI command. Indexing and search need an embedding
model installed first (the default Nomic v1.5 GGUF is fine):

```bash
lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF   # required before `lilbee add`
lilbee model browse                                     # pick a chat model interactively
```

Without an embedding model installed, every `lilbee add` call reports the file
as `failed` with `Model '…' not found in registry`.

### Index and search

```bash
lilbee add ~/Documents/manual.pdf       # add a file
lilbee add ~/notes/                     # add a directory
lilbee add ~/docs/*.md ~/data/report.pdf
lilbee add manual.pdf --force           # re-index in place (default is skip-if-present)

lilbee search "oil change interval"
lilbee search "oil change interval" --top-k 20

lilbee ask "What is the recommended oil change interval?"
lilbee ask "Explain this" --model qwen3
```

`search` only needs the embedding model; `ask` also needs a chat model.

### Manage documents

| Command | Description |
|---------|-------------|
| `lilbee remove manual.pdf` | Remove from the index (keeps source file) |
| `lilbee remove manual.pdf --delete` | Remove and delete the source file |
| `lilbee chunks manual.pdf` | Inspect how a document was chunked |
| `lilbee sync` | Re-index changed files |
| `lilbee rebuild` | Nuke the database and re-ingest everything |
| `lilbee reset` | Factory reset. Deletes all documents and data |

### Wiki

```bash
lilbee wiki build                      # build the wiki from the current index
lilbee wiki lint                       # find orphan pages, stale links, pending drafts
lilbee wiki synthesize                 # generate cross-source synthesis pages
lilbee wiki drafts list                # list pending drafts
lilbee wiki drafts accept <slug>       # promote a draft to concepts/ or entities/
lilbee wiki drafts reject <slug>       # discard a draft
lilbee wiki prune                      # move stale pages to archive/
```

MCP tools mirror the CLI: `wiki_list`, `wiki_read`, `wiki_synthesize`,
`wiki_lint`, `wiki_citations`, `wiki_drafts_list`, `wiki_drafts_diff`,
`wiki_prune`.

### Vault and status

```bash
lilbee init                            # create .lilbee/ in cwd
lilbee status                          # show active data dir, models, document count
lilbee --global status                 # skip any .lilbee/ and use the platform default
lilbee --data-dir ~/kb status          # use an explicit data dir
```

### Top-level flags

`--model` / `-m`, `--data-dir` / `-d`, `--global` / `-g`, `--vision`,
`--vision-timeout`, `--log-level`, `--json` / `-j`, `--version` / `-V`.

## HTTP server

The HTTP server exposes a REST API that any tool or GUI can hit. By default it
picks a random port and writes it to `<data_dir>/server.port` so callers on
the same machine can discover it. Start it with `lilbee serve`:

```bash
lilbee serve                           # random port
lilbee serve --port 8080               # fixed port
lilbee serve --host 0.0.0.0            # bind all interfaces (default: 127.0.0.1)
```

The surface covers search (with SSE streaming variants for `ask` and `chat`),
document lifecycle, crawling, model management, configuration (including a
defaults endpoint that powers per-setting reset), and status/health. The
Obsidian plugin uses the `/api/source` endpoint for vault-aware source
retrieval. Interactive REST API docs live at `/schema/redoc` when the server
is running, and the full OpenAPI schema is published at the
[REST API reference](https://lilbee.sh/api/). (Note: this is the HTTP server
reference. A Python-library API reference is still being written; for now,
the source under `src/lilbee/` is the canonical reference.)

Server-specific env vars live in the [Server table](#server) below.

### Running as a service

For tools that talk to lilbee's HTTP REST API all day (the Obsidian plugin, custom GUIs, anything hitting `/api/*`), your OS launcher can keep the HTTP server warm so requests skip the cold-start. This is the only lilbee surface designed for that pattern; the TUI, `lilbee chat`, the MCP server, and the rest of the CLI cold-start and exit on every invocation by design.

Pull at least one chat and embedding model first (`lilbee model pull <name>`). The daemon will start without one, but `/api/*` requests fail until a model is available. All recipes pin the server to `127.0.0.1:42697`.

**macOS (Homebrew):**

```bash
brew services start lilbee
```

**Linux (Arch / AUR, systemd):**

```bash
systemctl --user enable --now lilbee
```

On a headless server (no graphical login session), also run `loginctl enable-linger $USER` so the service survives logout.

**NixOS:** add the module to your `configuration.nix`:

```nix
{
  imports = [ lilbee.nixosModules.lilbee ];
  services.lilbee.enable = true;
}
```

## Data locations

lilbee resolves the data directory in this order (highest priority first):

| Priority | Method | Example |
|----------|--------|---------|
| 1 | `--data-dir` flag or `LILBEE_DATA` env var | `lilbee --data-dir ~/my-kb status` |
| 2 | `.lilbee/` directory (walks up from cwd) | Created by `lilbee init` |
| 3 | `--global` flag (skip `.lilbee/`, use platform default) | `lilbee --global status` |
| 4 | Platform default | See table below |

### Platform defaults

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/lilbee/` |
| Linux | `~/.local/share/lilbee/` |
| Windows | `%LOCALAPPDATA%/lilbee/` |

Run `lilbee init` to create a `.lilbee/` directory in your project. It
contains `documents/` (your indexed files), `data/` (the search index), and a
`.gitignore` that excludes derived data.

## Config file

Every persisted lilbee setting lives in a single TOML file at
**`<data-dir>/config.toml`**. For the platform default that's
`~/.local/share/lilbee/config.toml`; for a project, it's `.lilbee/config.toml`
next to your code. Same file, same shape, regardless of which data dir you're
on; the path just follows wherever the data dir is resolved to.

Sections mirror the settings categories in
[Environment variables](#environment-variables) below: `[general]`,
`[retrieval]`, `[chunking]`, `[generation]`, `[server]`, etc. Example:

```toml
# .lilbee/config.toml

[general]
chat_model = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
theme = "rose-pine"

[retrieval]
top_k = 12
max_distance = 0.6
reranker_model = "Qwen/Qwen3-Reranker-0.6B-GGUF/qwen3-reranker-0.6b.Q8_0.gguf"

[generation]
temperature = 0.2
num_ctx_max = 32768
```

You don't have to write the file by hand: the TUI's `/settings` screen and the
`/set <key> <value>` slash command both persist to it, and the equivalent
`LILBEE_<KEY>=...` env vars override individual values at runtime without
touching the file.

## Environment variables

Every setting has a default that works out of the box, and every value below is
also editable from `/settings` in the TUI. The tables exist for scripted
overrides, headless deployments, and CI; you should not normally need to touch
them by hand. Tables are grouped from most-commonly-touched to rarely-touched,
so you can skim the top and skip the bottom unless you have a specific reason.

### Common settings

The ones most users set at least once.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path. Overridden by `--data-dir` or a `.lilbee/` vault walked up from cwd |
| `LILBEE_CHAT_MODEL` | `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | Chat model. Native GGUF by default; with `pip install --pre 'lilbee[litellm]'` (or `uv tool install --prerelease=allow 'lilbee[litellm]'`), any remote name the SDK backend understands |
| `LILBEE_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf` | Embedding model. Changing this requires `lilbee rebuild` |
| `LILBEE_VISION_MODEL` | *(none)* | Vision OCR model. When set, takes precedence over Tesseract on scanned PDFs and images |
| `LILBEE_VISION_TIMEOUT` | `120` | Per-page vision OCR timeout in seconds (`0` = no limit) |
| `LILBEE_LOG_LEVEL` | `WARNING` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | Custom system prompt for RAG answers |
| `LILBEE_SHOW_REASONING` | `false` | Show the model's `<think>` reasoning tokens in chat output. Useful with Qwen3, DeepSeek-R1, and other reasoning models |
| `LILBEE_HF_TOKEN` | *(none)* | HuggingFace access token. Avoids the unauthenticated download rate limit and unlocks gated repos. Same token can be persisted via the `System` tab in `/settings` (stored in plain text at `config.toml`). `HF_TOKEN` env var also accepted |

### Retrieval tuning

Reach for these when search quality matters. Defaults are solid; tune only if
something feels off.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_TOP_K` | `8` | Number of retrieval results returned |
| `LILBEE_MAX_DISTANCE` | `0.65` | Cosine distance cutoff. Lower = stricter filtering, fewer but more relevant results. `1.0` disables filtering |
| `LILBEE_MMR_LAMBDA` | `0.5` | Relevance vs. diversity balance (1.0 = pure relevance, 0.0 = pure diversity). Raise for factual lookups, lower for exploratory queries |
| `LILBEE_DIVERSITY_MAX_PER_SOURCE` | `3` | Max chunks from a single source document in the top-K. Prevents one big file from dominating results |
| `LILBEE_QUERY_EXPANSION_COUNT` | `3` | LLM-generated query variants per search. `0` disables expansion entirely for faster queries |
| `LILBEE_RERANKER_MODEL` | *(none)* | GGUF cross-encoder reranker for a precision pass over top results. See [Cross-encoder reranking](#cross-encoder-reranking) |
| `LILBEE_RERANK_CANDIDATES` | `60` | Candidates to rerank when a reranker is configured |
| `LILBEE_HYDE` | `false` | Enable Hypothetical Document Embeddings: an LLM drafts a hypothetical answer, that's embedded, and results are merged with the original query's. Adds ~500 ms per query; helps on vague questions |
| `LILBEE_HYDE_WEIGHT` | `0.7` | How much to trust HyDE results relative to the direct query (0.0-1.0) |
| `LILBEE_ADAPTIVE_THRESHOLD` | `false` | When too few results pass `LILBEE_MAX_DISTANCE`, widen the threshold step by step. Useful on small or noisy corpora |
| `LILBEE_ADAPTIVE_THRESHOLD_STEP` | `0.2` | How much to widen per step when adaptive threshold triggers |
| `LILBEE_TEMPORAL_FILTERING` | `true` | When the query contains temporal cues ("recent", "last week"), filter results by document date and sort by recency |
| `LILBEE_MAX_CONTEXT_SOURCES` | `6` | Max chunks included in the LLM's RAG context. Raise for more coverage, lower for shorter prompts |

### Ingestion and chunking

How documents become chunks. Changes here require `lilbee rebuild` to take
effect on already-indexed material.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_CHUNK_SIZE` | `512` | Target tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between adjacent chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk passed to the embedder |
| `LILBEE_SEMANTIC_CHUNKING` | `false` | Experimental topic-aware chunking. See [Semantic chunking](#semantic-chunking) |
| `LILBEE_TOPIC_THRESHOLD` | `0.75` | Cosine boundary threshold for semantic chunking (lower = more splits) |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding dimensionality. Must match the embedding model |

### Generation

LLM output shape. lilbee sets conservative defaults below; the model's own
defaults apply only when a value is explicitly unset in code or config.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_TEMPERATURE` | `0.1` | Sampling temperature |
| `LILBEE_TOP_P` | `0.9` | Nucleus sampling threshold |
| `LILBEE_TOP_K_SAMPLING` | `40` | Top-k sampling |
| `LILBEE_REPEAT_PENALTY` | `1.1` | Repetition penalty |
| `LILBEE_NUM_CTX` | *(auto)* | Context window size. Empty = sized automatically (aims for `LILBEE_CHAT_N_CTX_TARGET`, ceiling at `LILBEE_NUM_CTX_MAX` or the model's training_ctx). Set explicitly to lock a specific value |
| `LILBEE_CHAT_N_CTX_TARGET` | *(auto)* | Target context size for the dynamic picker, scaled by total host RAM: `<16 GiB → 8192`, `16-32 GiB → 12288`, `32-64 GiB → 16384`, `≥64 GiB → 24576`. 8192 is the floor on smaller hosts. The picker still clamps the result to the model's training context and available memory at worker start. Set explicitly to override |
| `LILBEE_NUM_CTX_MAX` | *(auto)* | Explicit ceiling for the dynamic context picker. Empty = use the model's training_ctx from GGUF metadata as the only ceiling. Set to cap below training_ctx on memory-constrained hosts |
| `LILBEE_FLASH_ATTENTION` | *(auto)* | Flash attention. Empty/`auto` enables it with a TypeError fallback for older llama-cpp-python builds; `1`/`true`/`on` forces on; `0`/`false`/`off` disables. Resolves the `padding V cache to 1024` warning on models with uneven per-layer V dims |
| `LILBEE_KV_CACHE_TYPE` | `q8_0` | KV cache element type: `f16`, `f32`, `q8_0`, `q4_0`. `q8_0` (default) halves KV memory vs `f16` with no measurable chat-quality loss; `q4_0` quarters it with a small quality cost. Quantized variants require flash attention to be enabled |
| `LILBEE_N_GPU_LAYERS` | *(auto)* | Layers to offload to GPU. Empty/`auto` = all (recommended), `cpu` = none, integer = partial offload for tight VRAM |
| `LILBEE_SEED` | *(model default)* | Random seed for reproducibility |

### Server

Only relevant when running the HTTP server.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_SERVER_HOST` | `127.0.0.1` | Bind address |
| `LILBEE_SERVER_PORT` | random | Port (overridden by `--port`) |
| `LILBEE_CORS_ORIGINS` | *(none)* | Comma-separated list of extra allowed CORS origins, e.g. `https://my-app.com`. Additive; the default regex below still applies |
| `LILBEE_CORS_ORIGIN_REGEX` | *(see usage)* | Regex for allowed origins. Default matches `app://obsidian.md`, `capacitor://localhost`, and any `http(s)://localhost`, `127.0.0.1`, or `[::1]` with any port. Set to `^$` to opt out and rely solely on `LILBEE_CORS_ORIGINS` |

### Wiki tuning (experimental)

Only relevant if you run `lilbee wiki build`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_WIKI_INGEST_UPDATE_CAP` | `20` | Max changed sources processed by incremental wiki updates during `lilbee sync`. Prevents a big re-ingest from churning concepts |
| `LILBEE_WIKI_CONCEPT_MAX_CHUNKS_PER_PAGE` | `25` | Top-K chunks behind each wiki page section |

### Advanced

Rarely touched. Defaults derived from published IR research; there's usually a
reason the defaults are the defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_EXPANSION_SKIP_THRESHOLD` | `0.8` | BM25 confidence threshold above which query expansion is skipped (90th-percentile sigmoid-normalized score) |
| `LILBEE_EXPANSION_SKIP_GAP` | `0.15` | Minimum score gap between top-1 and top-2 for expansion to skip (ensures the match is unambiguous) |
| `LILBEE_EXPANSION_GUARDRAILS` | `true` | Filter expansion variants whose embedding drifts too far from the original query |
| `LILBEE_EXPANSION_SIMILARITY_THRESHOLD` | `0.5` | Minimum query-variant cosine similarity to survive the guardrail |
| `LILBEE_CANDIDATE_MULTIPLIER` | `3` | Extra candidates to retrieve before MMR reranking |

## Optional extras

lilbee works out of the box with llama-cpp for local inference. These optional
extras add capabilities that require heavier dependencies:

```bash
# pip
pip install --pre 'lilbee[graph]'      # concept graph: topic clustering + search boosting
pip install --pre 'lilbee[crawler]'    # web crawling: index websites alongside local docs
pip install --pre 'lilbee[litellm]'    # remote providers: connect to any SDK-backed provider

# uv tool
uv tool install --prerelease=allow 'lilbee[graph]'
uv tool install --prerelease=allow 'lilbee[crawler]'
uv tool install --prerelease=allow 'lilbee[litellm]'
```

Install multiple at once:

```bash
pip install --pre 'lilbee[graph,crawler,litellm]'
uv tool install --prerelease=allow 'lilbee[graph,crawler,litellm]'
```

**NVIDIA users**: the default Vulkan build works, but the CUDA flavour is
faster and dodges the Vulkan-loader crash that affects NVIDIA-on-Windows
setups. Same `lilbee` command, links straight against your NVIDIA driver:

```bash
pip install --pre lilbee --extra-index-url https://lilbee.sh/cu125/
brew install tobocop2/lilbee/lilbee-cuda
paru -S lilbee-cuda
nix run github:tobocop2/lilbee#lilbee-cuda
```

The `lilbee-cuda` AUR package conflicts with `lilbee` and provides it, so
`paru -S lilbee-cuda` swaps automatically. On Homebrew, run `brew
uninstall lilbee` first because the two formulas both ship a `lilbee`
binary. For older drivers swap `cu125` for `cu124` or `cu121` (run
`nvidia-smi` to see which matches). cu121 and cu124 are wheel-index and
direct-download only; cu125 is the variant fanned out to the package
managers.

While 0.6.66 is in beta, the `--pre` flag (or uv's `--prerelease=allow`) is
required on every install.

Cross-encoder reranking is built in (no extra required); see
[Cross-encoder reranking](#cross-encoder-reranking) below.

---

### Concept graph

Builds a topic map of your documents at index time. Related concepts are linked
in a co-occurrence graph, which is used to boost search results and expand
queries with related terms, all without extra LLM calls.

**What it does:** Extracts noun phrases from every chunk using spaCy, computes
PMI co-occurrence weights between concepts, and clusters them with the Leiden
algorithm. At search time, queries are expanded with graph neighbors and
results overlapping query concepts get a relevance boost.

**When to use it:** Large corpora (100+ documents) where the same topics
appear across multiple files. The graph helps surface connections that pure
vector similarity misses. For example, finding "deployment" documents when
searching for "CI/CD" because those concepts co-occur frequently.

**Install:** `pip install --pre 'lilbee[graph]'` or
`uv tool install --prerelease=allow 'lilbee[graph]'`

**Configuration:**

```bash
export LILBEE_CONCEPT_GRAPH=true              # enable (default: true when deps installed)
export LILBEE_CONCEPT_BOOST_WEIGHT=0.3        # how much concept overlap matters (0.0-1.0)
export LILBEE_CONCEPT_MAX_PER_CHUNK=10        # max concepts extracted per chunk
```

The graph is built automatically during `lilbee sync`. No extra commands
needed; search results are boosted transparently.

Based on: Microsoft Research's LazyGraphRAG technique, Church & Hanks 1990
(PMI), Traag et al. 2019 (Leiden).

---

### Web crawling

Index web pages alongside your local documents. Crawl single pages or follow
links recursively.

**What it does:** Fetches web pages using a headless browser (Playwright),
extracts markdown content, and indexes it. Supports recursive crawling with
configurable depth, concurrent fetching, live progress, cancel, per-domain
rate-limit + retries on HTTP 429/503, and SSRF protection against internal
network access.

**When to use it:** When your library spans both local files and web content
such as documentation sites, wikis, or internal tools. Crawled content is
hash-tracked so re-crawling only re-indexes changed pages.

**Install:** `pip install --pre 'lilbee[crawler]'` or
`uv tool install --prerelease=allow 'lilbee[crawler]'`

**Usage:**

```bash
# Single page (no --crawl)
lilbee add https://docs.example.com/guide

# Whole-site crawl (recursive, unbounded by default)
lilbee add https://docs.example.com --crawl

# Cap depth or page count
lilbee add https://docs.example.com --crawl --depth 2 --max-pages 200

# Multiple URLs
lilbee add https://docs.example.com https://wiki.example.com
```

Also available via MCP (`crawl`), REST API (`POST /api/crawl`), and TUI
(`/crawl`).

**Configuration (all optional):**

```bash
# Global ceilings. Unset = no cap. Explicit --depth/--max-pages always win.
export LILBEE_CRAWL_MAX_DEPTH=3          # cap link-following depth
export LILBEE_CRAWL_MAX_PAGES=1000       # cap total pages

# Pacing within a single crawl.
export LILBEE_CRAWL_MEAN_DELAY=0.5       # seconds between requests
export LILBEE_CRAWL_MAX_DELAY_RANGE=0.5  # random jitter on top
export LILBEE_CRAWL_CONCURRENT_REQUESTS=3

# Per-domain rate-limit + retries on HTTP 429/503.
export LILBEE_CRAWL_RETRY_ON_RATE_LIMIT=true
export LILBEE_CRAWL_RETRY_BASE_DELAY_MIN=1.0
export LILBEE_CRAWL_RETRY_BASE_DELAY_MAX=3.0
export LILBEE_CRAWL_RETRY_MAX_BACKOFF=30.0
export LILBEE_CRAWL_RETRY_MAX_ATTEMPTS=3

# Other.
export LILBEE_CRAWL_TIMEOUT=30           # per-page timeout (seconds)
export LILBEE_CRAWL_MAX_CONCURRENT=0     # 0 = CPU count (top-level concurrency)
export LILBEE_CRAWL_SYNC_INTERVAL=30     # seconds between periodic syncs during crawl
```

---

### Remote providers (SDK backend)

Connect to hosted or local OpenAI-compatible LLM backends alongside lilbee's
native in-process inference.

**What it does:** Routes chat and embedding calls to any provider reachable
via the SDK backend. The routing provider automatically detects which models
are available locally vs. remotely and routes each call to the right backend.

**When to use it:** When you want a frontier model for chat while keeping
embeddings local for privacy, or to surface models from a local
OpenAI-compatible daemon alongside lilbee's native GGUF models.

**Install:** `pip install --pre 'lilbee[litellm]'` or
`uv tool install --prerelease=allow 'lilbee[litellm]'` (the extra retains the
adapter library name).

**Configuration:**

```bash
export LILBEE_LLM_PROVIDER=auto          # "auto" routes between local and remote
export LILBEE_REMOTE_BASE_URL=http://localhost:11434  # local backend URL
export LILBEE_LLM_API_KEY=sk-...         # API key for your provider
export LILBEE_CHAT_MODEL=your-model      # any remotely-supported model name
```

Provider options: `auto` (default, routes intelligently), `llama-cpp` (local
only), `remote` (hosted only).

---

## Cross-encoder reranking

Built-in. Re-scores retrieval candidates with a cross-encoder for precision on
the top results. Unlike the extras above, no extra install is required;
reranking is off by default and turns on as soon as you set
`LILBEE_RERANKER_MODEL` (or pick a reranker from `/settings`).

**What it does:** After the hybrid search pipeline (BM25 + vector + RRF)
returns candidates, a GGUF cross-encoder scores each `(query, chunk)` pair and
results are blended with position-aware weights. Top-ranked candidates keep
more of the original ranking; lower-ranked candidates trust the reranker more.

**When to use it:** When you need high-precision answers and are willing to
trade roughly 200 to 500 ms per query. Most useful with large candidate sets
where top-5 ordering matters.

**Configuration:**

```bash
export LILBEE_RERANKER_MODEL="bge-reranker-v2-m3"   # any GGUF reranker
export LILBEE_RERANK_CANDIDATES=20                  # how many candidates to rerank
```

Without a reranker set, hybrid search + MMR already provides good results for
most use cases.

Based on: Nogueira & Cho 2019 (Passage Re-ranking with BERT), Burges et al.
2005 (Learning to Rank).

---

## Semantic chunking

Experimental. Off by default. lilbee ships with two chunking strategies; which
one serves you depends on what you're indexing.

**Fixed-size (default).** Breaks documents into roughly equal token windows
with overlap. Fast, deterministic, works well on code, reference manuals, user
guides, API specs, and anything with clear structural boundaries. The
assumption is that each chunk only needs to be coherent enough for retrieval,
and the model will handle the rest from a small window of context.

**Semantic (experimental).** Uses embedding similarity to detect topic
boundaries and splits there instead of at fixed sizes. Each chunk tends to
represent one coherent thought rather than an arbitrary slice through one. The
benefit shows up on prose-heavy material: novels, essays, long-form research
papers, interview transcripts, qualitative research notes, anything where an
argument develops across paragraphs. When you ask a question, the retrieved
chunk is more likely to contain the full section that matches rather than the
first half of it plus unrelated setup.

**Trade-off:** Enabling semantic chunking triggers a one-time download of
kreuzberg's ONNX embedding model (separate from the chunk-to-vector embedder)
and runs roughly 9x more downstream embedding calls during indexing. Indexing
takes longer; retrieval latency is unchanged.

### How to enable it

Three equivalent paths:

```bash
# Environment variable
export LILBEE_SEMANTIC_CHUNKING=true

# TUI /set command (interactive)
/set semantic_chunking true

# config.toml in your .lilbee/ vault
[general]
semantic_chunking = true
```

After enabling, run `lilbee rebuild` so existing documents are re-chunked
under the new strategy. New documents added from that point use semantic
chunking automatically.

### Tuning

```bash
export LILBEE_TOPIC_THRESHOLD=0.75   # cosine threshold for topic boundaries (0.0-1.0)
```

---

## OCR

OCR is how lilbee gets text out of scanned PDFs and images. It's one step in
indexing, not a substitute for it: however the text comes out (a native parser,
Tesseract, or a vision model), it still gets embedded, so you still need an
embedding model installed.

For PDFs without embedded text, lilbee supports two OCR backends. When a
vision model is configured, it takes precedence.

| | Tesseract | Vision model |
|---|---|---|
| **Output** | Plain text | Structured markdown (tables, headings) |
| **Retrieval quality** | Fragments lose context | Chunks preserve semantic boundaries |
| **Install** | System package (`brew`/`apt`) | Native GGUF via the built-in mtmd backend, or any vision model reachable via the SDK backend (`pip install --pre 'lilbee[litellm]'` / `uv tool install --prerelease=allow 'lilbee[litellm]'`) |
| **Best for** | Simple text-only scans | Tables, multi-column layouts, formatted docs |

See [model benchmarks](benchmarks/vision-ocr.md) for detailed comparisons.

### Tesseract

[Tesseract](https://github.com/tesseract-ocr/tesseract) is used automatically
when no vision model is configured. No flags needed.

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### Vision models

lilbee runs vision OCR in one of two ways:

1. **Native mtmd backend.** Point `LILBEE_VISION_MODEL` at a GGUF vision model
   (e.g. `lightonocr`) and lilbee will load it with llama-cpp's mtmd backend
   directly, in-process. This is the recommended path and supports an SSE
   heartbeat for long scans.
2. **Remote vision model.** With `pip install --pre 'lilbee[litellm]'` (or
   `uv tool install --prerelease=allow 'lilbee[litellm]'`), set the vision
   model to any remote name your SDK backend understands. lilbee will route
   vision calls accordingly.

```bash
lilbee add report.pdf --vision                # prompts for model if none set
lilbee add report.pdf --vision-timeout 30     # per-page timeout (default: 120s, 0 = no limit)
export LILBEE_VISION_MODEL=lightonocr         # persist across runs (GGUF via mtmd)
```

Pick or change a vision model interactively via `/settings` or `/setup` in the
TUI; the selection is saved to `config.toml` and persists across sessions.
