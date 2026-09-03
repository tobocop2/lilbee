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
- [Keeping files out of the index](#keeping-files-out-of-the-index)
- [Cloud models](#cloud-models)
- [Reference for advanced users](#reference-for-advanced-users)
  - [CLI commands](#cli-commands)
  - [HTTP server](#http-server)
  - [Data locations](#data-locations)
  - [Config file (config.toml)](#config-file)
  - [Environment variables](#environment-variables)
  - [Settings reference](settings.md)
  - [Optional extras](#optional-extras)
  - [Cross-encoder reranking](#cross-encoder-reranking)
  - [Semantic chunking](#semantic-chunking)
  - [OCR](#ocr)

---

## The TUI

`lilbee` (no args) launches the full Textual app: streaming chat with clickable
citations, a model bar, a Task Center for background jobs, and screens for the
catalog, settings, and the generated wiki.

Press `?` at any time for the keybinding cheat sheet, `Ctrl+P` for the Textual
command palette, and `/help` for the slash-command catalog. Every action lilbee
can take is reachable from one of those three.

### Terminal recommendations

The TUI is a full-screen Textual app. It runs everywhere, including macOS
Terminal.app, plain xterm and over SSH: every screen, panel and key works the
same, and nothing is hidden on a lesser terminal.

Two things adapt to what the terminal can do, both detected automatically.
Panel edges are drawn with partial-block glyphs, which only tile seamlessly in
fonts that draw them cell-exact, so a terminal that does not gets box-drawing
edges instead. And a terminal with 24-bit "true color" shows the themes as
designed, while a 256-color one has no dark tinted colors to offer, so surfaces
come out as neutral greys and the tint shows only in the accents. Layout,
contrast and layering are the same either way.

- **Give it room.** The model bar and layout want about 100 columns; narrower
  and panels start to wrap.
- **Over SSH is fine,** but redraws are only as fast as the link, so a laggy
  connection makes scrolling and model switches feel heavier than they are.
- **One multiplexer layer, not two.** tmux and screen work well. Nested tmux (a
  local tmux, then SSH into a second tmux on the remote) is the one setup to
  avoid: two multiplexers fight over the same escape sequences and mangle
  box-drawing and the cursor. Attach to the remote session directly, or start
  lilbee outside the inner tmux.
- **Want the full palette over SSH?** SSH does not forward `COLORTERM`, so a
  remote lilbee uses 256 colors even when you are sitting in a truecolor
  terminal. If your local terminal really does render 24-bit color (iTerm2,
  kitty, WezTerm, Ghostty, Windows Terminal), forward it: `SendEnv COLORTERM` in
  `~/.ssh/config` plus `AcceptEnv COLORTERM` in the server's `sshd_config`.

  Do not do this from macOS Terminal.app. It sets `COLORTERM=truecolor` while
  being unable to draw 24-bit color, and lilbee only knows to ignore that claim
  by reading `TERM_PROGRAM`, which SSH does not forward either. Forwarding
  `COLORTERM` from Terminal.app tells the remote to send color the terminal
  will render as garbage. Left alone, SSH from Terminal.app is already correct.

### First run

A fresh install opens on the model catalog. Each card carries a fit chip that
says whether the model runs on this machine. Pick a chat model and pull it; the
first model you install becomes the active one, and a toast tells you chat is
ready. Models pulled earlier from the CLI are adopted automatically, and that
launch lands straight in chat.

An embedding model works the same way, but nothing asks for one up front. The
first feature that needs one (Search mode, `/remember`, indexing) sends you to
the catalog's embedding tab.

After that you're in chat. `/add` indexes documents, `/crawl` indexes a website
(crawler extra required), `/settings` opens the settings tabs.

### Adding documents

Press `/add` to add files, directories, or web pages. Paths tab-complete; the
job runs in the Task Center, so you can keep chatting while indexing happens.

If a file with the same name is already indexed, `add` skips it. To re-index in
place, remove the document first with `/delete name`, or pass `--force` from
the CLI.

Adding a code repository pulls in more than you want? See
[Keeping files out of the index](#keeping-files-out-of-the-index).

### Search vs Chat mode

The bar above the prompt has a two-state pill that toggles between **Search**
and **Chat**. F3 flips it.

- **Search** (default). Every prompt goes through document retrieval first.
  Relevant chunks are passed to the chat model as context, and the
  reply ends with a Sources block of clickable citations. If retrieval finds
  nothing usable, lilbee says so plainly instead of answering from the
  model's general knowledge; switch to Chat mode when you want an
  off-corpus answer.
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
- **Wiki** appears only when the wiki layer is enabled
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
| `/prune-ignored` | | Drop indexed documents a `.lilbeeignore` now excludes |
| `/remove <name>` | | Remove an installed model |
| `/wiki` | | Open the generated wiki |
| `/remember <text>` | | Save a memory (prefix with `pref:` for a preference). Needs memory enabled |
| `/memories` | | Browse, delete, or share saved memories |
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
are per concept or entity rather than per document, so something that shows up
repeatedly in your library gets its own page, written and cited from the source
that mentions it most. Coverage across sources comes from synthesis pages
(`lilbee wiki synthesize`) and the `[[wiki link]]` graph.

A build spends one LLM call per source document, so it is GPU-heavy and its cost
scales with the size of your library, not with how many pages you read. A few dozen
documents takes minutes; a few thousand takes hours. Nothing generates until you ask
for it.

Open it with `/wiki`. Pages live under `$LILBEE_DATA/wiki/`:

| Directory | Contents |
|-----------|----------|
| `concepts/` | One page per LLM-identified concept (e.g. `braking-systems.md`) |
| `entities/` | One page per proper-noun entity extracted by NER (e.g. `henry-ford.md`) |
| `drafts/` | Low-faithfulness or parse-failure pages awaiting your accept/reject |
| `summaries/` | Fallback landing spot for an accepted draft with no published counterpart and no recorded origin type |
| `archive/` | Pages retired by `lilbee wiki prune` |
| `synthesis/` | Cross-source pages produced by `lilbee wiki synthesize` |
| `index.md` | Auto-generated table of contents, grouped by page type |
| `log.md` | Append-only audit trail of every build, synthesize, ingest, lint, and prune, with a per-run line reporting what the quality gates did |

Every section is citation-verified against the source chunks and scored for
embedding faithfulness; low-confidence output routes to `drafts/`. Plain-text
concept slugs inside page bodies are rewritten to Obsidian `[[wiki links]]` so
the graph view shows how ideas connect. The directory is Obsidian-compatible
out of the box.

Turning the wiki on does not generate anything by itself. You wikify
explicitly: run `lilbee wiki build` (`wiki update` is the same full rebuild),
press `b` on the wiki screen in the TUI, pick **Wikify** from the command
palette, or call the build endpoints over HTTP or MCP. In the TUI the run
happens in the background with progress in the Task Center; over HTTP the
build, update, and synthesize routes stream per-phase and per-page progress as
server-sent events.

If you would rather the wiki keep itself current, set `wiki_auto_update` (or
`LILBEE_WIKI_AUTO_UPDATE=1`). Every sync then regenerates the pages its changed
documents touched, without proposing new concepts, so day-to-day re-ingest
never churns existing concept slugs. A sync that touches more than
`LILBEE_WIKI_INGEST_UPDATE_CAP` pages (default 20) skips the regeneration and
tells you to run `lilbee wiki update` yourself, so a bulk import cannot fire
hundreds of LLM calls behind your back.

A build writes every page at once and spends one LLM call per source document,
so its cost tracks how large your library is rather than how much of it you
read. The browse index is the cheap half: `lilbee wiki index` (and every sync)
extracts the names your documents mention with no LLM call at all, so the tree
lists a page as soon as its document lands. `lilbee wiki generate <slug>` then
writes one of those pages for one call, drawing on that subject's chunks across
every source naming it, which is more evidence than a build gives it.

Turning the wiki back off stops new pages being written, but the pages already
generated stay on disk and their rows stay in the index. Remove them with
`lilbee wiki wipe`, the **Delete wiki** command in the TUI palette (or `W` on
the wiki screen while the wiki is still on), `DELETE /api/wiki`, or the
`wiki_wipe` MCP tool. The TUI offers it for you when you switch the setting off. A wipe deletes only what the wiki generated; your documents are untouched.
While the wiki is off, generated pages are excluded from search results even if
you never wipe them.

Lint, drafts review, and prune are available as CLI commands (see
[Wiki commands](#wiki-1)) and as MCP tools.

## Sessions

Conversations save automatically. The first message opens a session and names
it after what you asked; each turn is appended as it lands. There is no save
step.

- **The drawer** (`ctrl+o` or `/sessions`) docks the session list beside the
  live chat. Type to filter, `enter` resumes, `^n` new chat, `^r` rename,
  `^d` delete.
- **The Sessions tab** is the same list full-screen.
- **The CLI**: `lilbee sessions list / show / rename / delete` (see
  [Sessions commands](#sessions-1)).

Resuming restores the transcript and switches back to the model the
conversation used, if it is still installed; otherwise lilbee keeps the
current model and says so.

Sessions are append-only JSONL files under `<data_dir>/sessions/`, one per
conversation. No database; back them up or sync them like any other file.
The same surface exists over HTTP (list, read, create, append, rename,
delete), so a script can own a conversation the way the TUI does. The TUI,
HTTP server, and CLI are one conversation space: start a chat in Obsidian,
continue it in the terminal.

Agents get the same tools over MCP, but they are off by default: most agent
hosts already track their own history, and the tools cost context on every
request. Turn on `mcp_sessions_enabled` if you want an agent keeping its own
saved conversations. Agent sessions stay separate from yours: they never
appear in your session list, and agents cannot list or read yours. The one
bridge is explicit: an agent can take over a session whose id you hand it
(`claim=true`, which moves it to the agent's space), and
`POST /api/sessions/{id}/claim` brings one back.

To keep nothing, turn `sessions_enabled` off in Settings: nothing is written
to disk, `ctrl+o` leaves the footer, and the Sessions view says sessions are
off instead of showing an empty list. That covers the TUI, HTTP server, and
CLI; `mcp_sessions_enabled` governs the agent half independently.

### When a conversation outgrows the model

By default, turns that no longer fit the model's context window stop being
sent. They stay on screen; a `context` gauge by the prompt shows the window
filling, and a rule in the transcript marks where the model's view now
begins.

To keep the old context instead, turn on `chat_compaction` in Settings. When
the window fills, lilbee folds the oldest turns into short notes the model
keeps reading. The fold is a model call (the gauge shows `condensing…` while
it runs), around a second or two with a small model, even on CPU. Either way
the transcript on screen and on disk is never touched; only what the model
is sent changes.

## The engine lifecycle

lilbee loads the model lazily, the way Ollama and LM Studio do: the TUI never
waits for the engine, a real llama.cpp server that starts loading in the
background the moment the app opens. Ask something before it's ready and the
answer bubble carries the load until your answer streams. Nothing freezes and
nothing is silently queued.

One engine serves every lilbee process on the machine: open a second TUI, point
a coding agent at `lilbee mcp`, or run `lilbee serve` alongside, and they all
bind to the same loaded models instead of building their own. The engine stops
when the last lilbee process exits, so the machine is left clean by default; no
VRAM, RAM, or stray processes are held while nothing is running. Two knobs
adjust that:

- **Keep engine warm** lets the engine outlive lilbee entirely, so the next
  session's first answer skips the load. Recommended if you relaunch often or
  use lilbee mainly through coding agents, whose sessions come and go.
- **Engine idle ttl minutes** bounds how long idle weights stay in memory in
  every mode, warm included, five minutes by default (the same idea as Ollama's
  keep_alive). The memory frees itself after that many idle minutes; a small
  proxy process stays behind, a few tens of MB and no VRAM. `0` keeps weights
  loaded until the engine stops.
- **`lilbee engine stop`** frees the shared engine immediately from any
  terminal, no TUI needed, whichever process started it. It reaches the machine
  slot and this project root's own overflow engine; a private overflow engine
  built for a different project root is stopped by running the command from
  that root.

Both knobs live in the TUI Settings screen, MCP `lilbee_settings_set`, the HTTP
config API, and `config.toml`. Changing a model takes effect on your next
prompt, from whichever surface you changed it. If no one else is using the
engine it restarts on the new configuration; if other lilbee processes are
serving from it, theirs keeps running and yours either rebinds, when the
running engine already serves what you asked for, or starts its own alongside
it. The first launch after a reboot is always a cold one.

To stop the engine without opening the TUI:

```bash
lilbee engine stop
```

It reports whether anything was running, frees the GPU immediately, and is safe
to run at any time.

## Memory

lilbee can remember durable facts about you and standing preferences for how
you want answers, and recall them into context on later turns regardless of
which conversation they came from. Memory is **off by default**, so users who
don't want it pay nothing and never expose the injection surface. Turn it on
once:

```
/set memory_enabled true
```

Then save things to remember. A leading `pref:` stores a standing preference
(always recalled); anything else is stored as a fact (recalled by relevance):

```
/remember pref: keep answers terse, show code first
/remember the Crown Vic manual is my main brake-work reference
```

Next time you ask a question, the relevant preference and facts are folded
into the system prompt before the model answers. Memory is never mixed into
the document citations: it shapes the answer but only your actual sources show
up under `Sources:`. Open `/memories` to list, delete, or share entries.

Memory lives in the active library's data directory, so it follows
[per-project libraries](#per-project-libraries) automatically. A factory
`/reset` clears it; a `/rebuild` (which only re-indexes documents) leaves it
alone. If you switch embedding models, your memories are re-embedded from their
stored text during the rebuild, so recall keeps working.

**Sharing model.** Your own memories are private to you unless you mark them
shared. Agent memories (see below) are private to that agent by default and
are never folded into your prompt. This asymmetry keeps an agent's notes out
of your chat unless you opt in.

**Auto-extraction (optional, off by default).** With `memory_auto_extract`
on, the TUI runs a small background pass after each answer that saves durable
facts and preferences from the exchange, so memory builds up as you chat. The
saved memories are recalled like any others; review and prune them anytime in
`/memories`.

```
/set memory_auto_extract true
```

The `lilbee memory` CLI group (`add` / `list` / `recall` / `remove`) and the
REST `/api/memories` routes operate on the same memories from outside the TUI.

| Setting | Default | Meaning |
|---------|---------|---------|
| `memory_enabled` | `false` | Master switch for the whole subsystem |
| `memory_auto_extract` | `false` | Background extraction after each TUI turn |
| `memory_top_k` | `5` | Max facts recalled per turn |
| `memory_max_distance` | `0.6` | Recall cutoff (lower is stricter) |
| `memory_token_budget` | `512` | Token cap on the injected memory block |
| `memory_max_per_owner` | `200` | Soft cap before oldest memories are evicted |

## Agent integration

lilbee is also the retrieval backend for AI coding agents. Wire it into any
agent that speaks MCP (Claude Code, opencode, Cursor, anything else) and the
agent calls `lilbee_search` / `lilbee_add` and gets back cited snippets it can
quote back. Your files, the embeddings, and the index stay on your computer.
See the [`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) for the full MCP
tool list and workflows. Non-MCP agents can use the [JSON CLI
fallback](#json-cli-fallback) below.

### Client config blocks

`lilbee agent-config <client>` prints a paste-ready config for one client,
filled in with the running server's address and session token:

```bash
lilbee agent-config claude     # Claude Code mcpServers block
lilbee agent-config opencode   # opencode.json provider + MCP server
lilbee agent-config hermes     # hermes config.yaml fragment
lilbee agent-config litellm    # LiteLLM proxy config.yaml snippet
```

opencode and hermes get lilbee as a model provider and as an MCP server.
Claude Code brings its own model, so it gets the MCP tools only. The claude
block registers the running server over HTTP; the alternative block printed on
stderr has Claude Code start `lilbee mcp` itself instead.

The same blocks come off the running server over HTTP, which is what a GUI
client should use. `GET /api/agent-config` lists the supported clients and
whether each one's CLI is installed on this machine:

```json
{
  "clients": [
    { "client": "claude", "cli_detected": true, "cli_path": "/opt/homebrew/bin/claude" },
    { "client": "hermes", "cli_detected": false, "cli_path": null },
    { "client": "opencode", "cli_detected": true, "cli_path": "/usr/local/bin/opencode" }
  ]
}
```

`GET /api/agent-config/<client>` returns that client's document. JSON clients
get it in `config`, hermes gets its rendered YAML in `content`, and claude also
carries the subprocess form in `stdio_config`. Both routes need the session
token like every other route. Fetch the document when you need it rather than
saving it: the token is minted fresh on every server start, so a copy written
to disk stops working after a restart.

```bash
curl -H "Authorization: Bearer $(jq -r .token <data-dir>/server.json)" \
  http://127.0.0.1:8080/api/agent-config/claude
```

### Serving a large agent fleet

One daemon serves many agents at once, and two limits bite before the GPU does.
Synchronous work is run off the event loop in a thread pool sized by
`mcp_tool_threads` (40 by default, which is what the async runtime uses when
nothing says otherwise); past that, retrieval calls queue while the disk and CPU
sit idle, so raise it when a lot of agents search at the same time. Each
connected agent also holds a socket, and macOS still defaults to 256 open files
(most Linux distributions to 1024), which shows up as connection failures rather
than slowness. Raise it in the shell you start the server from:

```bash
ulimit -n 4096
lilbee serve
```

The server logs the current limit at startup when it is low enough to matter.

### Agent memory

Agents get their own [memory](#memory) through the `lilbee_memory_remember` and
`lilbee_memory_recall` MCP tools (memory must be enabled first, e.g. the agent
calls `lilbee_settings_set({"memory_enabled": true})`). An agent's memories are
scoped to that agent and private by default: another agent won't see them, and
they're never folded into the human TUI's prompt. An agent can recall its own
memories plus any of yours you've marked shared.

Each agent's memories are namespaced by an owner like `agent:opencode`.
lilbee derives the agent id from the MCP client name when it can, but pin it
explicitly so it stays stable across sessions and clients. Set
`LILBEE_AGENT_ID` in the agent's MCP server config:

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "lilbee",
      "args": ["mcp"],
      "env": { "LILBEE_AGENT_ID": "opencode" }
    }
  }
}
```

```
lilbee_memory_remember({"text": "retrieval knobs live in core/config/model.py", "kind": "fact"})
# -> stored under owner agent:opencode
lilbee_memory_recall({"query": "where are retrieval settings"})
# -> returns that note next session; lilbee_search results never mix in memory
```

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
lilbee --json add ~/docs ~/notes               # register files / dirs, index them in place
lilbee --json add https://example.com/page     # URL becomes a markdown source
lilbee --json sync                             # re-index after edits to the documents directory
lilbee --json rebuild                          # nuke the index and re-ingest everything
lilbee --json remove manual.pdf                # drop chunks (source files are always kept)
lilbee --json remove reports/2024              # a folder: remove everything indexed beneath it
lilbee --json remove '**/*.log'                # a glob: remove every matching source
lilbee --json ask "question"                   # full local RAG (local llama-server fleet or remote backend)
lilbee --json model pull <ref>                 # download a model, streams JSON progress events
lilbee --json model pull <ref> --allow-unsupported  # override the architecture-compat check
lilbee --json model rm <ref>                   # delete an installed model
lilbee --json reset --yes                      # factory reset (destructive, requires --yes)
lilbee --json init [path]                      # create a .lilbee/ in a directory
```

`add` is the most common entry point: files, directories, and URLs all go
through it, and indexing happens in the same call. Long ops take seconds to
minutes; the final JSON includes per-file outcomes and counts.

**Wiki (opt-in):**

```bash
lilbee --json wiki status                      # page counts + wiki_enabled flag
lilbee --json wiki build                       # generate the topic / entity wiki
lilbee --json wiki update                      # refresh after a sync (full rebuild today)
lilbee --json wiki synthesize                  # cross-source synthesis pages
lilbee --json wiki lint                        # orphans, stale citations, pending drafts
lilbee --json wiki citations <page>            # citations a page makes
lilbee --json wiki citations --source <doc>    # pages citing a source document
lilbee --json wiki drafts list                 # pending drafts with drift + faithfulness
lilbee --json wiki drafts diff <slug>          # unified diff between a draft and the live page
lilbee --json wiki drafts accept <slug>        # publish a draft (concepts/, entities/, synthesis/, or summaries/)
lilbee --json wiki drafts reject <slug>        # discard a draft
lilbee --json wiki index                       # rebuild the browse index (no LLM call)
lilbee --json wiki generate <slug>             # write one indexed page (one LLM call)
lilbee --json wiki prune                       # archive stale pages
lilbee --json wiki wipe --yes                  # delete every generated page and its rows
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
{"config": {...}, "sources": [{"filename": "manual.pdf", "chunk_count": 42}], "total_chunks": 42, "skipped": [], "skipped_total": 0}

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
>    (native GGUF on the managed llama-server fleet, or a local
>    OpenAI-compatible server) if your documents must stay private.

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

## Keeping files out of the index

### What lilbee already excludes

Before you write a pattern, check what sync already excludes on its own:

- Formats it cannot read. lilbee indexes the file types it can extract,
  plus source code, and nothing else.
- Every dot-file and dot-directory, including `.git/`.
- A built-in directory list: `node_modules`, `__pycache__`, `venv`,
  `build`, `dist`, `target`, `vendor`, `_build`, `coverage`, `htmlcov`,
  and any name ending in `.egg-info`.

`LILBEE_IGNORE_DIRS` adds more directory names, comma-separated. It only
adds names. There is no way to take a built-in name off the list.

```bash
LILBEE_IGNORE_DIRS=generated,fixtures lilbee sync
```

### .lilbeeignore

For what that list misses, write a `.lilbeeignore`. The syntax is
`.gitignore`'s: `*` and `**` globs, a trailing `/` to match directories
only, and a leading `!` to add something back.

Use it when the problem is a supported file type rather than a directory
name: vendored JavaScript, test fixtures, generated Markdown, a
checked-in copy of `site-packages`.

lilbee never reads a repo's `.gitignore`. Repos commit binaries and
vendor trees, and a repo that excludes `docs/` from git holds the content
you most want indexed.

### Where to put it

**One file covers everything you index.** It sits in your data directory,
so in a `lilbee init` project it is `.lilbee/.lilbeeignore`, and
otherwise it is the global data directory. `lilbee init` writes a
commented copy for you.

**A `.lilbeeignore` inside a tree covers that directory and everything
below it.** Put one at the top of a repo to scope patterns to that repo,
or deeper to narrow them further.

The deeper file wins, so it can add back what the wider one excludes:

```
# .lilbee/.lilbeeignore -- everything you index
*.min.js
*.parquet

# ~/code/myrepo/.lilbeeignore -- this repo only
testdata/
!jquery.min.js
```

sync never indexes the ignore files themselves.

### sync does not walk an excluded directory

The `!` above works because `*.min.js` matches files. An excluded
*directory* behaves differently: sync never walks into it, so no pattern
below it can add a file back. `!testdata/keep.md` under `testdata/` does
nothing. This is git's rule.

To keep one file out of a tree you otherwise index, exclude the files
rather than the directory.

### Changing patterns later

The patterns govern what sync takes in. A pattern you add today stops
those files being indexed from now on; it does not touch what an earlier
sync already indexed. This matches `.gitignore`, which does not untrack
a file you have already committed.

To drop what a new pattern excludes, ask for it:

```bash
lilbee sync --prune-ignored
```

That removes the matching documents from the index and reports them as
`Removed`. Source files are never touched. The same option is on every
surface: `/prune-ignored` in the TUI, `{"prune_ignored": true}` to
`/api/sync`, and `prune_ignored=true` on `lilbee_sync`.

Nothing is written to hold a pruned file out. Delete the pattern and the
next sync indexes it again, so a prune you regret costs a re-index, not
the document.

lilbee always indexes a file you name yourself. `lilbee add
~/notes/report.pdf` beats a pattern that would exclude it, because
naming a file is the stronger instruction.

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
| `lilbee remove manual.pdf` | Remove from the index (source files are always kept) |
| `lilbee remove reports/2024` | Remove every document indexed under a folder |
| `lilbee remove '**/*.log'` | Remove every source matching a glob pattern |
| `lilbee chunks manual.pdf` | Inspect how a document was chunked |
| `lilbee sync` | Re-index changed files |
| `lilbee sync --prune-ignored` | Re-index, and drop documents a `.lilbeeignore` now excludes |
| `lilbee rebuild` | Nuke the database and re-ingest everything |
| `lilbee export pages.parquet` | Write a per-page text dataset (parquet or jsonl, no vectors) |
| `lilbee import pages.parquet` | Import a dataset, re-embedding it with the current model |
| `lilbee reset` | Factory reset. Deletes all documents and data |

### Wiki

```bash
lilbee wiki build                      # build the wiki from the current index
lilbee wiki build --dry-run            # preview the entity candidates, no LLM calls
lilbee wiki list                       # list pages with type, source count, date
lilbee wiki read <slug>                # print a page's markdown
lilbee wiki citations <page>           # citations a page makes
lilbee wiki citations --source <doc>   # pages citing a source document
lilbee wiki lint                       # find orphan pages, stale links, pending drafts
lilbee wiki synthesize                 # generate cross-source synthesis pages
lilbee wiki drafts list                # list pending drafts
lilbee wiki drafts accept <slug>       # publish a draft (concepts/, entities/, synthesis/, or summaries/)
lilbee wiki drafts reject <slug>       # discard a draft
lilbee wiki index                      # rebuild the browse index (no LLM call)
lilbee wiki generate <slug>            # write one indexed page (one LLM call)
lilbee wiki prune                      # move stale pages to archive/
lilbee wiki wipe                       # delete every generated page and its rows
```

MCP tools mirror the CLI: `wiki_list`, `wiki_read`, `wiki_status`,
`wiki_build`, `wiki_update`, `wiki_synthesize`, `wiki_lint`, `wiki_citations`,
`wiki_drafts_list`, `wiki_drafts_diff`, `wiki_prune`. Accepting and rejecting
drafts is left off MCP on purpose: deciding whether a low-confidence page is
good enough to publish is your call, so use the CLI, the TUI, or the
authenticated HTTP API.

### Memory

Requires memory enabled (`LILBEE_MEMORY_ENABLED=1`, or `/set memory_enabled true`
in the TUI). See
[Memory](#memory) for the full model.

```bash
lilbee memory add "the project uses rust"        # remember a fact
lilbee memory add "answer tersely" --preference  # remember a standing preference
lilbee memory add "uses rust" --shared           # also expose it to agents
lilbee memory list                               # show stored memories
lilbee memory recall "what language"             # recall facts by relevance
lilbee memory remove <id>                        # delete a memory by id
```

### Sessions

See [Sessions](#sessions). Ids accept any unique prefix.

```bash
lilbee sessions list                   # saved conversations, newest first
lilbee sessions show 3f2a              # print a transcript by id prefix
lilbee sessions rename 3f2a "Brake specs"
lilbee sessions delete 3f2a            # asks first; --yes skips the prompt
lilbee --json sessions show 3f2a       # transcript + compaction summary as JSON
```

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

One server runs per data dir: a second `lilbee serve` against the same
`--data-dir` waits up to fifteen seconds for the first to exit, then stops with an
error (exit code 3) instead of competing for the port file and the model engine. A supervisor
that manages several data dirs (the Obsidian plugin's shared root) can pass
`LILBEE_EXCLUSIVE_SCOPE=<dir>` to extend the same guarantee across all of them;
the refusal then names the data dir the running server is serving. To replace a
running server, ask it to stop with `POST /api/shutdown` (token-authed): it
shuts down exactly as an external SIGTERM would, and its locks release the
moment it exits.

Every route needs the session token, reads included, `GET /api/health` among
them; the daemon writes the token to `server.json` (mode `0600`) next to the
port file, and `lilbee agent-config` hands it to local clients. The surface
covers search (with SSE streaming variants for `ask` and `chat`),
document lifecycle, crawling, model management, memory
(`GET`/`POST`/`PATCH`/`DELETE /api/memories`, when memory is enabled),
saved conversations (`/api/sessions`: list, read, create, append, rename,
delete, and the compaction summary), configuration (including a defaults
endpoint that powers per-setting reset), and status/health. The
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
contains `documents/` (your indexed files), `data/` (the search index), a
`.gitignore` that excludes derived data, and a commented `.lilbeeignore` to
fill in.

## Config file

Every persisted lilbee setting lives in a single TOML file at
**`<data-dir>/config.toml`**. For the platform default that's
`~/.local/share/lilbee/config.toml`; for a project, it's `.lilbee/config.toml`
next to your code. Same file, same shape, regardless of which data dir you're
on; the path just follows wherever the data dir is resolved to.

Sections mirror the setting groups in the
[settings reference](settings.md): `[general]`, `[retrieval]`, `[chunking]`,
`[generation]`, `[server]`, etc. Example:

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

Every setting is also an environment variable, named `LILBEE_` plus the setting
in upper case: `top_k` is `LILBEE_TOP_K`, `wiki` is `LILBEE_WIKI`. An
environment variable applies to the process you launch and does not persist;
`/settings` and `config.toml` persist.

A handful of variables are not settings and so have no `config.toml` entry:
`LILBEE_DATA` (the data directory), `LILBEE_LOG_LEVEL`, `LILBEE_ENGINE_DIR`,
`LILBEE_TOKEN`, and a few ingest and debugging knobs. The settings reference
lists them under
[Environment-only variables](settings.md#environment-only-variables).

**[docs/settings.md](settings.md) lists every setting**, with its type, default,
help text, and a column per surface saying whether the TUI, MCP, HTTP API, or
CLI can change it. It is generated from lilbee's own configuration fields, so it
never drifts from the code.

## Optional extras

Extras exist only on a `pip` or `uv` install. If you installed a bundled build
(standalone binary, Homebrew, AUR, Nix, Docker, Flatpak, Snap, Scoop) all four
are already inside it and none of this applies to you.

`[engine]` is the `llama-server` that runs your models, and it is the one extra
that is not really optional. `[graph]`, `[crawler]` and `[litellm]` add
capabilities you can happily live without.

<details>
<summary><strong>pip / uv: installing the engine extra</strong></summary>

`[engine]` is not on PyPI. The CUDA and ROCm wheels run 444 MiB to 863 MiB each,
several times [PyPI's default 100 MiB per-file limit](https://pypi.org/help/#file-size-limit),
so every backend is published from lilbee's own
[PEP 503](https://peps.python.org/pep-0503/) package index at `lilbee.sh`. That
is what `--extra-index-url` is for. Install without it and lilbee still starts,
but the first call that needs a model fails with an error naming the engine and
repeating the command for your hardware.

The builds are not interchangeable, so pick the index for your hardware: `cpu` is
built with every GPU backend off, `metal` ships only a macOS arm64 wheel, and
`vulkan` only Linux and Windows ones.

```bash
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cu125/    # NVIDIA (CUDA)
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/rocm/     # AMD (ROCm)
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/metal/    # Apple silicon
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/vulkan/   # other GPUs
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cpu/      # no GPU
```

</details>

<details>
<summary><strong>pip / uv: the graph, crawler and litellm extras</strong></summary>

These pull heavier dependencies, so they are opt-in:

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

Install several at once, engine included:

```bash
pip install --pre 'lilbee[engine,graph,crawler,litellm]' --extra-index-url https://lilbee.sh/cu125/
uv tool install --prerelease=allow 'lilbee[engine,graph,crawler,litellm]' --extra-index-url https://lilbee.sh/cu125/
```

</details>

**NVIDIA users**: the default Vulkan build works, but the CUDA flavour is
faster and dodges the Vulkan-loader crash that affects NVIDIA-on-Windows
setups. Same `lilbee` command, links straight against your NVIDIA driver:

```bash
brew install tobocop2/lilbee/lilbee-cuda
paru -S lilbee-cuda
nix run github:tobocop2/lilbee#lilbee-cuda
pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cu125/   # devs
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
export LILBEE_CONCEPT_MAX_PER_CHUNK=5         # max concepts extracted per chunk (default)
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
(`/crawl`). The MCP tool defaults to a single page (`depth=0`); agents pass a
depth, or `depth=null` for the whole site.

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
managed local llama-server engine.

**What it does:** Routes chat and embedding calls to any provider reachable
via the SDK backend. The routing provider automatically detects which models
are available locally vs. remotely and routes each call to the right backend.

**When to use it:** When you want a frontier model for chat while keeping
embeddings local for privacy, or to surface models from a local
OpenAI-compatible daemon alongside lilbee's native GGUF models.

**Install:** `pip install --pre 'lilbee[litellm]'` or
`uv tool install --prerelease=allow 'lilbee[litellm]'`.

**Configuration:**

```bash
export LILBEE_LLM_PROVIDER=auto          # "auto" routes between local and remote
export LILBEE_OLLAMA_BASE_URL=http://localhost:11434  # Ollama URL (blank = default)
export LILBEE_LM_STUDIO_BASE_URL=http://localhost:1234/v1  # LM Studio URL (blank = default)
export LILBEE_LLM_API_KEY=sk-...         # API key for your provider
export LILBEE_CHAT_MODEL=your-model      # any remotely-supported model name
```

Provider options: `auto` (default; native GGUF models run on the local managed
llama-server fleet, remote model names route to the SDK backend) and `remote`
(everything goes to an external OpenAI-compatible endpoint).

---

## Tuning GPU placement

By default lilbee decides which GPUs each model goes on automatically. It
bin-packs all four roles (chat, embed, rerank, vision) across your available
GPUs and tensor-splits anything too large for one card.

If you want to pin specific models to specific cards, you can set a placement
spec. The spec is a JSON object with one entry per role you want to control:

```json
{
  "chat":   { "devices": [0, 1], "tensor_split": [1, 1] },
  "embed":  { "devices": [2] },
  "rerank": { "devices": [2] },
  "vision": { "devices": [3] }
}
```

Each role takes `devices` (GPU indices), an optional `tensor_split` list for
spreading a single model across cards, and an optional `replicas` count for the
embed and vision roles. Omit a role and the auto planner handles it.

**To see what the auto planner would assign** (without changing anything):

```bash
lilbee placement preview
```

**To apply a spec from a file:**

```bash
lilbee placement set --spec placement.json
```

**To see the current spec** (or confirm auto is active):

```bash
lilbee placement show
```

**To go back to automatic:**

```bash
lilbee placement clear
```

The same operations are available in the TUI under the Placement screen and over
MCP (`set_placement`, `clear_placement`). Applying or clearing placement rebuilds
the shared fleet, so over HTTP `PUT`/`DELETE /api/placement` are refused by
default. Set `allow_http_placement` (or `LILBEE_ALLOW_HTTP_PLACEMENT=1`) to enable
them on a single-client or owned deployment, such as the Obsidian plugin's managed
server or a personally-owned pod where you run `lilbee serve` yourself. The spec
persists across restarts.

If a pinned placement no longer fits the card it names (after a hardware
change, for example), lilbee surfaces an error naming the card rather than
starting in a broken state. `lilbee placement clear` returns to automatic
placement in that case.

### Memory budget

How much of each card, and of host RAM, the planner is allowed to promise.
Defaults suit most machines; these exist for the cases where they don't.

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `usable_vram_fraction` | `LILBEE_USABLE_VRAM_FRACTION` | `0.9` | Fraction of a card's **free** VRAM the planner may fill. The rest absorbs driver overhead and allocator fragmentation, which the engine's own report does not include. Range 0.5–1.0 |
| `system_memory_reserve_gb` | `LILBEE_SYSTEM_MEMORY_RESERVE_GB` | `4.0` | Host RAM held back from any model that offloads to system memory, so the OS and everything else keep room. Range 0–64 |
| `gpu_memory_fraction` | `LILBEE_GPU_MEMORY_FRACTION` | `0.75` | Fraction of the card chat sizes its KV cache against. Keep it at or below `usable_vram_fraction`; raising it past that only shrinks what chat is charged |

Raise `usable_vram_fraction` toward `1.0` on a dedicated box with nothing else
on the card. Lower it if loads fail with out-of-memory despite the planner
reporting the model fits.

Lower `system_memory_reserve_gb` on a machine with a lot of RAM and nothing else
running, if a model that offloads to host memory is being refused.

Both are writable at runtime through `/settings`, `lilbee config set`, and MCP.

**If a model is refused for memory it looks like it should have:** run
`lilbee placement preview` to see what the planner budgeted. The planner charges
free VRAM rather than the card's total, so another process holding memory shows
up here. On a partially offloaded model, check `system_memory_reserve_gb` too:
the host side is budgeted separately from the card.

**If a load dies of out-of-memory,** lilbee re-plans that role smaller and
retries rather than respawning the same launch. Repeated shrink-and-retry
messages for one model mean the estimate is too optimistic for that card;
lowering `usable_vram_fraction` fixes it more directly than pinning placement.

---

## Cross-encoder reranking

Built-in. Re-scores retrieval candidates with a cross-encoder for precision on
the top results. Unlike the extras above, no extra install is required;
reranking is off by default and turns on as soon as you set
`LILBEE_RERANKER_MODEL` (or pick a reranker from `/settings`).

**What it does:** After hybrid search (BM25 and vector arms fused into one
canonical relevance score) returns candidates, a GGUF cross-encoder scores
each `(query, chunk)` pair and results are blended with position-aware
weights. Top-ranked candidates keep more of the original ranking;
lower-ranked candidates trust the reranker more.

**When to use it:** When you need high-precision answers and are willing to
trade roughly 200 to 500 ms per query. Most useful with large candidate sets
where top-5 ordering matters.

**Configuration:**

```bash
export LILBEE_RERANKER_MODEL="bge-reranker-v2-m3"   # any GGUF reranker
export LILBEE_RERANK_CANDIDATES=20                  # how many candidates to rerank
```

Without a reranker set, fused hybrid search already provides good results for
most use cases.

Based on: Nogueira & Cho 2019 (Passage Re-ranking with BERT), Burges et al.
2005 (Learning to Rank).

---

## Semantic chunking

Off by default. lilbee ships with two chunking strategies; which one serves you
depends on what you're indexing.

**Fixed-size (default).** Breaks documents into roughly equal token windows
with overlap. Fast, deterministic, works well on code, reference manuals, user
guides, API specs, and anything with clear structural boundaries. The
assumption is that each chunk only needs to be coherent enough for retrieval,
and the model will handle the rest from a small window of context.

**Semantic.** Uses embedding similarity to detect topic
boundaries and splits there instead of at fixed sizes. Each chunk tends to
represent one coherent thought rather than an arbitrary slice through one. The
benefit shows up on prose-heavy material: novels, essays, long-form research
papers, interview transcripts, qualitative research notes, anything where an
argument develops across paragraphs. When you ask a question, the retrieved
chunk is more likely to contain the full section that matches rather than the
first half of it plus unrelated setup.

**Trade-off:** Enabling semantic chunking triggers a one-time download of
xberg's ONNX embedding model (separate from the chunk-to-vector embedder)
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
| **Languages** | 100+ via `LILBEE_OCR_LANGUAGE` (e.g. `eng+deu`) | Many scripts (model-dependent) |
| **Retrieval quality** | Fragments lose context | Chunks preserve semantic boundaries |
| **Install** | System package (`brew`/`apt`) | Native GGUF via the built-in mtmd backend, or any vision model reachable via the SDK backend (`pip install --pre 'lilbee[litellm]'` / `uv tool install --prerelease=allow 'lilbee[litellm]'`) |
| **Best for** | Simple text-only scans | Tables, multi-column layouts, formatted docs |

### Tesseract

[Tesseract](https://github.com/tesseract-ocr/tesseract) is used automatically
when no vision model is configured. No flags needed.

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

By default Tesseract reads English (`eng`). To OCR other languages, set the
language codes; xberg downloads the Tesseract data on first use. Join multiple
languages with `+` (Tesseract's own convention), for example `eng+deu` for
English and German. The setting is the same value everywhere:

```bash
export LILBEE_OCR_LANGUAGE=eng+deu   # environment, this process only
```

```toml
# .lilbee/config.toml -- persisted
ocr_language = "eng+deu"
```

You can also set it in the `/settings` screen in the TUI, or over the HTTP and
MCP settings APIs. Commas work in place of `+` if you prefer (`eng,deu`).

### Vision models

lilbee runs vision OCR in one of two ways:

1. **Local vision model.** Point `LILBEE_VISION_MODEL` at a GGUF vision model
   (e.g. `lightonocr`) and lilbee serves it on `llama-server` with an `--mmproj`
   projector. This is the recommended path and supports an SSE heartbeat for
   long scans.
2. **Remote vision model.** With `pip install --pre 'lilbee[litellm]'` (or
   `uv tool install --prerelease=allow 'lilbee[litellm]'`), set the vision
   model to any remote name your SDK backend understands. lilbee will route
   vision calls accordingly.

```bash
lilbee add report.pdf --vision                # prompts for model if none set
lilbee add report.pdf --vision-timeout 30     # per-page timeout (default: 120s, 0 = no limit)
export LILBEE_VISION_MODEL=lightonocr         # persist across runs (GGUF via mtmd)
```

Pick or change a vision model interactively via `/settings` in the
TUI; the selection is saved to `config.toml` and persists across sessions.
