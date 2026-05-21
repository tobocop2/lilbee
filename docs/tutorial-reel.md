# Tutorial reel

Every GIF in the [README](../README.md) (plus the extras that don't fit there) as
an embedded video, with long-form captions. Same demos as the
[lilbee.sh/](https://lilbee.sh/) reel; the captions here are tutorial-shaped
(what you're seeing, why it matters) rather than the website's one-liners.

The nine that match the site reel, in order:

1. [First run](#first-run)
2. [TUI tour](#tui-tour)
3. [Chat with cited answers](#chat-with-cited-answers)
4. [Add files](#add-files)
5. [Crawl a URL](#crawl-a-url)
6. [Model catalog](#model-catalog)
7. [Settings](#settings)
8. [Agent: code (lilbee talking to lilbee)](#agent-code-lilbee-talking-to-lilbee)
9. [Agent: PDF](#agent-pdf)

Extras: [agent: live indexing](#agent-live-indexing), [agent: Godot codegen against the full class reference](#agent-godot-codegen-against-the-full-class-reference), [command surface](#command-surface).

## First run

First-launch wizard. Pick a chat model and an embedding model from the curated list;
both download in parallel and you can keep working while they pull.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.mp4" type="video/mp4">
  first-run wizard
</video>

## TUI tour

A quick sweep through every screen.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.mp4" type="video/mp4">
  tour
</video>

## Chat with cited answers

Ask the Crown Victoria owner's manual; every answer points back to the page. Inline
`[N]` markers are clickable in mouse-supporting terminals to open a source preview
at the exact passage.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.mp4" type="video/mp4">
  chat with cited answers
</video>

## Add files

`/add <path>` copies the file into your library and embeds it. Switching to the Task
Center mid-ingest shows the live progress bar; once it lands you can ask away.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.mp4" type="video/mp4">
  add and task center
</video>

## Crawl a URL

`/crawl <url>` fetches a page (or a small site) into your library, then you can ask
questions against it with the same cited-answer flow.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.mp4" type="video/mp4">
  crawl Wikipedia + cited answer
</video>

## Model catalog

Browse models from Hugging Face Hub. Cycle the inner tabs (Discover / Chat / Embed /
Vision / Rerank / Library), toggle grid / list, scroll for more, search the picks,
open model info, pull a tiny model live.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.mp4" type="video/mp4">
  model catalog
</video>

## Unsupported architectures, surfaced before the download

Hugging Face has thousands of GGUFs but the bundled llama.cpp build only supports a
subset of architectures. The catalog reads `general.architecture` from the Hub for
every row and tags ones that the runtime can't load with an `unsupported` pill on
the card and an italic tag in the list view. The clip searches HF Hub for
`gemma-4` (an architecture upstream hasn't shipped yet) and shows the rows
surfacing with the pill in both grid and list view. Trying to install one opens a
confirm dialog ("pull anyway?") so a heavy download never starts by accident.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-unsupported.mp4" type="video/mp4">
  unsupported architectures
</video>

## Settings

Tabbed editor for every knob: Models, Ingest, Generation, Retrieval, Display,
Crawling, API-Keys, System.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.mp4" type="video/mp4">
  settings
</video>

## Agent: code (lilbee talking to lilbee)

The headline grounding demo. An agent indexes lilbee's own source through lilbee's
MCP server, then answers questions about how lilbee works, citing
`src/lilbee/.../file.py:LINE` for every claim.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code.mp4" type="video/mp4">
  an agent indexes lilbee's own source through lilbee's MCP server, then answers questions about how lilbee works with file:line citations
</video>

## Agent: code self-tune (outline → source)

A two-turn variant of the above. Turn 1 asks how `lilbee_search` works end to end;
the agent at the OLD retrieval defaults answers with a structured ten-step outline
that names real lilbee methods but doesn't pull in source. Turn 2 asks for the
actual code: the agent calls `lilbee_settings_set` to widen retrieval, re-runs
`lilbee_search` against the richer pool, and answers with full function bodies
pasted inline, each followed by a `file.py:L<start>-L<end>` citation. The whole
self-tune loop runs end-to-end on a local 8B model.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code-self-tune.mp4" type="video/mp4">
  agent self-tunes lilbee mid-conversation: outline → settings_set → re-search → source with file:line citations
</video>

## Agent: PDF

The agent finds `cv-manual.pdf` in the project, delegates the index to
`lilbee-worker`, then `lilbee_search`-es and returns a page-cited answer.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-manual.mp4" type="video/mp4">
  mcp + manual
</video>

## Extras

These don't have a tab on the site, but they're part of the same reel.

### Agent: live indexing

The smaller agent-over-MCP demo. An MCP-aware coding agent indexes a Godot 4
pathfinding subset in a few seconds, then `lilbee_search`-es for `AStarGrid2D` and
answers method-by-method against the local files.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot-search.mp4" type="video/mp4">
  an MCP-driven coding agent indexes a small local godot subset and answers with cited methods
</video>

### Agent: Godot codegen against the full class reference

Same shape against a full XML reference library. The agent indexes Godot 4's class
reference (810 XML files, 3449 chunks) via `lilbee-worker`, then `lilbee_search`-es
for `AStarGrid2D`, `TileMap`, `RandomNumberGenerator`, and friends as it writes a
procedural level generator. Every API call is backed by a
`godot-classes/<Class>.xml:line` citation. See
[`docs/benchmarks/godot-level-generator.md`](benchmarks/godot-level-generator.md)
for the side-by-side against a no-RAG baseline (4 hallucinated APIs without
lilbee, 0 with).

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.mp4" type="video/mp4">
  mcp + godot class reference
</video>

### Command surface

`Ctrl+P` opens the Textual command palette; `?` toggles the keybinding cheat
sheet; `/help` opens the searchable slash-command catalog.

<video controls muted playsinline width="100%" preload="metadata">
  <source src="https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.mp4" type="video/mp4">
  command palette + help + slash catalog
</video>

## Agent setup

Each agent demo ships with a drop-in [`AGENTS.md`](../examples/agent-integration/AGENTS.md), a
[`lilbee-worker` subagent](../examples/agent-integration/.opencode/agents/lilbee-worker.md) that
handles the long-running ops (`lilbee_add`, `lilbee_sync`, `lilbee_crawl`,
`lilbee_model_pull`), and the
[`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) (opencode / Claude Skill
format) that documents every MCP tool with a quick-vs-long split. The agent runs
on the dev's default cloud model (MiniMax M2.7); the lilbee library stays local.
On screen, inline: `# lilbee_<tool>` for each tool call,
`Lilbee-Worker Task: Index ...` whenever indexing is delegated to the subagent,
and the cited answer.

## Written walkthroughs

For longer side-by-side comparisons and benchmarks, see
[`docs/benchmarks/`](benchmarks/):

- [Godot level generator](benchmarks/godot-level-generator.md): lilbee as a
  retrieval backend for an AI coding agent, with a side-by-side against a pure
  web-search baseline.
- [Vision OCR model comparison](benchmarks/vision-ocr.md): output quality and
  retrieval quality across vision OCR backends on a scanned PDF.

---

GIFs, stills, and tape sources all live on the
[`gh-pages` branch](https://github.com/tobocop2/lilbee/tree/gh-pages),
embedded here via `raw.githubusercontent.com` URLs. Tape sources are at
[`demos-src/`](https://github.com/tobocop2/lilbee/tree/gh-pages/demos-src);
re-render with `make demo-prep && make demo` (the targets are a thin
wrapper that runs the pipeline in a `gh-pages` worktree).
