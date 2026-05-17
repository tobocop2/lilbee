# The full reel

Same demos as on [lilbee.sh/](https://lilbee.sh/),
with the captions long-form and a handful of extras that don't fit in the site's
tab list.

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

![first-run wizard](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.gif)

## TUI tour

A quick sweep through every screen.

![tour](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif)

## Chat with cited answers

Ask the Crown Victoria owner's manual; every answer points back to the page. Inline
`[N]` markers are clickable in mouse-supporting terminals to open a source preview
at the exact passage.

![chat with cited answers](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

## Add files

`/add <path>` copies the file into your library and embeds it. Switching to the Task
Center mid-ingest shows the live progress bar; once it lands you can ask away.

![add and task center](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

## Crawl a URL

`/crawl <url>` fetches a page (or a small site) into your library, then you can ask
questions against it with the same cited-answer flow.

![crawl Wikipedia + cited answer](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.gif)

## Model catalog

Browse models from Hugging Face Hub. Cycle the inner tabs (Discover / Chat / Embed /
Vision / Rerank / Library), toggle grid / list, scroll for more, search the picks,
open model info, pull a tiny model live.

![model catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

## Settings

Tabbed editor for every knob: Models, Ingest, Generation, Retrieval, Display,
Crawling, API-Keys, System.

![settings](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.gif)

## Agent: self-tune (a better answer in one settings_set)

Same question asked twice against the godot class reference: with
defaults `lilbee_search "A star pathfinding"` returns AStar3D as the
only A* class. One `lilbee_settings_set diversity_max_per_source=8
top_k=15 max_distance=0.85` and the same query returns AStar3D plus
AStarGrid2D plus the NavigationPathQueryParameters family; the
re-answer cites multiple A* variants. Settings reset at the end leaves
the env clean.

![an opencode agent asks the same A-star pathfinding question against godot-classes twice, batches one lilbee_settings_set in between, and the tuned answer cites multiple A* variants instead of just AStar3D](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-self-tune.gif)

## Agent: code (lilbee talking to lilbee)

The headline grounding demo. An agent indexes lilbee's own source through lilbee's
MCP server, then answers questions about how lilbee works, citing
`src/lilbee/.../file.py:LINE` for every claim.

![an agent indexes lilbee's own source through lilbee's MCP server, then answers questions about how lilbee works with file:line citations](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code.gif)

## Agent: PDF

The agent finds `cv-manual.pdf` in the project, delegates the index to
`lilbee-worker`, then `lilbee_search`-es and returns a page-cited answer.

![mcp + manual](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-manual.gif)

## Extras

These don't have a tab on the site, but they're part of the same reel.

### Agent: live indexing

The smaller agent-over-MCP demo. An MCP-aware coding agent indexes a Godot 4
pathfinding subset in a few seconds, then `lilbee_search`-es for `AStarGrid2D` and
answers method-by-method against the local files.

![an MCP-driven coding agent indexes a small local godot subset and answers with cited methods](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot-search.gif)

### Agent: Godot codegen against the full class reference

Same shape against a full XML reference library. The agent indexes Godot 4's class
reference (810 XML files, 3449 chunks) via `lilbee-worker`, then `lilbee_search`-es
for `AStarGrid2D`, `TileMap`, `RandomNumberGenerator`, and friends as it writes a
procedural level generator. Every API call is backed by a
`godot-classes/<Class>.xml:line` citation. See
[`docs/benchmarks/godot-level-generator.md`](benchmarks/godot-level-generator.md)
for the side-by-side against a no-RAG baseline (4 hallucinated APIs without
lilbee, 0 with).

![mcp + godot class reference](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-godot.gif)

### Command surface

`Ctrl+P` opens the Textual command palette; `?` toggles the keybinding cheat
sheet; `/help` opens the searchable slash-command catalog.

![command palette + help + slash catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.gif)

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
