# Demos

GIFs and stills are hosted off `main` on the
[`gh-pages` branch](https://github.com/tobocop2/lilbee/tree/gh-pages/demos) and embedded
here via `raw.githubusercontent.com` URLs. Tape sources live next to this guide in
[`demos/`](../demos); re-render with `make demo-prep && make demo`.

## The TUI

The full terminal app: setup wizard, chat with cited answers, model catalog, settings,
task center, command palette, web crawl.

### Chat with cited answers

Ask the Crown Victoria owner's manual; every answer points back to the page. Inline `[N]`
markers are clickable in mouse-supporting terminals to open a source preview at the
exact passage.

![chat with cited answers](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

### Setup wizard

First-launch wizard. Pick a chat model and an embedding model from the curated list; both
download in parallel and you can keep working while they pull.

![setup wizard](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.gif)

### Add documents

`/add <path>` copies the file into the corpus and embeds it. Switching to the Task Center
mid-ingest shows the live progress bar; once it lands you can ask away.

![add and task center](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

### Model catalog

Browse models from Hugging Face Hub. Cycle the inner tabs (Discover / Chat / Embed /
Vision / Rerank / Library), toggle grid / list, scroll for more, search the picks, open
model info, pull a tiny model live.

![model catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

### Settings

Tabbed editor for every knob: Models, Ingest, Generation, Retrieval, Display, Crawling,
API-Keys, System.

![settings](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.gif)

### Command surface

`Ctrl+P` opens the Textual command palette; `?` toggles the keybinding cheat sheet;
`/help` opens the searchable slash-command catalog.

![command palette + help + slash catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-palette.gif)

### Crawl a URL

`/crawl <url>` fetches a page (or a small site) into the corpus, then you can ask
questions against it with the same cited-answer flow.

![crawl Wikipedia + cited answer](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-crawl.gif)

### Tour

A quick sweep through every screen.

![tour](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif)

## The CLI

What the Ollama CLI is good for: `init`, `model pull`, `model list`, `add`, `status`,
`sync`, `search`. No JSON RAG.

![CLI tour](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/cli.gif)

## lilbee over MCP

Two demos of [opencode](https://opencode.ai) driving lilbee entirely via MCP. Each demo
project ships with a drop-in [`AGENTS.md`](../demos/AGENTS.md), a
[`lilbee-worker` subagent](../demos/.opencode/agents/lilbee-worker.md) that handles the
long-running ops (`lilbee_add`, `lilbee_sync`, `lilbee_crawl`, `lilbee_model_pull`), and
the [`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) (opencode / Claude Skill
format) that documents every MCP tool with a quick-vs-long-cost split. The agent runs on
the dev's default cloud model (MiniMax M2.7); the lilbee corpus stays local.

What you'll see on screen, inline:

- `# lilbee_<tool>` for each tool call, with the query the agent picked.
- `Lilbee-Worker Task — Index ...` whenever indexing is delegated to the subagent.
- The cited answer (`cv-manual.pdf, page N` or `src/lilbee/.../file.py:LINE-RANGE`).

### Indexing and querying an owner's manual

The agent finds `cv-manual.pdf` in the project, delegates the index to `lilbee-worker`,
then `lilbee_search`-es for the oil capacity and returns a page-cited answer.

![mcp + manual](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-manual.gif)

### Indexing and querying lilbee's own source

Same shape against a code corpus. The agent indexes a slice of `src/lilbee/`, then
`lilbee_search`-es for the `.lilbee/` discovery mechanism and cites the actual file and
line range (`src/lilbee/core/system.py:23-33`).

![mcp + lilbee source](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/mcp-code.gif)

## Written walkthroughs

For longer side-by-side comparisons and benchmarks, see [`docs/benchmarks/`](benchmarks/):

- [Godot level generator](benchmarks/godot-level-generator.md): lilbee as a retrieval
  backend for an AI coding agent, with a side-by-side comparison of API hallucinations
  vs. a pure web-search baseline.
- [Vision OCR model comparison](benchmarks/vision-ocr.md): output quality and retrieval
  quality across vision OCR backends on a scanned PDF.
