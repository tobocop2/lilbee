# Demos

GIFs and stills are hosted off `main` on the
[`gh-pages` branch](https://github.com/tobocop2/lilbee/tree/gh-pages/demos) and embedded
here via `raw.githubusercontent.com` URLs. Tape sources live next to this guide in
[`demos/`](../demos); re-render with `make demo-prep && make demo`.

## The TUI

The full terminal app: setup wizard, chat with cited answers, model catalog, settings,
task center.

### Chat with cited answers

Ask the Crown Victoria owner's manual; every answer points back to the page.

![chat with cited answers](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-chat.gif)

### Setup wizard

Pick a chat model and an embedding model from the catalog; both download in the
background.

![setup wizard](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-setup.gif)

### Add documents

`/add <path>` copies the file into the corpus and embeds it; `/status` confirms.

![add and status](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-add.gif)

### Model catalog

Browse models from Hugging Face Hub. Toggle grid / list, search the picks, open model info
on a row.

![model catalog](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-catalog.gif)

### Settings

Tabbed editor for the 50+ knobs: chat / embedding / vision / reranker models, ingest,
generation, retrieval, display, system.

![settings](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-settings.gif)

### Tour

A quick sweep through the screens.

![tour](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/tui-tour.gif)

## The CLI

What the Ollama CLI is good for: `init`, `model pull`, `model list`, `add`, `status`,
`sync`. No JSON RAG.

![CLI tour](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/cli.gif)

## opencode + lilbee over MCP

Two demos of [opencode](https://opencode.ai) driving lilbee entirely via MCP. The opencode
project ships with a drop-in [`AGENTS.md`](../demos/AGENTS.md), a
[`lilbee-worker` subagent](../demos/.opencode/agents/lilbee-worker.md) that handles the
long-running ops (`lilbee_add`, `lilbee_sync`, `lilbee_crawl`, `lilbee_model_pull`), and the
[`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) (opencode / Claude Skill format)
that documents every MCP tool with a quick-vs-long-cost split. Both demos use
`opencode/deepseek-v4-flash-free` as the cloud model; the lilbee corpus stays local.

What you'll see on screen, inline:

- `⚙ lilbee_status` and `⚙ lilbee_search {"query":..., "top_k":N}` for each tool call,
  with the actual query the agent picked.
- `• Index ... ✓ Index ... Lilbee-Worker Agent` whenever indexing is delegated to the
  subagent.
- The cited answer (`file.pdf, page N` or `src/lilbee/.../file.py:LINE-RANGE`).

### Indexing and querying an owner's manual

The agent finds `cv-manual.pdf` in the project, delegates the index to `lilbee-worker`,
then `⚙ lilbee_search`-es for the oil capacity and returns a page-cited answer.

![opencode + manual](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/opencode-manual.gif)

### Indexing and querying lilbee's own source

Same shape against a code corpus. The agent indexes a slice of `src/lilbee/`, then
`⚙ lilbee_search`-es for the `.lilbee/` discovery mechanism and cites the actual file and
line range (`src/lilbee/core/system.py:23-33`).

![opencode + lilbee source](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/opencode-code.gif)

## Written walkthroughs

For longer side-by-side comparisons and benchmarks, see [`docs/benchmarks/`](benchmarks/):

- [Godot level generator](benchmarks/godot-level-generator.md): lilbee as a retrieval
  backend for an AI coding agent, with a side-by-side comparison of API hallucinations
  vs. a pure web-search baseline.
- [Vision OCR model comparison](benchmarks/vision-ocr.md): output quality and retrieval
  quality across vision OCR backends on a scanned PDF.
