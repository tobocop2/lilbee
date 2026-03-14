# lilbee for Obsidian

Search, ask, and chat with your vault's knowledge base — powered by [lilbee](https://github.com/tobocop2/lilbee) and local LLMs via [Ollama](https://ollama.com).

The plugin manages its own `.lilbee/` database inside your vault and starts/stops the lilbee server automatically. No terminal required.

## Demo

<details>
<summary><b>Scanned PDF → vision OCR → chat</b></summary>

Attaching a scanned 1998 Star Wars: X-Wing Collector's Edition manual (PDF with no extractable text), indexing it with vision OCR via [LightOnOCR-2](https://ollama.com/maternion/LightOnOCR-2), and chatting about the dev team credits — entirely local.

![Obsidian chat demo](demos/obsidian-chat.gif)
</details>

---

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Features](#features)
  - [Search](#search)
  - [Ask](#ask)
  - [Chat](#chat)
  - [Sync](#sync)
- [Settings](#settings)
  - [Server management](#server-management)
  - [Search & sync](#search--sync)
  - [Models](#models)
- [Supported formats](#supported-formats)
- [Vision OCR](#vision-ocr)
  - [Tesseract](#tesseract)
  - [Vision models](#vision-models)
- [Multiple vaults](#multiple-vaults)
- [Manual server mode](#manual-server-mode)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **[Ollama](https://ollama.com)** — local LLM runtime. The embedding model (`nomic-embed-text`) is auto-pulled on first sync. If no chat model is installed, lilbee prompts you to pick one.

2. **[lilbee](https://github.com/tobocop2/lilbee)** — the knowledge base engine.

   ```bash
   pip install lilbee        # or: uv tool install lilbee
   ```

3. **Optional — OCR for scanned PDFs and images:**
   - [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` / `apt install tesseract-ocr`) — free, fast, plain text output
   - Or an Ollama vision model (recommended for better quality) — see [Vision OCR](#vision-ocr)

> **First-time download:** Ollama models are large files downloaded once. For example, `qwen3:8b` is ~5 GB and `nomic-embed-text` is ~274 MB. After the initial download, models are cached locally and load in seconds. Check what's installed with `ollama list`.

## Quick start

1. Install and start **[Ollama](https://ollama.com)** (`ollama serve`)
2. Install the plugin (copy `main.js`, `manifest.json`, and `styles.css` into `.obsidian/plugins/lilbee/`)
3. Enable "lilbee" in Obsidian's community plugins settings
4. The plugin initializes `.lilbee/` in your vault root and starts the server automatically
5. Run **"lilbee: Sync vault"** from the command palette (`Ctrl/Cmd + P`) to index your vault
6. Run **"lilbee: Search knowledge base"** and start typing

That's it. The status bar shows server state: `starting...` → `ready`.

## Commands

Open the command palette (`Ctrl/Cmd + P`) and type "lilbee" to see all commands:

| Command | Description |
|---------|-------------|
| **lilbee: Search knowledge base** | Open the search modal with live results as you type |
| **lilbee: Ask a question** | Ask a natural language question — returns an answer with source citations |
| **lilbee: Open chat** | Open a chat sidebar with conversation history and streaming responses |
| **lilbee: Sync vault** | Sync vault documents to the knowledge base (add new, update changed, remove deleted) |
| **lilbee: Show status** | Show how many documents and chunks are indexed |

All commands can be bound to hotkeys in Obsidian's Hotkeys settings.

## Features

### Search

The search modal provides **instant semantic search** across your indexed vault. Start typing and results appear in real time (300ms debounce). Results are grouped by document with relevance-ranked excerpts showing page numbers and line ranges where applicable.

Click a source filename to open it in Obsidian.

### Ask

Ask a question and get a single answer synthesized from your vault's content, with source citations. The LLM reads the most relevant chunks and generates a grounded response.

### Chat

The chat sidebar provides a **multi-turn conversation** with your knowledge base:

- Streaming token-by-token responses
- Full conversation history within the session
- Source citations in expandable details under each response
- Clear chat button to start fresh
- Send with Enter, Shift+Enter for newlines

### Sync

Sync indexes your vault documents into the `.lilbee/` vector database:

- **Hash-based change detection** — only re-indexes files that changed
- **Progress tracking** — status bar shows current file and progress (`indexing 3/12 — notes.md`)
- **Summary notice** — shows counts of added, updated, removed, and failed files
- **Auto-sync mode** — optionally watch for file changes and sync automatically (see [Settings](#settings))

## Settings

Open Settings → Community plugins → lilbee to configure the plugin.

### Server management

| Setting | Default | Description |
|---------|---------|-------------|
| **Manage server** | On | Start and stop the lilbee server with Obsidian. When enabled, the plugin spawns `lilbee serve` on load and stops it on unload. |
| **Binary path** | (auto-detect) | Path to the `lilbee` binary. Leave empty to auto-detect from `$PATH`. Only shown when "Manage server" is on. |
| **Restart server** | — | Button to restart the managed server. Useful after changing models or if the server encounters an error. Only shown when "Manage server" is on. |

### Search & sync

| Setting | Default | Description |
|---------|---------|-------------|
| **Server URL** | `http://127.0.0.1:7433` | Address of the lilbee HTTP server. Auto-computed when "Manage server" is on. |
| **Ollama URL** | `http://127.0.0.1:11434` | Address of the Ollama server. Change this if Ollama runs on a different host or port. |
| **Results count** | 5 | Number of search results to return (1–20). |
| **Sync mode** | Manual | `Manual` — sync only via the command palette. `Auto` — watch for file create/modify/delete/rename and sync automatically after a debounce delay. |
| **Sync debounce** | 5000 | Delay in milliseconds before auto-sync triggers after a file change. Only shown when sync mode is "Auto". |

### Models

The Models section lets you manage chat and vision models directly from Obsidian settings:

- **Active model dropdown** — switch between installed models
- **Model catalog** — browse available models with size and description
- **Pull button** — download models from the Ollama registry (shows progress percentage)
- **Refresh** — reload the model list from the server

Chat models are used for the Ask and Chat features. Vision models are used for OCR on scanned PDFs and images (see [Vision OCR](#vision-ocr)).

## Supported formats

lilbee indexes a wide range of document and code formats. Text extraction is powered by [Kreuzberg](https://github.com/Goldziher/kreuzberg), code chunking by [tree-sitter](https://tree-sitter.github.io/tree-sitter/).

| Format | Extensions | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Extracts embedded text; falls back to OCR for scanned pages |
| Office | `.docx`, `.xlsx`, `.pptx` | |
| eBook | `.epub` | |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | Requires Tesseract or vision model |
| Data | `.csv`, `.tsv` | |
| Structured | `.xml`, `.json`, `.jsonl`, `.yaml`, `.yml` | Embedding-friendly preprocessing |
| Text | `.md`, `.txt`, `.html`, `.rst` | |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, [150+ more](https://github.com/Goldziher/tree-sitter-language-pack) | AST-aware chunking via tree-sitter |

See the [lilbee usage guide](https://github.com/tobocop2/lilbee/blob/main/docs/usage.md#adding-documents) for full details on format handling.

## Vision OCR

For PDFs without embedded text (scanned documents) and images, lilbee supports two OCR backends. When both are available, the vision model takes precedence.

### Tesseract

[Tesseract](https://github.com/tesseract-ocr/tesseract) is a free, fast OCR engine that produces plain text output.

Install:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
apt install tesseract-ocr

# Windows
choco install tesseract
```

Once installed, lilbee uses Tesseract automatically for scanned PDFs and images — no configuration needed.

### Vision models

Ollama vision models produce higher-quality OCR that preserves tables, headings, and layout as markdown. This is recommended over Tesseract for complex documents.

To enable vision OCR:

1. Open lilbee settings in Obsidian
2. Scroll to the **Models** section
3. In the **Vision Model** catalog, pull a vision model (e.g., `llava`, `minicpm-v`)
4. Select it from the "Active vision model" dropdown
5. Re-sync your vault to re-index scanned documents with vision OCR

Vision OCR is slower than Tesseract (each page is processed by the LLM) but significantly better for documents with structured content. See the [lilbee vision OCR benchmarks](https://github.com/tobocop2/lilbee/blob/main/docs/benchmarks/vision-ocr.md) for model comparisons.

You can configure a per-page timeout via the `LILBEE_VISION_TIMEOUT` environment variable (default: 120 seconds, set to `0` for no limit).

## Multiple vaults

Each vault gets its own isolated `.lilbee/` database and its own server instance. When multiple vaults are open simultaneously, the plugin assigns each vault a **deterministic port** derived from the vault path (range 7433–7932). This avoids port collisions without requiring configuration.

The port for a vault is always the same across restarts — it's computed from a hash of the vault's filesystem path.

## Manual server mode

If you prefer to run `lilbee serve` yourself (e.g., on a remote machine or with custom flags):

1. Disable **"Manage server"** in plugin settings
2. Start the server manually:
   ```bash
   cd /path/to/your/vault
   lilbee init
   lilbee serve --host 127.0.0.1 --port 7433
   ```
3. Set **"Server URL"** to match your server address

See the [lilbee agent integration docs](https://github.com/tobocop2/lilbee/blob/main/docs/agent-integration.md) for all server options.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Status bar shows "lilbee: ready (Ollama offline)" | Ollama is not running or unreachable. Start Ollama (`ollama serve`) or check the Ollama URL in settings. The plugin auto-detects Ollama every 30 seconds. |
| Status bar shows "lilbee: error" | Check that Ollama is running (`ollama serve`). Click the status bar or check Obsidian's developer console for details. |
| "lilbee not found" notice | Install lilbee (`pip install lilbee`) or set the binary path in settings. |
| Search returns no results | Run "lilbee: Sync vault" first to index your documents. |
| Sync fails | Ensure Ollama is running — sync needs the embedding model. Check the developer console (`Ctrl/Cmd + Shift + I`) for error details. |
| Port conflict | Disable "Manage server" and run the server manually on a different port. |
| Scanned PDFs not indexed | Install [Tesseract](#tesseract) or enable a [vision model](#vision-models). |
| Slow vision OCR | Vision OCR processes each page individually. Consider using a smaller/faster vision model, or increase the timeout via `LILBEE_VISION_TIMEOUT`. |

For issues with the plugin, [open an issue](https://github.com/tobocop2/lilbee/issues). For issues with lilbee itself, see the [lilbee repository](https://github.com/tobocop2/lilbee).

## License

MIT
