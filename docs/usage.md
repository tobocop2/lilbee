# Usage Guide

- [Getting started](#getting-started)
- [Adding documents](#adding-documents)
- [OCR](#ocr)
- [Querying](#querying)
- [Interactive chat](#interactive-chat)
- [Managing documents](#managing-documents)
- [Agent integration](#agent-integration)
- [Data locations](#data-locations)
- [Environment variables](#environment-variables)

---

## Getting started

lilbee uses a git-like per-project model. Running `lilbee init` creates a `.lilbee/` directory in the current folder — just like `git init` creates `.git/`. Once initialized, every lilbee command you run from that directory (or any subdirectory) automatically uses the local database:

```bash
cd ~/projects/my-engine
lilbee init                  # creates .lilbee/ here
lilbee add docs/manual.pdf   # indexes into .lilbee/
lilbee search "oil change"   # searches .lilbee/
```

If there's no `.lilbee/` in the current directory, lilbee walks up the directory tree looking for one — again, just like git. If none is found, it falls back to a global database at the platform default location (see [Data locations](#data-locations)).

This means running `lilbee` without `init` still works — it just uses the global database. Use `lilbee status` to see which database is active:

```bash
lilbee status
```

To explicitly skip any local `.lilbee/` and use the global database:

```bash
lilbee --global status
```

## Adding documents

Add files, directories, or a mix:

```bash
lilbee add ~/Documents/manual.pdf
lilbee add ~/notes/
lilbee add ~/docs/*.md ~/data/report.pdf
```

If a file with the same name already exists in the knowledge base, `add` skips it. Use `--force` to overwrite:

```bash
lilbee add manual.pdf --force
```

## OCR

For PDFs without embedded text, lilbee supports two OCR backends. When a vision model is configured, it takes precedence.

| | Tesseract | Vision model |
|---|---|---|
| **Output** | Plain text | Structured markdown (tables, headings) |
| **Retrieval quality** | Fragments lose context | Chunks preserve semantic boundaries |
| **Install** | System package (`brew`/`apt`) | Ollama + model (~1.5 GB for [LightOnOCR-2](https://ollama.com/maternion/LightOnOCR-2)) |
| **Best for** | Simple text-only scans | Tables, multi-column layouts, formatted docs |

See [model benchmarks](benchmarks/vision-ocr.md) for detailed comparisons.

### Tesseract

[Tesseract](https://github.com/tesseract-ocr/tesseract) is used automatically when no vision model is configured — no flags needed.

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### Vision models

Requires [Ollama](https://ollama.com/) with a vision-capable model.

```bash
lilbee add report.pdf --vision                # prompts for model if none set
lilbee add report.pdf --vision-timeout 30     # per-page timeout (default: 120s, 0 = no limit)
export LILBEE_VISION_MODEL=maternion/LightOnOCR-2  # persist across runs
```

In interactive chat:

```
/vision                          # show status + picker
/vision maternion/LightOnOCR-2   # set directly
/vision off                      # disable and clear saved model
```

The model is saved to `config.toml` and persists across sessions.

## Querying

Search returns relevant chunks from your indexed documents. No LLM needed — `search` works without Ollama running:

```bash
lilbee search "oil change interval"
lilbee search "oil change interval" --top-k 20   # more results
```

Ask a one-shot question — lilbee finds relevant chunks and passes them to a local LLM:

```bash
lilbee ask "What is the recommended oil change interval?"
lilbee ask "Explain this" --model qwen3           # different chat model
```

## Interactive chat

Run `lilbee` or `lilbee chat` to enter an interactive REPL with conversation history, streaming responses, and slash commands:

```bash
lilbee
```

### Slash commands

| Command | Description |
|---------|-------------|
| `/status` | Show indexed documents and config |
| `/add [path]` | Add a file or directory (tab-completes paths) |
| `/model [name]` | Switch chat model — no args opens an interactive picker (tab-completes installed models) |
| `/vision [name\|off]` | Switch vision OCR model — no args opens a picker, `off` disables |
| `/settings` | Show all current configuration values |
| `/set <key> <value>` | Change a setting (e.g. `/set temperature 0.7`) |
| `/version` | Show lilbee version |
| `/reset` | Delete all documents and data (asks for confirmation) |
| `/help` | Show available commands |
| `/quit` | Exit chat |

Slash commands and paths tab-complete. A spinner shows while waiting for the first token from the LLM.

## Managing documents

| Command | Description |
|---------|-------------|
| `lilbee remove manual.pdf` | Remove from knowledge base (keeps source file) |
| `lilbee remove manual.pdf --delete` | Remove and delete the source file |
| `lilbee chunks manual.pdf` | Inspect how a document was chunked |
| `lilbee sync` | Re-index changed files |
| `lilbee rebuild` | Nuke the database and re-ingest everything |
| `lilbee reset` | Factory reset — deletes all documents and data |

## Agent integration

lilbee works as a retrieval backend for AI coding agents via MCP or JSON CLI. See [agent-integration.md](agent-integration.md) for setup and [demos/godot-with-lilbee/](../demos/godot-with-lilbee/) for a real-world example.

> [!CAUTION]
> **Private data and cloud agents**
>
> When an agent queries lilbee, retrieved chunks are sent to whatever LLM the agent uses — including cloud-hosted models. If your knowledge base contains private, confidential, or sensitive documents, verify two things before connecting an agent:
>
> 1. **Check which database is active** — run `lilbee status` and confirm the data directory is the one you intend the agent to access. lilbee walks up the directory tree to find `.lilbee/`, so you may be exposing a different project's data than you expect.
> 2. **Know where your agent sends data** — if the agent uses a cloud-hosted model, your document chunks will leave your machine. Use a local model via Ollama if your documents must stay private.

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

Run `lilbee init` to create a `.lilbee/` directory in your project. It contains `documents/`, `data/`, and a `.gitignore` that excludes derived data.

## Environment variables

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path |
| `LILBEE_CHAT_MODEL` | `qwen3:8b` | Ollama chat model |
| `LILBEE_VISION_MODEL` | *(none)* | Vision model for PDF OCR — when set, takes precedence over Tesseract |
| `LILBEE_VISION_TIMEOUT` | `120` | Per-page vision OCR timeout in seconds (`0` = no limit) |
| `LILBEE_TOP_K` | `10` | Number of retrieval results |
| `LILBEE_LOG_LEVEL` | `WARNING` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | Custom system prompt for RAG answers |

**Server:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_SERVER_HOST` | `127.0.0.1` | Server bind address |
| `LILBEE_SERVER_PORT` | `7433` | Server port |
| `LILBEE_CORS_ORIGINS` | *(none)* | Comma-separated list of allowed CORS origins (e.g. `app://obsidian.md,https://my-app.com`). Localhost is always allowed. |

**Generation** — tune LLM output:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_TEMPERATURE` | *(model default)* | Sampling temperature |
| `LILBEE_TOP_P` | *(model default)* | Nucleus sampling threshold |
| `LILBEE_TOP_K_SAMPLING` | *(model default)* | Top-k sampling |
| `LILBEE_REPEAT_PENALTY` | *(model default)* | Repetition penalty |
| `LILBEE_NUM_CTX` | *(model default)* | Context window size |
| `LILBEE_SEED` | *(model default)* | Random seed for reproducibility |

**Advanced** — most users won't need to change these:

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding dimensions (must match model) |
| `LILBEE_CHUNK_SIZE` | `512` | Tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk for embedding |

CLI flags: `--model` / `-m`, `--data-dir` / `-d`, `--global` / `-g`, `--vision`, `--vision-timeout`, `--log-level`, `--json` / `-j`, `--version` / `-V`.
