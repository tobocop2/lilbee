# Usage Guide

- [Getting started](#getting-started)
- [Adding documents](#adding-documents)
- [OCR](#ocr)
- [Querying](#querying)
- [Interactive chat](#interactive-chat)
- [Managing documents](#managing-documents)
- [Wiki](#wiki)
- [Agent integration](#agent-integration)
- [HTTP Server](#http-server)
- [Data locations](#data-locations)
- [Environment variables](#environment-variables)
- [Optional extras](#optional-extras)
  - [Concept graph](#concept-graph)
  - [Web crawling](#web-crawling)
  - [Remote providers (SDK backend)](#remote-providers-sdk-backend)
- [Cross-encoder reranking](#cross-encoder-reranking)
- [Semantic chunking](#semantic-chunking)

---

## Getting started

lilbee uses a git-like per-project model. Running `lilbee init` creates a `.lilbee/` directory in the current folder, just like `git init` creates `.git/`. Once initialized, every lilbee command you run from that directory (or any subdirectory) automatically uses the local database:

```bash
cd ~/projects/my-engine
lilbee init                  # creates .lilbee/ here
lilbee add docs/manual.pdf   # indexes into .lilbee/
lilbee search "oil change"   # searches .lilbee/
```

If there's no `.lilbee/` in the current directory, lilbee walks up the directory tree looking for one (again, just like git). If none is found, it falls back to a global database at the platform default location (see [Data locations](#data-locations)).

This means running `lilbee` without `init` still works; it just uses the global database. Use `lilbee status` to see which database is active:

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

If a file with the same name is already indexed, `add` skips it. Use `--force` to overwrite:

```bash
lilbee add manual.pdf --force
```

## OCR

For PDFs without embedded text, lilbee supports two OCR backends. When a vision model is configured, it takes precedence.

| | Tesseract | Vision model |
|---|---|---|
| **Output** | Plain text | Structured markdown (tables, headings) |
| **Retrieval quality** | Fragments lose context | Chunks preserve semantic boundaries |
| **Install** | System package (`brew`/`apt`) | Native GGUF via the built-in mtmd backend, or any vision model reachable via the SDK backend (`pip install --pre 'lilbee[litellm]'` / `uv tool install --prerelease=allow 'lilbee[litellm]'`) |
| **Best for** | Simple text-only scans | Tables, multi-column layouts, formatted docs |

See [model benchmarks](benchmarks/vision-ocr.md) for detailed comparisons.

### Tesseract

[Tesseract](https://github.com/tesseract-ocr/tesseract) is used automatically when no vision model is configured. No flags needed.

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### Vision models

lilbee runs vision OCR in one of two ways:

1. **Native mtmd backend.** Point `LILBEE_VISION_MODEL` at a GGUF vision model
   (e.g. `lightonocr`) and lilbee will load it with llama-cpp's mtmd backend
   directly. No Ollama, no extra services. This is the recommended path and
   supports an SSE heartbeat for long scans.
2. **Remote vision model.** With `pip install --pre 'lilbee[litellm]'` (or
   `uv tool install --prerelease=allow 'lilbee[litellm]'`), set the vision
   model to any remote name your SDK backend understands (Ollama, OpenAI,
   Anthropic, Gemini, etc.). lilbee will route vision calls accordingly.

```bash
lilbee add report.pdf --vision                # prompts for model if none set
lilbee add report.pdf --vision-timeout 30     # per-page timeout (default: 120s, 0 = no limit)
export LILBEE_VISION_MODEL=lightonocr         # persist across runs (GGUF via mtmd)
```

Pick or change a vision model interactively via `/settings` or `/setup` in the
TUI; the selection is saved to `config.toml` and persists across sessions.

## Querying

Search returns relevant chunks from your indexed documents. No LLM needed; `search` works without any model loaded:

```bash
lilbee search "oil change interval"
lilbee search "oil change interval" --top-k 20   # more results
```

Ask a one-shot question. lilbee finds relevant chunks and passes them to the configured chat model:

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

All slash commands available from the TUI:

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

Slash commands and paths tab-complete. A spinner shows while waiting for the
first token from the LLM. Background jobs (sync, crawl, wiki build, model pull)
appear in the Task Center and are cancellable with `/cancel`.

## Managing documents

| Command | Description |
|---------|-------------|
| `lilbee remove manual.pdf` | Remove from the index (keeps source file) |
| `lilbee remove manual.pdf --delete` | Remove and delete the source file |
| `lilbee chunks manual.pdf` | Inspect how a document was chunked |
| `lilbee sync` | Re-index changed files |
| `lilbee rebuild` | Nuke the database and re-ingest everything |
| `lilbee reset` | Factory reset. Deletes all documents and data |

## Wiki

lilbee analyzes the documents you've indexed and writes a wiki about them,
inspired by Andrej Karpathy's [LLM Wiki](https://karpathy.ai/llmwiki/). Pages
compound across sources instead of being one-per-document, so concepts and
entities that show up repeatedly in your corpus get their own page with
citations from every source that mentions them.

**Layout** (under `$LILBEE_DATA/wiki/` by default):

| Directory | Contents |
|-----------|----------|
| `concepts/` | One page per LLM-identified concept (e.g. `braking-systems.md`) |
| `entities/` | One page per proper-noun entity extracted by NER (e.g. `henry-ford.md`) |
| `drafts/` | Low-faithfulness or parse-failure pages awaiting your accept/reject |
| `archive/` | Pages retired by `lilbee wiki prune` |
| `synthesis/` | Cross-source pages produced by `lilbee wiki synthesize` |
| `index.md` | Auto-generated table of contents, grouped by page type |
| `log.md` | Append-only audit trail of every build, ingest, lint, and prune |

**Commands:**

```bash
lilbee wiki build         # build the wiki from the current index
lilbee wiki lint          # find orphan pages, stale links, pending drafts
lilbee wiki synthesize    # generate cross-source synthesis pages
lilbee wiki drafts list   # list pending drafts
lilbee wiki drafts accept <slug>   # promote a draft to concepts/ or entities/
lilbee wiki drafts reject <slug>   # discard a draft
lilbee wiki prune         # move stale pages to archive/
```

Every section is citation-verified against the source chunks and scored for
embedding faithfulness; low-confidence output routes to `drafts/`. Plain-text
concept slugs inside page bodies are rewritten to Obsidian `[[wiki links]]` so
the graph view shows how ideas connect. The directory is Obsidian-compatible
out of the box.

The wiki is built incrementally during `lilbee sync` (with a cap of
`LILBEE_WIKI_INGEST_UPDATE_CAP` changed sources per sync, default 20) so
day-to-day re-ingest never churns existing concept slugs. Run
`lilbee wiki build` explicitly to rebuild from scratch.

MCP tools mirror the CLI: `wiki_list`, `wiki_read`, `wiki_synthesize`,
`wiki_lint`, `wiki_citations`, `wiki_drafts_list`, `wiki_drafts_diff`,
`wiki_prune`. See [Agent integration](#agent-integration).

## Agent integration

lilbee works as a retrieval backend for AI coding agents via MCP or JSON CLI.
See [agent-integration.md](agent-integration.md) for setup.

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

## HTTP Server

`lilbee serve` starts a REST API that any tool or GUI can hit. By default it
picks a random port and writes it to `<data_dir>/server.port` so callers on the
same machine can discover it:

```bash
lilbee serve                      # random port
lilbee serve --port 8080          # fixed port
lilbee serve --host 0.0.0.0       # bind all interfaces (default: 127.0.0.1)
```

The surface covers search (with SSE streaming variants for `ask` and `chat`),
document lifecycle, crawling, model management, configuration (including a
defaults endpoint that powers per-setting reset), and status/health. The
Obsidian plugin uses the `/api/source` endpoint for vault-aware source
retrieval. Interactive API docs live at `/schema/redoc` when the server is
running, and the full OpenAPI schema is published at the
[API reference](https://tobocop2.github.io/lilbee/api/).

**Configuration via env vars:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_SERVER_HOST` | `127.0.0.1` | Bind address |
| `LILBEE_SERVER_PORT` | random | Port (overridden by `--port`) |
| `LILBEE_CORS_ORIGINS` | *(none)* | Extra allowed CORS origins (comma-separated) |
| `LILBEE_CORS_ORIGIN_REGEX` | *(see [Environment variables](#environment-variables))* | Regex for allowed origins |

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

Every setting has a default that works out of the box. The tables below are grouped from most-commonly-touched to rarely-touched, so you can skim the top and skip the bottom unless you have a specific reason.

### Common settings

The ones most users set at least once.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_DATA` | *(platform default)* | Data directory path. Overridden by `--data-dir` or a `.lilbee/` vault walked up from cwd |
| `LILBEE_CHAT_MODEL` | `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` | Chat model. Native GGUF by default; with `pip install --pre 'lilbee[litellm]'` (or `uv tool install --prerelease=allow 'lilbee[litellm]'`), any remote name the SDK backend understands |
| `LILBEE_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf` | Embedding model. Changing this requires `lilbee rebuild` |
| `LILBEE_VISION_MODEL` | *(none)* | Vision OCR model. When set, takes precedence over Tesseract on scanned PDFs and images |
| `LILBEE_VISION_TIMEOUT` | `120` | Per-page vision OCR timeout in seconds (`0` = no limit) |
| `LILBEE_LOG_LEVEL` | `WARNING` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | Custom system prompt for RAG answers |
| `LILBEE_SHOW_REASONING` | `false` | Show the model's `<think>` reasoning tokens in chat output. Useful with Qwen3, DeepSeek-R1, and other reasoning models |

### Retrieval tuning

Reach for these when search quality matters. Defaults are solid; tune only if something feels off.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_TOP_K` | `10` | Number of retrieval results returned |
| `LILBEE_MAX_DISTANCE` | `0.9` | Cosine distance cutoff. Lower = stricter filtering, fewer but more relevant results. `1.0` disables filtering |
| `LILBEE_MMR_LAMBDA` | `0.5` | Relevance vs. diversity balance (1.0 = pure relevance, 0.0 = pure diversity). Raise for factual lookups, lower for exploratory queries |
| `LILBEE_DIVERSITY_MAX_PER_SOURCE` | `3` | Max chunks from a single source document in the top-K. Prevents one big file from dominating results |
| `LILBEE_QUERY_EXPANSION_COUNT` | `3` | LLM-generated query variants per search. `0` disables expansion entirely for faster queries |
| `LILBEE_RERANKER_MODEL` | *(none)* | GGUF cross-encoder reranker for a precision pass over top results. See [Cross-encoder reranking](#cross-encoder-reranking) |
| `LILBEE_RERANK_CANDIDATES` | `20` | Candidates to rerank when a reranker is configured |
| `LILBEE_HYDE` | `false` | Enable Hypothetical Document Embeddings: an LLM drafts a hypothetical answer, that's embedded, and results are merged with the original query's. Adds ~500 ms per query; helps on vague questions |
| `LILBEE_HYDE_WEIGHT` | `0.7` | How much to trust HyDE results relative to the direct query (0.0-1.0) |
| `LILBEE_ADAPTIVE_THRESHOLD` | `false` | When too few results pass `LILBEE_MAX_DISTANCE`, widen the threshold step by step. Useful on small or noisy corpora |
| `LILBEE_ADAPTIVE_THRESHOLD_STEP` | `0.2` | How much to widen per step when adaptive threshold triggers |
| `LILBEE_TEMPORAL_FILTERING` | `true` | When the query contains temporal cues ("recent", "last week"), filter results by document date and sort by recency |
| `LILBEE_MAX_CONTEXT_SOURCES` | `5` | Max chunks included in the LLM's RAG context. Raise for more coverage, lower for shorter prompts |

### Ingestion and chunking

How documents become chunks. Changes here require `lilbee rebuild` to take effect on already-indexed material.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_CHUNK_SIZE` | `512` | Target tokens per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between adjacent chunks |
| `LILBEE_MAX_EMBED_CHARS` | `2000` | Max characters per chunk passed to the embedder |
| `LILBEE_SEMANTIC_CHUNKING` | `false` | Experimental topic-aware chunking. See [Semantic chunking](#semantic-chunking) |
| `LILBEE_TOPIC_THRESHOLD` | `0.75` | Cosine boundary threshold for semantic chunking (lower = more splits) |
| `LILBEE_EMBEDDING_DIM` | `768` | Embedding dimensionality. Must match the embedding model |

### Generation

LLM output shape. Unset values fall through to the model's own defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_TEMPERATURE` | *(model default)* | Sampling temperature |
| `LILBEE_TOP_P` | *(model default)* | Nucleus sampling threshold |
| `LILBEE_TOP_K_SAMPLING` | *(model default)* | Top-k sampling |
| `LILBEE_REPEAT_PENALTY` | *(model default)* | Repetition penalty |
| `LILBEE_NUM_CTX` | *(auto)* | Context window size. Empty = sized automatically to the host's available memory, capped at `LILBEE_NUM_CTX_MAX`. Set explicitly to lock a specific value |
| `LILBEE_NUM_CTX_MAX` | `16384` | Upper bound for the auto-sized context picker. Higher allows more retrieval context on hosts with spare memory |
| `LILBEE_FLASH_ATTENTION` | *(auto)* | Flash attention. Empty/`auto` enables it with a TypeError fallback for older llama-cpp-python builds; `1`/`true`/`on` forces on; `0`/`false`/`off` disables. Resolves the `padding V cache to 1024` warning on models with uneven per-layer V dims |
| `LILBEE_KV_CACHE_TYPE` | `f16` | KV cache element type: `f16`, `f32`, `q8_0`, `q4_0`. Quantized variants halve or quarter cache memory but require flash attention to be enabled |
| `LILBEE_N_GPU_LAYERS` | *(auto)* | Layers to offload to GPU. Empty/`auto` = all (recommended), `cpu` = none, integer = partial offload for tight VRAM |
| `LILBEE_SEED` | *(model default)* | Random seed for reproducibility |

### Server

Only relevant when running `lilbee serve`.

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
| `LILBEE_WIKI_CONCEPT_MAX_CHUNKS_PER_PAGE` | `25` | Top-K chunks grounding each wiki page section |

### Advanced

Rarely touched. Defaults derived from published IR research; there's usually a reason the defaults are the defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `LILBEE_EXPANSION_SKIP_THRESHOLD` | `0.8` | BM25 confidence threshold above which query expansion is skipped (90th-percentile sigmoid-normalized score) |
| `LILBEE_EXPANSION_SKIP_GAP` | `0.15` | Minimum score gap between top-1 and top-2 for expansion to skip (ensures the match is unambiguous) |
| `LILBEE_EXPANSION_GUARDRAILS` | `true` | Filter expansion variants whose embedding drifts too far from the original query |
| `LILBEE_EXPANSION_SIMILARITY_THRESHOLD` | `0.5` | Minimum query-variant cosine similarity to survive the guardrail |
| `LILBEE_CANDIDATE_MULTIPLIER` | `3` | Extra candidates to retrieve before MMR reranking |

CLI flags: `--model` / `-m`, `--data-dir` / `-d`, `--global` / `-g`, `--vision`, `--vision-timeout`, `--log-level`, `--json` / `-j`, `--version` / `-V`.

## Optional extras

lilbee works out of the box with llama-cpp for local inference. These optional extras add capabilities that require heavier dependencies:

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

For NVIDIA users wanting CUDA-native acceleration (default install already covers GPU via Vulkan), append `--extra-index-url https://tobocop2.github.io/lilbee/cu125/` (or `cu124/` for older drivers).

While 0.6.66 is in beta, the `--pre` flag (or uv's `--prerelease=allow`) is required on every install.

Cross-encoder reranking is built in (no extra required); see
[Cross-encoder reranking](#cross-encoder-reranking) below.

---

### Concept graph

Builds a topic map of your documents at index time. Related concepts are linked in a co-occurrence graph, which is used to boost search results and expand queries with related terms, all without extra LLM calls.

**What it does:** Extracts noun phrases from every chunk using spaCy, computes PMI co-occurrence weights between concepts, and clusters them with the Leiden algorithm. At search time, queries are expanded with graph neighbors and results overlapping query concepts get a relevance boost.

**When to use it:** Large corpora (100+ documents) where the same topics appear across multiple files. The graph helps surface connections that pure vector similarity misses. For example, finding "deployment" documents when searching for "CI/CD" because those concepts co-occur frequently.

**Install:** `pip install --pre 'lilbee[graph]'` or `uv tool install --prerelease=allow 'lilbee[graph]'`

**Configuration:**

```bash
export LILBEE_CONCEPT_GRAPH=true              # enable (default: true when deps installed)
export LILBEE_CONCEPT_BOOST_WEIGHT=0.3        # how much concept overlap matters (0.0-1.0)
export LILBEE_CONCEPT_MAX_PER_CHUNK=10        # max concepts extracted per chunk
```

The graph is built automatically during `lilbee sync`. No extra commands needed; search results are boosted transparently.

Based on: Microsoft Research's LazyGraphRAG technique, Church & Hanks 1990 (PMI), Traag et al. 2019 (Leiden).

---

### Web crawling

Index web pages alongside your local documents. Crawl single pages or follow links recursively.

**What it does:** Fetches web pages using a headless browser (Playwright), extracts markdown content, and indexes it. Supports recursive crawling with configurable depth, concurrent fetching, live progress, cancel, per-domain rate-limit + retries on HTTP 429/503, and SSRF protection against internal network access.

**When to use it:** When your corpus spans both local files and web content such as documentation sites, wikis, or internal tools. Crawled content is hash-tracked so re-crawling only re-indexes changed pages.

**Install:** `pip install --pre 'lilbee[crawler]'` or `uv tool install --prerelease=allow 'lilbee[crawler]'`

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

Also available via MCP (`crawl`), REST API (`POST /api/crawl`), and TUI (`/crawl`).

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

Connect to hosted LLM providers instead of (or alongside) local llama-cpp inference.

**What it does:** Routes chat and embedding calls to any provider reachable via the SDK backend (Ollama, OpenAI, Anthropic, Gemini, and many more). The routing provider automatically detects which models are available locally vs. remotely and routes each call to the right backend.

**When to use it:** When you want to use your favorite frontier model for chat while keeping embeddings local for privacy, or when you're already running Ollama and want to use its models.

**Install:** `pip install --pre 'lilbee[litellm]'` or `uv tool install --prerelease=allow 'lilbee[litellm]'` (the extra retains the adapter library name).

**Configuration:**

```bash
export LILBEE_LLM_PROVIDER=auto          # "auto" routes between local and remote
export LILBEE_REMOTE_BASE_URL=http://localhost:11434  # Ollama default
export LILBEE_LLM_API_KEY=sk-...         # API key for your provider
export LILBEE_CHAT_MODEL=your-model      # any remotely-supported model name
```

Provider options: `auto` (default, routes intelligently), `llama-cpp` (local only), `remote` (hosted only).

---

## Cross-encoder reranking

Built-in. Re-scores retrieval candidates with a cross-encoder for precision on the top results. Unlike the extras above, no extra install is required; reranking is off by default and turns on as soon as you set `LILBEE_RERANKER_MODEL`.

**What it does:** After the hybrid search pipeline (BM25 + vector + RRF) returns candidates, a GGUF cross-encoder scores each `(query, chunk)` pair and results are blended with position-aware weights. Top-ranked candidates keep more of the original ranking; lower-ranked candidates trust the reranker more.

**When to use it:** When you need high-precision answers and are willing to trade roughly 200 to 500 ms per query. Most useful with large candidate sets where top-5 ordering matters.

**Configuration:**

```bash
export LILBEE_RERANKER_MODEL="bge-reranker-v2-m3"   # any GGUF reranker
export LILBEE_RERANK_CANDIDATES=20                  # how many candidates to rerank
```

Without a reranker set, hybrid search + MMR already provides good results for most use cases.

Based on: Nogueira & Cho 2019 (Passage Re-ranking with BERT), Burges et al. 2005 (Learning to Rank).

---

## Semantic chunking

Experimental. Off by default. lilbee ships with two chunking strategies; which one serves you depends on what you're indexing.

**Fixed-size (default).** Breaks documents into roughly equal token windows with overlap. Fast, deterministic, works well on code, reference manuals, user guides, API specs, and anything with clear structural boundaries. The assumption is that each chunk only needs to be coherent enough for retrieval, and the model will handle the rest from a small window of context.

**Semantic (experimental).** Uses embedding similarity to detect topic boundaries and splits there instead of at fixed sizes. Each chunk tends to represent one coherent thought rather than an arbitrary slice through one. The benefit shows up on prose-heavy material: novels, essays, long-form research papers, interview transcripts, qualitative research notes, anything where an argument develops across paragraphs. When you ask a question, the retrieved chunk is more likely to contain the full passage that matches rather than the first half of it plus unrelated setup.

**Trade-off:** Enabling semantic chunking triggers a one-time download of kreuzberg's ONNX embedding model (separate from the chunk-to-vector embedder) and runs roughly 9x more downstream embedding calls during indexing. Indexing takes longer; retrieval latency is unchanged.

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

After enabling, run `lilbee rebuild` so existing documents are re-chunked under the new strategy. New documents added from that point use semantic chunking automatically.

### Tuning

```bash
export LILBEE_TOPIC_THRESHOLD=0.75   # cosine threshold for topic boundaries (0.0-1.0)
```

Lower values produce more, smaller chunks (more splits). Higher values produce fewer, larger chunks (the chunker holds related content together until similarity drops sharply).
