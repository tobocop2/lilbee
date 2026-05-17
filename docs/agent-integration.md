# Agent Integration

lilbee serves as a local retrieval backend for AI coding agents. Two entry
points are available: MCP (recommended) and JSON CLI.

## Fastest start: have the agent configure lilbee for you

The shortest path from "fresh install" to "useful lilbee" is to hand the
setup to an MCP-aware agent. It can pick models for your hardware, pull
them, wire them into the embedding / reranker / vision roles, and tune
retrieval for the kind of questions you actually want to ask. See
[Fine-tuning lilbee from your agent](#fine-tuning-lilbee-from-your-agent)
below for the canonical example prompt.

## MCP Server (recommended)

`lilbee mcp` launches an MCP server that agents call directly as tools. No
shell-out needed.

### Setup

Add to your MCP client's configuration:

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "lilbee",
      "args": ["mcp"]
    }
  }
}
```

For opencode, an `opencode.json` in the project root works too. This one denies the
built-in search tools so the agent has to use lilbee, and allows the `task` tool so it can
delegate long ops to a subagent:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "codesearch": "deny",
    "websearch": "deny",
    "webfetch": "deny",
    "read": "allow",
    "write": "allow",
    "edit": "allow",
    "bash": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "task": "allow",
    "lilbee_*": "allow"
  },
  "mcp": {
    "lilbee": { "type": "local", "command": ["lilbee", "mcp"] }
  }
}
```

### Drop-in agent files

For a project where you want the agent to use lilbee reliably, copy three things in:

1. An `AGENTS.md` (or `CLAUDE.md`) that names lilbee as the retrieval backend, lists the
   citation rule, and says long ops go to a worker subagent. The lilbee repo ships a copy at
   [`demos/AGENTS.md`](../demos/AGENTS.md).
2. A `lilbee-worker` subagent that handles `lilbee_add` / `lilbee_sync` / `lilbee_crawl` /
   `lilbee_model_pull`. Copy from
   [`demos/.opencode/agents/lilbee-worker.md`](../demos/.opencode/agents/lilbee-worker.md).
3. The [`lilbee-mcp` skill](agent-skills/lilbee-mcp/SKILL.md) (opencode / Claude
   Skill format), copied into `.opencode/skills/lilbee-mcp/` or
   `.claude/skills/lilbee-mcp/`. A single `SKILL.md` that documents every lilbee
   MCP tool with a quick-vs-long split, so the agent knows which calls block and
   which don't.

### Tools

| Tool | Description | Requires LLM backend |
|------|-------------|---------------------|
| `search(query, top_k, scope)` | Retrieve relevant chunks. `scope` is `"raw"` (source docs), `"wiki"` (wiki pages), or `"both"` (default) | No (uses pre-computed embeddings) |
| `status()` | Show indexed documents, config, and chunk counts | No |
| `sync()` | Sync the documents directory into the vector store | Yes (for embedding) |
| `add(paths, force, enable_ocr, ocr_timeout)` | Add files, directories, or URLs and index them | Yes (for embedding) |
| `crawl(url, depth, max_pages)` | Start a non-blocking crawl. Returns a `task_id` for polling | No (crawl only; sync separately) |
| `crawl_status(task_id)` | Check a running crawl's progress, errors, and completion | No |
| `init(path)` | Create a local `.lilbee/` in the given directory | No |
| `remove(names, delete_files)` | Remove documents from the index (optionally delete sources) | No |
| `list_documents()` | List all indexed documents with chunk counts | No |
| `reset(confirm)` | Delete all documents and data (factory reset; pass `confirm=true`) | No |
| `model_list(source, task)` | List installed models, optionally filtered by source or role | No |
| `model_show(model)` | Show catalog + installed metadata for a model ref | No |
| `model_pull(model, source)` | Download a model, streaming progress via MCP notifications | Yes (download) |
| `model_rm(model, source)` | Remove an installed model | No |
| `catalog_browse(task, search, size, installed, featured, sort, limit, offset)` | Browse the lilbee model catalog (curated + Hugging Face) so the agent can pick what to pull | No |
| `settings_list(group)` | List every writable setting with value, default, type, help text, choices, and `reindex_required` | No |
| `settings_get(key)` | Get one setting's current value and metadata | No |
| `settings_set(updates)` | Atomically update a batch of writable settings; validates, persists, and invalidates the in-process model and provider caches | No |
| `settings_reset(keys)` | Reset writable settings to their built-in defaults | No |
| `wiki_list()` | List all wiki pages grouped by type | No |
| `wiki_read(slug)` | Return the body and metadata of a single wiki page | No |
| `wiki_status()` | Page counts, generator settings, last build timestamp | No |
| `wiki_synthesize()` | Generate cross-source synthesis pages into `synthesis/` | Yes (LLM) |
| `wiki_lint(wiki_source)` | Find orphan pages, stale links, and pending drafts | No |
| `wiki_citations(wiki_source)` | Return per-section citation coverage for a source | No |
| `wiki_drafts_list()` | List pending drafts with drift, faithfulness, and pairing info | No |
| `wiki_drafts_diff(slug)` | Show the diff between a pending draft and the live page | No |
| `wiki_prune()` | Move stale wiki pages to `archive/` | No |

### Example responses

**`search("oil change interval", top_k=3)`**

```json
[
  {"source": "manual.pdf", "chunk": "Change oil every 5,000 miles...", "distance": 0.23, "chunk_type": "raw"}
]
```

**`status()`**

```json
{
  "config": {"documents_dir": "...", "chat_model": "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", "embedding_model": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf", "reranker_model": "", "enable_ocr": false},
  "sources": [{"filename": "manual.pdf", "chunk_count": 42}],
  "total_chunks": 42
}
```

**`wiki_list()`**

```json
{
  "concepts": [{"slug": "braking-systems", "sources": 5}],
  "entities": [{"slug": "henry-ford", "sources": 3}],
  "drafts": [{"slug": "tire-pressure", "reason": "low_faithfulness"}]
}
```

## API keys never come back over MCP

API-key fields (and `hf_token`) carry a `write_only` flag on the
config. `settings_set` still accepts writes, but `settings_list`
skips them and `settings_get` errors. Secrets do not round-trip.

## Fine-tuning lilbee from your agent

Drop this into any MCP-aware agent:

> I'm going to index `~/projects/my-stack/` with lilbee and ask
> questions about how the auth layer is wired and which functions call
> which. Assess my hardware, recommend embedding / reranker / vision
> models, pull them in the background, then adapt the lilbee defaults
> for this corpus and question style.

The agent answers the questions itself, so it only touches model roles
that affect retrieval (`embedding_model`, `reranker_model`,
`vision_model`). The `chat_model` slot is for the human's later TUI use.

Typical flow: `lilbee_status` + `lilbee_settings_list` to see baseline,
`lilbee_catalog_browse(task=...)` to pick models per role, `lilbee_model_pull`
via the `lilbee-worker` subagent, one batched `lilbee_settings_set` to wire
the models and tune retrieval (raise `top_k` / `diversity_max_per_source`
for code-heavy corpora; enable `concept_graph`; lower `chunk_size`).
If the result includes `reindex_required: true`, the agent should hand
`lilbee_sync(force_rebuild=true)` to the worker.

## JSON CLI

Every command accepts `--json` (or `-j`) before the subcommand for structured output. Use this when MCP isn't available or when the agent needs to shell out.

### Two modes

- **`search`.** Raw chunk retrieval. No LLM call at query time. Use when your agent has its own LLM and just needs relevant chunks.
- **`ask`.** Full local RAG via llama-cpp (or the SDK backend when installed). Use for fully-local workflows.

### Commands

```bash
# Retrieve chunks (no LLM call at query time)
lilbee --json search "query" --top-k 5
# {"command": "search", "query": "...", "results": [...]}

# Ask a question with local RAG
lilbee --json ask "question"
# {"command": "ask", "question": "...", "answer": "...", "sources": [...]}

# Check what's indexed
lilbee --json status
# {"command": "status", "config": {...}, "sources": [...], "total_chunks": N}

# Trigger document sync
lilbee --json sync
# {"command": "sync", "added": [...], "updated": [...], "removed": [...]}
```

### JSON output format

Every command returns a single JSON object on stdout. Errors return non-zero exit + `{"error": "message"}`. Results include `distance` scores (lower = more relevant). Vectors are stripped from output.

## REST API

The built-in HTTP server (`lilbee serve`) exposes a full REST API. Streaming endpoints use Server-Sent Events (SSE). See the [REST API reference](https://lilbee.sh/api/) for the complete OpenAPI schema and [the usage guide](usage.md#http-server) for invocation options.

### Crawl endpoint

`POST /api/crawl` streams SSE progress events while crawling a URL:

```bash
curl -X POST http://localhost:7433/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "depth": 1, "max_pages": 50}'
```

SSE events emitted: `crawl_start`, `crawl_page`, `crawl_done`, then `done` (or `error` on failure).

## Recommendations

- Prefer `search` over `ask` if your agent has its own LLM. It's faster and skips the LLM call at query time.
- Use MCP when available. It's more direct than shelling out.
- Run `status` / `status()` first to confirm the right index is active.
- Run `sync` / `sync()` after adding documents to refresh the index.
- An LLM backend is needed for: (1) embedding during sync/indexing, (2) `ask` for answers, (3) wiki generation, (4) `model_pull`. Once indexed, `search` works without an LLM. By default, llama-cpp handles everything locally. Install `lilbee[litellm]` to route through external backends like Ollama, OpenAI, Anthropic, or Gemini.
