# Agent Integration

lilbee exposes a local document knowledge base that AI agents can query. Two integration methods are available: MCP (recommended) and JSON CLI.

## MCP Server (recommended)

lilbee ships an MCP server that agents can call directly as tools — no shell needed.

### Setup

Add to your MCP client's MCP server configuration:

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

### Tools

| Tool | Description | Requires LLM backend |
|------|-------------|---------------------|
| `lilbee_search(query, top_k)` | Search for relevant document chunks by vector similarity | No (uses pre-computed embeddings) |
| `lilbee_status()` | Show indexed documents, config, and chunk counts | No |
| `lilbee_sync()` | Sync documents directory with the vector store | Yes (for embedding) |
| `lilbee_add(paths, force, vision_model)` | Add files, dirs, or URLs and sync them into the vector store | Yes (for embedding) |
| `lilbee_crawl(url, depth, max_pages)` | Start a non-blocking crawl of a web page (or site); returns task_id for polling | No (crawl only; sync separately) |
| `lilbee_crawl_status(task_id)` | Check status of a running crawl task (pages crawled, errors, completion) | No |
| `lilbee_init(path)` | Initialize a local `.lilbee/` knowledge base in a directory | No |
| `lilbee_remove(names, delete_files)` | Remove documents from the index (optionally delete source files) | No |
| `lilbee_list_documents()` | List all indexed documents with chunk counts | No |
| `lilbee_reset()` | Delete all documents and data (factory reset) | No |
| `lilbee_model_list(source, task)` | List installed models, optionally filtered by source or task | No |
| `lilbee_model_show(model)` | Show catalog and installed metadata for a model ref | No |
| `lilbee_model_pull(model, source)` | Download a model, streaming progress via MCP progress notifications | Yes (download) |
| `lilbee_model_rm(model, source)` | Remove an installed model | No |

### Example responses

**`lilbee_search("oil change interval", top_k=3)`**
```json
[
  {"source": "manual.pdf", "chunk": "Change oil every 5,000 miles...", "distance": 0.23, ...}
]
```

**`lilbee_status()`**
```json
{
  "config": {"documents_dir": "...", "chat_model": "qwen3:8b", ...},
  "sources": [{"filename": "manual.pdf", "chunk_count": 42}],
  "total_chunks": 42
}
```

## JSON CLI

All commands accept `--json` (or `-j`) before the subcommand for structured output. Use this when MCP isn't available or the agent needs to shell out.

### Two modes

- **`search`** — Raw chunk retrieval. No LLM call at query time. Use when your agent has its own LLM and just needs relevant document chunks.
- **`ask`** — Full local RAG via llama-cpp (or litellm if installed). Use for fully-local workflows.

### Commands

```bash
# Search for relevant chunks (no LLM call at query time)
lilbee --json search "query" --top-k 5
# Returns: {"command": "search", "query": "...", "results": [...]}

# Ask a question with local RAG
lilbee --json ask "question"
# Returns: {"command": "ask", "question": "...", "answer": "...", "sources": [...]}

# Check what's indexed
lilbee --json status
# Returns: {"command": "status", "config": {...}, "sources": [...], "total_chunks": N}

# Trigger document sync
lilbee --json sync
# Returns: {"command": "sync", "added": [...], "updated": [...], "removed": [...], ...}
```

### JSON output format

Every command returns a single JSON object on stdout. Errors return non-zero exit + `{"error": "message"}`. Results include `distance` scores (lower = more relevant). Vectors are stripped from output.

## REST API

The built-in HTTP server (`lilbee serve`) exposes a REST API. Streaming endpoints use Server-Sent Events (SSE).

### Crawl endpoint

`POST /api/crawl` streams SSE progress events while crawling a URL:

```bash
curl -X POST http://localhost:7433/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "depth": 1, "max_pages": 50}'
```

SSE events emitted: `crawl_start`, `crawl_page`, `crawl_done`, then `done` (or `error` on failure).

## Recommendations

- Prefer `search` over `ask` if your agent has its own LLM — it's faster and skips the LLM call at query time
- Use MCP when available — it's more direct than shelling out
- Run `status` / `lilbee_status()` first to check if documents are indexed
- Run `sync` / `lilbee_sync()` after adding documents to update the index
- An LLM backend is needed for: (1) embedding during sync/indexing, (2) `ask` for LLM answers. Once indexed, `search` works without an LLM. By default, llama-cpp handles both. Install `lilbee[litellm]` to use external backends like Ollama or OpenAI.
