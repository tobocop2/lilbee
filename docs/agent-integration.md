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

| Tool | Description | Requires Ollama |
|------|-------------|-----------------|
| `lilbee_search(query, top_k)` | Search for relevant document chunks by vector similarity | No (uses pre-computed embeddings) |
| `lilbee_ask(question)` | Ask a question answered by local RAG | Yes |
| `lilbee_status()` | Show indexed documents, config, and chunk counts | No |
| `lilbee_sync()` | Sync documents directory with the vector store | Yes (for embedding) |

### Example responses

**`lilbee_search("oil change interval", top_k=3)`**
```json
[
  {"source": "manual.pdf", "chunk": "Change oil every 5,000 miles...", "distance": 0.23, ...}
]
```

**`lilbee_ask("What is the oil capacity?")`**
```json
{
  "answer": "The oil capacity is 5 quarts.",
  "sources": [{"source": "manual.pdf", "chunk": "...", "distance": 0.25}]
}
```

**`lilbee_status()`**
```json
{
  "config": {"documents_dir": "...", "chat_model": "mistral", ...},
  "sources": [{"filename": "manual.pdf", "chunk_count": 42}],
  "total_chunks": 42
}
```

## JSON CLI

All commands accept `--json` (or `-j`) before the subcommand for structured output. Use this when MCP isn't available or the agent needs to shell out.

### Two modes

- **`search`** — Raw chunk retrieval. No LLM call at query time. Use when your agent has its own LLM and just needs relevant document chunks.
- **`ask`** — Full local RAG via Ollama. Use for fully-local workflows when Ollama is running.

### Commands

```bash
# Search for relevant chunks (no Ollama needed at query time)
lilbee --json search "query" --top-k 5
# Returns: {"command": "search", "query": "...", "results": [...]}

# Ask a question with local RAG (requires Ollama)
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

## Recommendations

- Prefer `search` over `ask` if your agent has its own LLM — it's faster and doesn't need Ollama at query time
- Use MCP when available — it's more direct than shelling out
- Run `status` / `lilbee_status()` first to check if documents are indexed
- Run `sync` / `lilbee_sync()` after adding documents to update the index
- Ollama is needed for two things: (1) embedding during sync/indexing, (2) `ask` for LLM answers. Once indexed, `search` works without Ollama
