# Agent Integration (JSON API)

lilbee exposes a local document knowledge base that any AI agent can query. All commands accept `--json` (or `-j`) before the subcommand for structured output.

## Two modes

- **`search`** — Raw chunk retrieval. No LLM needed. Use when your agent has its own LLM and just needs relevant document chunks.
- **`ask`** — Full local RAG via Ollama. Use for fully-local workflows when Ollama is running.

## Commands

```bash
# Search for relevant chunks (no Ollama needed)
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

## JSON output format

Every command returns a single JSON object on stdout. Errors return non-zero exit + `{"error": "message"}`. Results include `distance` scores (lower = more relevant). Vectors are stripped from output.

## Recommendations

- Prefer `search` if your agent has its own LLM — it's faster and doesn't need Ollama
- Use `ask` for fully-local workflows where no data should leave the machine
- Run `status` first to check if documents are indexed
- Run `sync` after adding documents to update the index
