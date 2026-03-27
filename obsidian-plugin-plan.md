# Plan: lilbee Obsidian Plugin

## Context

lilbee is a local knowledge base tool. The user wants to use it from Obsidian — search docs, ask questions with RAG, auto-sync the vault, and add notes to the knowledge base. lilbee already has an MCP server (stdio) and JSON CLI, but Obsidian plugins communicate via HTTP to localhost. We need: (1) an HTTP server in lilbee, (2) an Obsidian plugin that consumes it.

User requirements:
- **Litestar** framework (not FastAPI)
- **Monorepo** — plugin lives in `plugins/obsidian/` for now
- **Human-friendly results** — group by document with excerpts, not raw chunks

---

## Part 1: HTTP Server (`lilbee serve`)

### 1.1 Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
server = ["litestar>=2.0", "uvicorn[standard]>=0.30"]
```

### 1.2 New module: `src/lilbee/results.py`

Human-friendly result grouping — transforms raw chunk lists into document-centric results:

```python
@dataclass
class Excerpt:
    content: str
    page_start: int | None
    page_end: int | None
    line_start: int | None
    line_end: int | None
    relevance: float  # 0.0–1.0 (1 = best match)

@dataclass
class DocumentResult:
    source: str
    content_type: str
    excerpts: list[Excerpt]
    best_relevance: float
```

Logic: group chunks by `source`, normalize `_distance` to 0–1 relevance, sort documents by `best_relevance` descending. This lives outside the server so the CLI `search` command can reuse it too.

### 1.3 New module: `src/lilbee/server.py`

Litestar app factory with these endpoints:

| Method | Path | Wraps | Returns |
|--------|------|-------|---------|
| GET | `/api/health` | — | `{ status, version }` |
| GET | `/api/status` | `gather_status()` | config + sources + chunks |
| GET | `/api/search?q=...&top_k=5` | `query.search_context()` → `results.group()` | `list[DocumentResult]` |
| POST | `/api/ask` | `query.ask_raw()` | `{ answer, sources[] }` |
| POST | `/api/ask/stream` | `query.ask_stream()` | SSE events |
| POST | `/api/chat` | `query.ask_raw(history=...)` | `{ answer, sources[] }` |
| POST | `/api/chat/stream` | `query.ask_stream(history=...)` | SSE events |
| POST | `/api/sync` | `ingest.sync()` | **SSE** with per-file progress events |
| POST | `/api/add` | copy file + sync | **SSE** with per-file progress events |

**Chat/Ask SSE format:**
```
event: token
data: {"token": "The"}

event: sources
data: [{"source": "doc.pdf", ...}]

event: done
data: {}
```

**Sync/Add SSE format (indexing progress):**
```
event: progress
data: {"file": "notes/intro.md", "status": "ingested", "current": 1, "total": 12}

event: progress
data: {"file": "papers/study.pdf", "status": "ingested", "current": 2, "total": 12}

event: progress
data: {"file": "bad.docx", "status": "failed", "current": 3, "total": 12}

event: done
data: {"added": [...], "updated": [...], "removed": [...], "unchanged": 5, "failed": [...]}
```

**Key details:**
- `create_app() -> Litestar` factory (no global state)
- CORS: allow `app://obsidian.md` and `http://localhost:*`
- Bind `127.0.0.1` only (security)
- Accept `?data_dir=` query param to target a specific `.lilbee/`
- Lazy imports for litestar/uvicorn (heavy deps)
- Use `litestar.response.Stream` for SSE

### 1.4 Config additions (`config.py`)

- `LILBEE_SERVER_HOST` → `server_host` (default `127.0.0.1`)
- `LILBEE_SERVER_PORT` → `server_port` (default `7433`)

### 1.5 CLI command (`cli/commands.py`)

```python
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    port: int = typer.Option(7433, "--port", "-p"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
```

### 1.6 Tests

- `tests/test_results.py` — grouping logic, relevance normalization, edge cases
- `tests/test_server.py` — Litestar `TestClient`, mock Ollama, test all endpoints + SSE + CORS
- Follow `test_mcp.py` pattern: `isolated_env` fixture, snapshot/restore `cfg`

---

## Part 2: Obsidian Plugin (`plugins/obsidian/`)

### 2.1 Structure

```
plugins/obsidian/
  manifest.json
  package.json
  tsconfig.json
  esbuild.config.mjs
  styles.css
  src/
    main.ts           # Plugin entry, commands, auto-sync
    api.ts            # Typed HTTP client for lilbee server
    types.ts          # Interfaces (DocumentResult, etc.)
    settings.ts       # Settings tab
    views/
      search-modal.ts # Search UI
      chat-view.ts    # Sidebar chat panel
      results.ts      # Result rendering helpers
```

### 2.2 Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `serverUrl` | `http://127.0.0.1:7433` | lilbee server address |
| `topK` | `5` | Results to return |
| `autoSync` | `true` | Watch vault for changes |
| `syncDebounceMs` | `5000` | Debounce interval |

### 2.3 Commands

| Command | Description |
|---------|-------------|
| `lilbee:search` | Open search modal |
| `lilbee:ask` | One-shot question modal |
| `lilbee:chat` | Open/focus sidebar chat |
| `lilbee:add-note` | Add active note to knowledge base |
| `lilbee:sync` | Trigger vault sync |
| `lilbee:status` | Show status notice |

### 2.4 Search Modal

- Extends Obsidian `Modal`
- Shows results as document cards (not chunks):
  - Document name (clickable → opens in vault)
  - Relevance indicator (visual, not numeric)
  - 1–3 excerpts with location info
  - Content type badge

### 2.5 Chat Sidebar

- Extends `ItemView` (registered as right sidebar leaf)
- Streaming via `fetch` + `ReadableStream` (POST+SSE — `EventSource` only supports GET)
- Sources shown as collapsible section below each response
- Chat history maintained in view state

### 2.6 Auto-Sync & Indexing Progress

- Listen to vault `create`/`modify`/`delete`/`rename` events
- Debounce, then POST `/api/sync`
- **Progress UI**: status bar shows "lilbee: syncing 3/12..." with current file name
- On completion: brief notice "lilbee: synced — 5 added, 2 updated"
- On failure: notice listing failed files

### 2.7 Build

- esbuild → `main.js`
- Add Makefile targets: `plugin-build`, `plugin-dev`

---

## Implementation Order

1. `src/lilbee/results.py` + `tests/test_results.py` — pure logic, no deps
2. Config additions (`server_host`, `server_port`)
3. `src/lilbee/server.py` + `tests/test_server.py` — Litestar app
4. `pyproject.toml` — optional `[server]` deps
5. `cli/commands.py` — `serve` command
6. `plugins/obsidian/` — scaffold (manifest, package.json, tsconfig, esbuild)
7. `plugins/obsidian/src/api.ts` + `types.ts` — HTTP client
8. `plugins/obsidian/src/main.ts` + `settings.ts` — plugin core
9. `plugins/obsidian/src/views/` — search modal, chat sidebar

---

## Verification

1. `make check` passes (lint, format, typecheck, tests at 100% coverage)
2. `uv run lilbee serve` starts, `curl http://127.0.0.1:7433/api/health` returns OK
3. `curl http://127.0.0.1:7433/api/search?q=test` returns grouped DocumentResults
4. SSE streaming: `curl http://127.0.0.1:7433/api/ask/stream` returns token events
5. `cd plugins/obsidian && npm run build` produces `main.js`
6. Manual test: install plugin in Obsidian, search/chat/add-note work

---

## Notes

- **LanceDB concurrency**: concurrent reads are fine, but only one process should write (sync) at a time. Document that `lilbee serve` should be the sync authority when running.
- **Plugin distribution**: initially manual install (copy `main.js` + `manifest.json` to `.obsidian/plugins/lilbee/`). Community plugin submission later.
- **`ask_stream` modification needed**: current `ask_stream` yields citations as a final string. The server needs raw sources separately to send as structured SSE. May need a new `ask_stream_raw` that yields tokens then returns sources, or refactor the existing one.

## Critical Files

- `src/lilbee/query.py` — `search_context`, `ask_raw`, `ask_stream` (lines 62–156)
- `src/lilbee/cli/commands.py` — add `serve` command following existing patterns
- `src/lilbee/cli/helpers.py` — `gather_status()`, `copy_paths()` to reuse
- `src/lilbee/config.py` — add `server_host`, `server_port` fields
- `src/lilbee/mcp.py` — reference for wrapping core APIs
- `pyproject.toml` — add `[server]` optional deps
- `tests/test_mcp.py` — test pattern reference
