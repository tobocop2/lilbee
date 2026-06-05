---
name: lilbee-mcp
description: Search and manage the user's local lilbee knowledge base over MCP. Use whenever the user has indexed code, docs, PDFs, or web pages into lilbee and you need cited answers, or whenever they ask you to ingest content, swap models, or tune retrieval against their library. Every fact returned cites file and line. Indexing, crawling, and model pulls are long ops that must go to a worker subagent so the chat stays responsive.
---

# lilbee-mcp

[lilbee](https://github.com/tobocop2/lilbee) is a local retrieval engine. It indexes the
user's code, documents, PDFs, and crawled web pages into a per-project `.lilbee/` store and
exposes the library over MCP. Every tool here is prefixed `lilbee_`. Data and embeddings stay
on the user's machine; the only thing leaving is what you, the agent, decide to quote.

## In 30 seconds

```
lilbee_status                     → see what's loaded
lilbee_search(query, top_k)       → get cited chunks
[answer with file:line citations] → never invent
```

Three rules cover 90% of usage: **search before answering**, **cite every claim with the
chunk's `source` + line range**, and **delegate indexing / crawling / model pulls to the
`lilbee-worker` subagent** because they block the shared embedder.

## Install

Drop this folder under one of:

```
.opencode/skills/lilbee-mcp/      # opencode (project)
.claude/skills/lilbee-mcp/        # Claude (project)
~/.config/opencode/skills/lilbee-mcp/   # opencode (global)
~/.claude/skills/lilbee-mcp/      # Claude (global)
```

Register lilbee as an MCP server (opencode example):

```json
{
  "mcp": {
    "lilbee": { "type": "local", "command": ["lilbee", "mcp"] }
  }
}
```

A drop-in `AGENTS.md`, the `lilbee-worker` subagent, and an `opencode.json` template live
in `examples/agent-integration/` in the lilbee repo. Copy them in if the user wants the
full setup.

## The shared-embedder rule (read this first)

The MCP server hosts one embedder worker. Indexing (`lilbee_add`, `lilbee_sync`,
`lilbee_crawl`, `lilbee_model_pull`, plus the experimental wiki builds) pins it;
`lilbee_search` also needs it to embed the query. Run them
concurrently and `lilbee_search` will hang until your host times out.

**Procedure:**

1. If indexing is needed, delegate to `lilbee-worker` and **wait** for the worker's `task`
   call to return. Don't fire any `lilbee_*` tool from your own thread while it runs.
2. After the worker returns, call `lilbee_status` once to confirm the expected counts.
3. Then search.

If `lilbee_search` returns an MCP timeout, treat it as "indexing isn't fully done yet":
wait ~10s, re-check `lilbee_status`, retry. Don't switch tools.

## Tools by cost

### Inline (cheap, sub-second)

| Tool | Use |
|---|---|
| `lilbee_search(query, top_k, scope)` | Retrieve relevant chunks. `top_k` omitted falls back to `cfg.top_k` so `settings_set` governs candidate count. `scope` = `"raw"` / `"wiki"` / `"both"` (default). No LLM call. |
| `lilbee_status()` | Indexed sources, total chunks, active model refs. First call of any session. |
| `lilbee_list_documents()` | All indexed sources with chunk counts. |
| `lilbee_init(path)` | Create a `.lilbee/` in the given dir and switch the session to it. |
| `lilbee_remove(names, delete_files)` | Remove documents from the index, optionally deleting the source files. |
| `lilbee_crawl_status(task_id)` | Poll a non-blocking crawl: `pending` / `running` / `done` / `failed`. |
| `lilbee_model_list(source, task)` | Locally-installed models, optionally filtered. |
| `lilbee_model_show(model)` | Catalog + installed metadata for one model ref. |
| `lilbee_model_rm(model, source)` | Delete an installed model from disk. |
| `lilbee_catalog_browse(task, search, size, installed, featured, sort, limit, offset)` | Browse the curated catalog + Hugging Face. Use before `lilbee_model_pull` to pick what to install. Each returned entry includes the model's `architecture` and a `compat` field (`supported` / `unsupported` / `unknown`); check `compat` before pulling. |
| `lilbee_settings_list(group)` | Every writable setting with value, default, type, help text, choices, `reindex_required`. |
| `lilbee_settings_get(key)` | One setting's current value + metadata. |
| `lilbee_settings_set(updates)` | Atomically update writable settings. Persists to `config.toml`, invalidates in-process model and provider caches. |
| `lilbee_settings_reset(keys)` | Reset writable settings to their built-in defaults. |
| `lilbee_memory_remember(text, kind, shared, agent_id)` | Save a durable note (`kind` = `"fact"` / `"preference"`). Embeds text, so it obeys the shared-embedder rule. Needs `memory_enabled`. |
| `lilbee_memory_recall(query, limit, agent_id)` | Recall your saved memories by relevance. Embeds the query (shared-embedder rule). |
| `lilbee_memory_list(agent_id)` | List your stored memories. No embedding. |
| `lilbee_memory_forget(memory_id)` | Delete one of your memories by id. No embedding. |

### Long (must go through `lilbee-worker`)

| Tool | Use |
|---|---|
| `lilbee_add(paths, force, enable_ocr, ocr_timeout)` | Copy files / dirs / URLs into the library and index them. Seconds to minutes. |
| `lilbee_sync(force_rebuild, retry_skipped)` | Re-index the documents directory after edits. Minutes on large libraries. |
| `lilbee_crawl(url, depth, max_pages)` | Start a non-blocking crawl. Returns `task_id`; poll `lilbee_crawl_status`. |
| `lilbee_model_pull(model, source, allow_unsupported)` | Download a model. Streams progress as MCP notifications. Large models take minutes. Set `allow_unsupported=true` to override the architecture-compat check; without it, the call returns a structured error with `code: "unsupported_arch"` and the supported-architecture list. |
| `lilbee_reset(confirm)` | Wipe the entire index and data dir. Pass `confirm=true`. Destructive. |

(Experimental wiki tools are documented at the end of this skill.)

## Common workflows

### 1. User asks a question about their library

```
lilbee_status                           # confirm a library exists
lilbee_search(query)                    # top_k defaults to cfg.top_k (12)
answer with file:line citations
```

If `total_chunks == 0`, tell the user the index is empty and offer to add content.

### 2. User wants to add new content

```
lilbee-worker:   lilbee_add(["/path/to/dir", "https://docs.example.com/page"])
[wait for worker to return]
lilbee_status                           # confirm new sources appeared
```

`lilbee_add` accepts absolute paths, directories (recursive), and URLs (which get crawled
to markdown). It runs `lilbee_sync` after copying, so the new content is indexed in one
call. For continuous web monitoring, use `lilbee_crawl` instead.

### 3. User wants to crawl a docs site

```
task_id = lilbee_crawl("https://docs.example.com", depth=2, max_pages=200)
[poll lilbee_crawl_status(task_id) until status == "done"]
lilbee_status                           # new "_web/..." sources should appear
```

The crawl writes pages into the documents directory as it goes; a final auto-sync indexes
them. Pages already crawled this session are skipped.

### 4. User wants to set lilbee up for their hardware and files

You manipulate the retrieval surface only: `embedding_model`, `reranker_model`,
`vision_model`, and the retrieval / ingest knobs. The `chat_model` slot is for the user's
later TUI / CLI sessions; leave it unless the user explicitly asks.

```
lilbee_status                                       # see what's wired up
lilbee_settings_list(group="Retrieval")             # baseline knobs
lilbee_catalog_browse(task="embedding")             # discover candidates
lilbee_catalog_browse(task="rerank")
lilbee_catalog_browse(task="vision")
lilbee-worker:  lilbee_model_pull(<picked model>)   # one worker call per pull
lilbee_settings_set({
    "embedding_model": "...",
    "reranker_model":  "...",
    "vision_model":    "...",
    # plus retrieval tuning, all batched:
    "top_k": 12,
    "diversity_max_per_source": 6,
    "concept_graph": true,
})
```

If the response includes `reindex_required: true` (triggered by `chunk_size`,
`chunk_overlap`, or swapping `embedding_model` to a different ref), hand
`lilbee_sync(force_rebuild=true)` to the worker before searching again.

Tell the user which knobs you moved and why; `lilbee_settings_reset([...])` rolls any of
them back.

### 5. Your first answer feels thin -- self-tune and retry

When the user asks a broad question against a dense pile of reference
docs (godot class XMLs, an API reference, kreuzberg-style docstrings)
and your first `lilbee_search` returns only one or two relevant hits
where you'd expect a family, the retrieval defaults are too narrow for
the shape of what's indexed. Self-tune in-place rather than handing the
user a thin answer:

```
lilbee_search("user's natural query")          # baseline
# If results are visibly narrow for the indexed shape:
lilbee_settings_set({
    "top_k": 15,                                # wider candidate pool
    "diversity_max_per_source": 8,              # more chunks per file ok
    "max_distance": 0.85,                       # accept fuzzier matches
})
lilbee_search("user's natural query")          # same query, richer pool
# Answer from the richer results, cite every class you found.
lilbee_settings_reset(["top_k", "diversity_max_per_source", "max_distance"])
```

Tell the user one sentence on what you widened and that you've reset
afterward, so the next question gets the unmodified defaults.

### 6. User wants to delete or replace content

```
lilbee_list_documents                                # find the source name
lilbee_remove(["old-manual.pdf"], delete_files=False) # keep the file, drop chunks
# or
lilbee_remove(["old-manual.pdf"], delete_files=True)  # delete file + chunks
```

For a clean slate: `lilbee_reset(confirm=true)` via the worker.

## Cheat sheet: which knobs to move

| Question style | Tune |
|---|---|
| Code Q&A, call graphs | Set `reranker_model`, raise `rerank_candidates` to 24-48, enable `concept_graph`, lower `chunk_size` to 256-384. |
| Long-document walkthroughs | Raise `top_k` to 12-16 and `max_context_sources`; drop `diversity_max_per_source` to 1 to let one source dominate. |
| Fact lookup across many sources | Raise `top_k` and `candidate_multiplier`; keep `reranker_model` set; keep `diversity_max_per_source` at default. |
| Vocabulary mismatch (English question, code answer) | Enable `hyde`; raise `max_distance` to ~0.8 so semantically-related chunks aren't clipped. |

## Runbook: when things go wrong

- **`total_chunks == 0`.** The library is empty. Don't search. Tell the user and offer
  `lilbee_add` or `lilbee_crawl`.
- **`lilbee_search` returns 0 results.** Try a more specific noun phrase. If still zero,
  the content isn't indexed. Verify with `lilbee_list_documents`.
- **`lilbee_search` times out.** Indexing is in flight. Wait 10s, re-check
  `lilbee_status`, retry. Do not switch tools.
- **`lilbee_settings_set` returns `error`.** The boundary rejected the value (unknown
  key, wrong type, model not installed for a role swap). Show the error to the user
  verbatim; don't paper over it.
- **`lilbee_settings_set` succeeds with `reindex_required: true`.** The persisted vector
  store is no longer valid for the new `chunk_size` / `chunk_overlap`. Run
  `lilbee_sync(force_rebuild=true)` through the worker.
- **`lilbee_model_pull` failed mid-stream.** The progress notifications stop. Retry the
  same call; partially-downloaded files are skipped.
- **Want to set a model role but the catalog ref isn't installed.** Pull first, then
  `settings_set`. The boundary checks the catalog-task assignment but not whether the
  file is present on disk.
- **Confused about what changed in this session.** `lilbee_settings_list` always
  reflects current state.

## Secrets stay out of MCP reads

API keys (`*_api_key` and `hf_token`) carry a `write_only` flag on the lilbee config.
`lilbee_settings_list` skips them; `lilbee_settings_get` errors on them;
`lilbee_settings_set` still writes (so you can configure a key on the user's behalf).
Never assume you can read a key back.

## Citation rules

- Every fact you state must trace to a chunk `lilbee_search` returned. Cite as the
  source file and the line range the chunk reported, exactly as returned (e.g.
  `src/lilbee/data/ingest/pipeline.py:216-247`).
- If a chunk doesn't actually support a claim, drop the claim. Re-search before
  inventing.
- "Not in the indexed files" is a valid answer. Say so plainly and suggest indexing the right
  path if the user expected it to be there.
- When the user reads code in their editor that you've cited, your line numbers must
  match. If you see the chunk metadata report a line range, do not guess a different one.

## What this skill is not

- **Not a code editor.** Use the host's `read` / `edit` / `write` tools after pulling
  context through `lilbee_search`.
- **Not a web search.** `lilbee_crawl` fetches a specific URL into the library on the
  user's explicit request; it isn't a general fallback when search misses.
- **Not a chat model.** lilbee can run its own chat / wiki / OCR models locally, but as
  a tool consumer you stay on the host's model. Don't manipulate `chat_model` unless the
  user is setting lilbee up for their own later TUI use.
- **Not a substitute for codebase discovery tools.** If the host has codesearch / glob /
  grep and the user's question is about a path that isn't indexed yet, offer to index it
  rather than guessing through filesystem tools.

## Other skills

- **Wiki tools** (experimental per-concept and per-entity pages with citations) live in the separate [`lilbee-mcp-wiki` skill](../lilbee-mcp-wiki/SKILL.md). Load that only when the user asks about wiki / synthesis pages, or when `lilbee_status` already shows a built wiki.
- **Non-MCP agents** that need to shell out to lilbee's CLI instead of MCP can use the `--json` flag on any command. See the [JSON CLI fallback](../../usage.md#json-cli-fallback) in the usage guide.
