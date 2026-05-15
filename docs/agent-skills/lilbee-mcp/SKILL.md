---
name: lilbee-mcp
description: Drive a local lilbee retrieval-and-corpus server over MCP. Search an indexed corpus of code and documents with file:line citations, manage what's indexed, crawl docs sites, and manage local models. Use this whenever you'd otherwise be guessing about code or documents the user has loaded into lilbee.
---

# lilbee-mcp

[lilbee](https://github.com/tobocop2/lilbee) is a local retrieval engine: it indexes the
user's code and documents and exposes an MCP server you (the agent) talk to. Every tool name
is prefixed `lilbee_`. lilbee runs entirely on the user's machine; no data leaves it unless
you also call out to a cloud model yourself.

## Install (in the project the agent runs in)

Copy this folder into one of:

```
.opencode/skills/lilbee-mcp/      # opencode (project)
.claude/skills/lilbee-mcp/        # Claude (project)
~/.config/opencode/skills/lilbee-mcp/   # opencode (global)
~/.claude/skills/lilbee-mcp/      # Claude (global)
```

Add `lilbee` to the MCP servers in the host's config (opencode example):

```json
{
  "mcp": {
    "lilbee": { "type": "local", "command": ["lilbee", "mcp"] }
  }
}
```

A matching `AGENTS.md` snippet and a `lilbee-worker` subagent live in
`docs/agent-integration.md` in the lilbee repo.

## How to use it

1. **First contact:** call `lilbee_status`. If `total_chunks` is 0, the corpus is empty and
   you need to index before you can answer from it. If `.lilbee/` doesn't exist yet, run
   `lilbee_init` first (or let the worker handle both).
2. **Search before answering.** Pass the most distinct noun phrase as the query, not the
   whole sentence. `lilbee_search` returns chunks with the source file and line; cite them
   exactly as returned. If the answer isn't in any chunk, say so. Don't invent.
3. **Long ops go to a subagent.** They block. Run them through a `lilbee-worker` agent (or
   any background-capable subagent the host provides) so you stay responsive.

## Hard rule: search and indexing cannot run at the same time

The lilbee MCP server hosts one shared embedder worker. Indexing (`lilbee_add`,
`lilbee_sync`, `lilbee_crawl`, `lilbee_model_pull`) pins that worker for as long as it runs;
`lilbee_search` needs the same worker to embed the query. Until indexing finishes, search
calls will hang and your MCP client will time them out.

The rule for you (the agent):

1. Decide whether to index. If yes, delegate to a `lilbee-worker` subagent and **wait** for
   the worker's `task` call to return. Don't fire `lilbee_search` (or any other inline
   `lilbee_*` tool) from your own thread while the worker is still running.
2. After the worker returns, run `lilbee_status` once to confirm the expected counts.
3. Then search.

If a `lilbee_search` call ever returns an MCP timeout, treat it as "indexing isn't fully
done yet" — wait 10 seconds, re-check `lilbee_status`, and retry. Do **not** change
strategy and reach for other tools; the search will work the moment the embedder is free.

## Tools, by cost

### Quick (run inline)

| Tool | What it does |
|---|---|
| `lilbee_search(query, top_k, scope)` | Retrieve relevant chunks. `scope` is `"raw"` (source docs), `"wiki"` (wiki pages), or `"both"` (default). No LLM call. |
| `lilbee_status()` | Indexed documents, configuration, total chunks. Use to check what's loaded before searching. |
| `lilbee_list_documents()` | All indexed documents with chunk counts. |
| `lilbee_init(path)` | Create a `.lilbee/` in the given dir. Switches the session to it. |
| `lilbee_remove(names, delete_files)` | Remove documents from the index (optionally delete sources). |
| `lilbee_crawl_status(task_id)` | Poll a running crawl: `status` is `"pending"`, `"running"`, `"done"`, or `"failed"`. |
| `lilbee_model_list(source, task)` | Installed models, optionally filtered. |
| `lilbee_model_show(model)` | Catalog + installed metadata for one model. |

### Long (delegate to a subagent)

| Tool | What it does |
|---|---|
| `lilbee_add(paths, force, enable_ocr, ocr_timeout)` | Copy files / dirs / URLs into the corpus and index them. Blocks for seconds to minutes depending on size. |
| `lilbee_sync(force_rebuild, retry_skipped)` | Re-index the documents directory (after edits to existing files). Blocks. |
| `lilbee_crawl(url, depth, max_pages)` | Start a non-blocking crawl. Returns a `task_id`; poll `lilbee_crawl_status` until done. |
| `lilbee_model_pull(model, source)` | Download a model from HF (or another source). Streams progress as MCP notifications; still slow on large models. |
| `lilbee_reset(confirm)` | Wipe the index and data. Pass `confirm=true`. Destructive. |

### Wiki (experimental, optional)

`lilbee_wiki_list`, `lilbee_wiki_read`, `lilbee_wiki_status`, `lilbee_wiki_synthesize`,
`lilbee_wiki_lint`, `lilbee_wiki_citations`, `lilbee_wiki_drafts_list`,
`lilbee_wiki_drafts_diff`, `lilbee_wiki_prune`, `lilbee_wiki_build`, `lilbee_wiki_update`.
Auto-generated concept and entity pages over the indexed sources. See lilbee's usage guide
for the build / draft / prune cycle.

## Citation rules

- Every fact you state must trace to a chunk `lilbee_search` returned. Cite as the source
  file and the line range the chunk reported (e.g. `src/lilbee/data/ingest/pipeline.py:216`).
- If a chunk doesn't actually support a claim, drop the claim.
- "Not in the corpus" is a valid answer. Say so plainly. Suggest indexing the right path if
  the user expected it to be there.

## What this skill is not

- Not a code editor. Use the host's `read` / `edit` / `write` tools for that, after you've
  pulled the right context through `lilbee_search`.
- Not a web search. `lilbee_crawl` fetches a specific URL into the corpus on the user's
  request; it is not a general fallback when search misses.
- Not a chat completion. lilbee can run its own chat model locally, but as a tool consumer
  you stay on the host's model.
