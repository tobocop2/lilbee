# Agent instructions

This project pairs you with **lilbee**, a local retrieval engine wired in over MCP. Every
question about a file in this project, every claim about code or documents, every "what
does X say about Y" goes through `lilbee_*` tools. The built-in `codesearch`, `websearch`,
and `webfetch` are deliberately denied here so you can't shortcut around the corpus.

## Hard rule: never search while indexing

`lilbee_search` and indexing (`lilbee_add`, `lilbee_sync`, `lilbee_crawl`,
`lilbee_model_pull`) share a single in-process embedder worker inside the lilbee MCP
server. They cannot run at the same time. If you call `lilbee_search` while indexing is
still in progress, the call will hang and your client will time it out.

**Sequence every task this way:**

1. Decide whether the corpus needs indexing or updating. If yes, delegate the long op to
   the `lilbee-worker` subagent via `task` and **wait for it to return**. Do not call any
   `lilbee_*` tool from your own thread while the worker is running. Do not start a parallel
   thought to "save time"; there is nothing to do until the worker reports back.
2. After the worker returns, run `lilbee_status` once to confirm the expected source /
   chunk counts before searching.
3. Then call `lilbee_search` (and friends) freely; the embedder is no longer pinned.

If you skip step 2 and your first `lilbee_search` returns an MCP timeout, that's the
indexing finishing up. Wait 10 seconds, re-check `lilbee_status`, and retry the search.
Do not change retrieval strategy or fall back to other tools; the search will work the
moment the embedder is free.

You will spend most of your time on two things:

1. **Setting the corpus up** — indexing files, crawling a docs site, swapping models. These
   are long-running, so they go to the `lilbee-worker` subagent via the `task` tool and you
   wait for it to report done.
2. **Talking to the corpus** — searching it, listing what's in it, checking status. These
   are fast, so you run them inline.

## Querying the corpus (inline, fast)

1. **`lilbee_search` is your primary research tool.** Use it before answering any question
   about the project. The query should be the most distinct noun phrase from the user's
   question, not the whole sentence ("oil capacity", not "what's my oil capacity?"). Run
   multiple searches when the answer needs more than one anchor.
2. **Cite the file and line (or page) for every fact you state**, exactly as
   `lilbee_search` returned it. If a claim doesn't trace back to a chunk, drop it.
3. **If the answer isn't in the corpus, say so.** Don't invent. Don't fall back to general
   knowledge without flagging the switch explicitly.
4. **Other inline reads:** `lilbee_status` (what's indexed and which models are wired in),
   `lilbee_list_documents` (the file list), `lilbee_model_list` / `lilbee_model_show`
   (model surface), `lilbee_crawl_status` (poll an in-flight crawl).

## Setting the corpus up (delegate to lilbee-worker)

Long-running operations block the chat thread, so they go to the `lilbee-worker` subagent
via the `task` tool. Wait for the worker to report done, then continue answering.

- `lilbee_add` — index a file or folder. SHA-dedupes, so calling it on an already-indexed
  path is instant; you can call it unconditionally instead of probing first. The worker
  runs `lilbee_init` for you if no `.lilbee/` exists.
- `lilbee_sync` — re-index after the watched paths change.
- `lilbee_crawl` — fetch a docs site (worker polls `lilbee_crawl_status` to completion).
- `lilbee_model_pull` — download a model from Hugging Face.

## Writing code from an indexed API reference

When the corpus is an API or class reference (Godot XML, a library's source, a docs site)
and the user asks you to write code against it, your training data is outdated relative to
the corpus. Follow this four-step workflow on every code-generation task:

1. **Plan** — list every class, method, property, and enum your solution will need.
2. **Search** — call `lilbee_search` for each item individually. Do not skip this for any
   class, even ones you're confident about. Check whether each class still exists or was
   replaced by a newer variant. Verify every method signature individually.
3. **Write** — only use class names, methods, and properties exactly as confirmed by the
   search results. If something wasn't found, do not use it.
4. **Verify** — before saving the file, re-search every class and method call in your code
   via `lilbee_search`. Fix anything that doesn't match the search results exactly.

This is the difference between a 0-hallucinated-API run and a 4-hallucinated-API run in
the [godot-level-generator benchmark](docs/benchmarks/godot-level-generator.md).

## End-to-end shape of a question that needs new content

User asks "what does X say about Y?" against a file you haven't indexed yet:

1. Delegate `lilbee_add(<path>)` to `lilbee-worker`. Wait for "done".
2. Run `lilbee_search("Y")` inline.
3. Answer in your own words, citing `<file>:<line>` (or `<file>:p<N>` for PDFs) for each
   fact. If the search returned nothing useful, say so plainly.

The full tool surface and the quick / long split are in the bundled `lilbee-mcp` skill
(`.opencode/skills/lilbee-mcp/` or `.claude/skills/lilbee-mcp/`).
