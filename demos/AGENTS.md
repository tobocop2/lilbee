# Agent instructions

This project pairs you with **lilbee**, a local retrieval engine wired in over MCP. Every
question about a file in this project, every claim about code or documents, every "what
does X say about Y" goes through `lilbee_*` tools. The built-in `codesearch`, `websearch`,
and `webfetch` are deliberately denied here so you can't shortcut around the library.

## Hard rule: never search while indexing

`lilbee_search` and indexing (`lilbee_add`, `lilbee_sync`, `lilbee_crawl`,
`lilbee_model_pull`) share a single in-process embedder worker inside the lilbee MCP
server. They cannot run at the same time. If you call `lilbee_search` while indexing is
still in progress, the call will hang and your client will time it out.

**Sequence every task this way:**

1. Decide whether the library needs indexing or updating. If yes, hand the long op to the
   `lilbee-worker` subagent — opencode supports `@lilbee-worker` mention syntax, Claude
   Code uses the Task/Agent tool. Use whichever your host supports. **Wait for the worker
   to report back** before doing anything else. Do not call any `lilbee_*` tool from your
   own thread while the worker is running. Do not start a parallel thought to "save time";
   there is nothing to do until the worker reports done.
2. After the worker returns, run `lilbee_status` once to confirm the expected source /
   chunk counts before searching.
3. Then call `lilbee_search` (and friends) freely; the embedder is no longer pinned.

If you skip step 2 and your first `lilbee_search` returns an MCP timeout, that's the
indexing finishing up. Wait 10 seconds, re-check `lilbee_status`, and retry the search.
Do not change retrieval strategy or fall back to other tools; the search will work the
moment the embedder is free.

You will spend most of your time on two things:

1. **Setting the library up.** Indexing files, crawling a docs site, swapping models.
   These are long-running, so they go to the `lilbee-worker` subagent (your host's
   subagent invocation: `@lilbee-worker` in opencode, the Task / Agent tool in Claude
   Code). You wait for it to report done.
2. **Talking to the library.** Searching it, listing what's in it, checking status. These
   are fast, so you run them inline.

## Querying the library (inline, fast)

1. **`lilbee_search` is your primary research tool.** Use it before answering any question
   about the project. The query should be the most distinct noun phrase from the user's
   question, not the whole sentence ("oil capacity", not "what's my oil capacity?"). Run
   multiple searches when the answer needs more than one anchor.
2. **Cite the file and line (or page) for every fact you state**, exactly as
   `lilbee_search` returned it. If a claim doesn't trace back to a chunk, drop it.
3. **If the answer isn't in the library, say so.** Don't invent. Don't fall back to general
   knowledge without flagging the switch explicitly.
4. **Other inline reads:** `lilbee_status` (what's indexed and which models are wired in),
   `lilbee_list_documents` (the file list), `lilbee_model_list` / `lilbee_model_show`
   (model surface), `lilbee_crawl_status` (poll an in-flight crawl).

## Setting the library up (delegate to lilbee-worker)

Long-running operations block the chat thread, so they go to the `lilbee-worker` subagent
through your host's subagent mechanism (opencode: `@lilbee-worker` mention; Claude Code:
the Task / Agent tool). Wait for the worker to report done, then continue answering.

- `lilbee_add` — index a file or folder. SHA-dedupes, so calling it on an already-indexed
  path is instant; you can call it unconditionally instead of probing first. The worker
  runs `lilbee_init` for you if no `.lilbee/` exists.
- `lilbee_sync` — re-index after the watched paths change.
- `lilbee_crawl` — fetch a docs site (worker polls `lilbee_crawl_status` to completion).
- `lilbee_model_pull` — download a model from Hugging Face.

## Writing code against an indexed API reference

**This is the most important section for code-generation tasks.** When the indexed files
are an API or class reference (Godot XML, a software package's source, a docs site,
vendor SDK headers), assume your training data is outdated relative to those files and
follow the Plan / Search / Write / Verify workflow below on every code-generation task.
**Do not skip steps to "save time."** The benchmark difference is concrete: 0 hallucinated
APIs with this workflow vs 4 hallucinated APIs without, in the
[godot-level-generator benchmark](docs/benchmarks/godot-level-generator.md).

1. **Plan.** Before writing any code, list every class, method, property, enum, signal,
   and constant your solution will need. Be exhaustive: if your solution touches 12 named
   things, plan for 12 searches.
2. **Search.** Call `lilbee_search` for each item from your plan **individually**. Do not
   batch them into a single query string. Do not skip an item because you're confident
   about it; the point of the workflow is to verify confidence against the indexed reference.
   For each search:
   - Use the exact class or method name as the query.
   - If a class returns nothing, it was renamed or removed in this version. Search for
     near-matches.
   - For each method, confirm the exact signature (parameter order, types, defaults,
     return type) from the chunk text.
3. **Write.** Only use class names, methods, properties, and constants exactly as
   confirmed by the search results. If a name wasn't found, do not use it. If a signature
   doesn't match what you'd write, rewrite the code to fit the confirmed signature, not
   the other way around.
4. **Verify.** Before the final write of the file, re-search every name in the code you
   just wrote. Treat this as an audit, not a sanity check. Fix any mismatch by rewriting
   the code, not by ignoring the search result.

If you find yourself thinking "I already know this API, I can skip step 2 for this one,"
that is the exact moment the benchmark says you start hallucinating. Run the search.

## End-to-end shape of a question that needs new content

User asks "what does X say about Y?" against a file you haven't indexed yet:

1. Hand `lilbee_add(<path>)` to `lilbee-worker` via your host's subagent mechanism
   (`@lilbee-worker` in opencode). Wait for "done".
2. Run `lilbee_search("Y")` inline.
3. Answer in your own words, citing `<file>:<line>` (or `<file>:p<N>` for PDFs) for each
   fact. If the search returned nothing useful, say so plainly.

The full tool surface and the quick / long split are in the bundled `lilbee-mcp` skill
(`.opencode/skills/lilbee-mcp/` or `.claude/skills/lilbee-mcp/`).
