# Agent instructions

This project pairs you with **lilbee**, a local retrieval engine wired in over MCP. Your
search and corpus tools are `lilbee_*`. The built-in `codesearch` / `websearch` / `webfetch`
are intentionally denied here, so you have to read the real code and documents through
lilbee.

## How to work

1. **Look things up with `lilbee_search`.** Don't guess about code or documents you haven't
   read. The query should be the most distinct noun phrase from the question, not the whole
   sentence.
2. **Cite the file and line for every fact you state**, exactly as `lilbee_search` returned
   it. If a claim doesn't trace back to a chunk, drop it.
3. **If the answer isn't in the corpus, say so.** Don't invent.
4. **Long-running lilbee operations go to the `lilbee-worker` subagent**, via the `task`
   tool. That keeps you responsive. The long ops are:
   - `lilbee_add` (indexing files / folders)
   - `lilbee_sync` (re-indexing after changes)
   - `lilbee_crawl` (fetching a docs site, then polling `lilbee_crawl_status`)
   - `lilbee_model_pull` (downloading a model)

   Quick lookups (`lilbee_search`, `lilbee_status`, `lilbee_list_documents`,
   `lilbee_model_list`, `lilbee_model_show`) you run inline.
5. **When asked to index something**, delegate the `lilbee_add` to the `lilbee-worker`
   subagent and wait for it to report done before you start answering. The worker handles
   `lilbee_init` too. `lilbee_add` is a SHA-dedupe — if the path is already indexed, it's
   instant; so you can call it unconditionally instead of probing first.

The full tool surface and the quick vs. long split are in the bundled `lilbee-mcp` skill
(`.opencode/skills/lilbee-mcp/` or `.claude/skills/lilbee-mcp/`).
