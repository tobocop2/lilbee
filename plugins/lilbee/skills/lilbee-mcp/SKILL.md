---
name: lilbee-mcp
description: Searches the user's local lilbee knowledge base (indexed code, documents, PDFs and crawled pages) and answers with file and line citations. Use when the user asks about their own files, code or docs, or asks to index, crawl or remove content. Works through lilbee MCP tools when connected, or the lilbee command line with --json when not.
---

# lilbee

[lilbee](https://github.com/tobocop2/lilbee) is a local retrieval engine. It indexes the
user's files into a `.lilbee/` store. The data stays on the user's machine. Only the text
you quote leaves it. By default, search may ask the chat model for query variants (it skips
very short queries), so it can be slow while that model is busy.

## When to use it

Reach for lilbee first when the user asks where something is, how something in their own
code or documents works, or what a document says. Web-fetch and file-read tools cannot see
the index. Do not use lilbee as a web search or as a code editor.

## How to reach it

Pick one path and stay on it.

- **MCP.** If your tool list has tools from the lilbee MCP server, call them. The host adds
  its own prefix to the tool name, so look for a tool whose name contains `lilbee` and ends
  in the tool name, such as `search` (for example `lilbee_search`, `mcp__lilbee__search` or
  `mcp__plugin_lilbee_lilbee__search`). The table below uses the `lilbee_` form.
- **CLI.** Otherwise run `lilbee --json <command>` in a shell. Each command prints one JSON
  object per line. An error exits non-zero with `{"error": "..."}`. If `lilbee` is not on
  the PATH, tell the user to install it. Do not guess another path.

| Step | MCP | CLI |
|---|---|---|
| Check the library | `lilbee_status` | `lilbee --json status` |
| Search | `lilbee_search` | `lilbee --json search "query" --top-k 12` |
| Add content | `lilbee_add` | `lilbee --json add <path or URL>...` |
| Re-index after edits | `lilbee_sync` | `lilbee --json sync` |
| Remove content | `lilbee_remove` | `lilbee --json remove <name, folder or glob>` |
| List sources | `lilbee_list_documents` | `lilbee --json status --list-sources` |

The MCP server sends the description of every tool. Run `lilbee --json <command> --help`
for CLI options. These tools also exist, and each description says what it does:

- **Crawl and create.** `lilbee_crawl` starts a non-blocking crawl and returns a
  `task_id`. Poll `lilbee_crawl_status`, or stop it with `lilbee_crawl_cancel`.
  `lilbee_init` creates a `.lilbee/` in a directory and switches the session to it.
- **Models.** `lilbee_model_list`, `lilbee_model_show`, `lilbee_catalog_browse`,
  `lilbee_model_pull` (long) and `lilbee_model_rm`. Check the `compat` field from
  `lilbee_catalog_browse` before a pull.
- **Datasets.** `lilbee_export_dataset` writes a per-page dataset with no embedding.
  `lilbee_import_dataset` re-embeds every page (long).
- **Sessions.** `lilbee_sessions_list`, `lilbee_session_get`, `lilbee_session_create`,
  `lilbee_session_fork`, `lilbee_session_add_message`, `lilbee_session_set_summary`,
  `lilbee_session_rename` and `lilbee_session_delete`. They cover your own sessions only.
  Ask before you take over a session id the user gives you (`claim=true` on
  `lilbee_session_add_message`). They appear only when `mcp_sessions_enabled` is on.
- **Memory.** `lilbee_memory_remember`, `lilbee_memory_recall`, `lilbee_memory_list` and
  `lilbee_memory_forget` keep durable notes. They appear only when `memory_enabled` is on.
- **Profiles.** `lilbee_profile_list`, `lilbee_profile_show`, `lilbee_profile_apply`,
  `lilbee_profile_manage`, `lilbee_analyze` and `lilbee_analyze_dismiss` read and change
  the project's profile. They appear only when `mcp_profiles_enabled` is on. Open the
  settings reference before you use them.
- **Server presets.** `lilbee_preset_list`, `lilbee_preset_add`, `lilbee_preset_edit` and
  `lilbee_preset_remove` store OpenAI-compatible servers. They appear only when
  `mcp_presets_enabled` is on. Open the settings reference before you use them.

`lilbee_analyze` reads the indexed documents, or an absolute `directory` on the server. It
can take a while on a large corpus. Treat it as a long operation.

## The core loop

1. Check the library once per session. If `total_chunks` is 0, the index is empty: say so
   and offer to add content. Do not search. If the status reports an upgrade in progress
   instead, wait and check again.
2. Search with the user's own words. Pass `top_k` (`--top-k`) on the call to widen a thin
   result, for example 20. Keep it near 20 or lower for small-context models.
3. Answer from the returned chunks only. Cite each claim.
4. If the first result is thin, search again with a more specific noun phrase or a larger
   `top_k`. Do not change settings to do this.

## Long operations

Adding, syncing, crawling, importing, model downloads and wiki builds can take minutes.
They hold the embedder that search also needs. A search sent meanwhile hangs until the
host times out.

- If your host has a `lilbee-worker` or any subagent or background-task mechanism, hand the
  long operation to it and wait for it to return.
- Otherwise run the operation in the foreground and wait for it to finish.
- Send no search, and no other lilbee call, while it runs.
- When it returns, check the library again, then search.

If a search times out, indexing is still running. Wait about 10 seconds, check the library,
and retry the search on the same path.

## Citation rules

- Every fact you state must trace to a chunk that search returned. Cite the source file and
  the line range the chunk reported, exactly as returned (for example
  `src/app/pipeline.py:216-247`). The CLI wraps the chunks in a `results` list. When
  `line_start` and `line_end` are 0, the chunk has no line range: cite the file, and the
  page for a PDF.
- If a chunk does not support a claim, drop the claim. Search again before you guess.
- "Not in the indexed files" is a valid answer. Say so, and offer to index the right path.
- The user may open your citation in an editor. Never change a reported line range.

## Runbook

- **Search returns nothing.** Try a more specific noun phrase. If still empty, list the
  sources to see whether the content is indexed.
- **A command or tool returns an error.** Show the error to the user verbatim. Do not retry
  silently.
- **A model download failed part way.** Run the same command again. Finished files are skipped.
- **Remove content.** `remove` drops chunks and keeps the source files. Wiping the whole
  index (`lilbee_reset`, `lilbee --json reset --yes`) is destructive. Do it only when the
  user asks.

## Settings and models

Change settings, models or GPU placement only when the user asks. A change is saved to
disk and every client of the same lilbee sees it. Do not change a setting to improve one
search and reset it afterwards. Use the per-call arguments above.

When the user does ask, open the reference file for the task:

- [reference-tuning.md](reference-tuning.md): retrieval tuning and model setup.
- [reference-settings.md](reference-settings.md): settings that change the tool list,
  secrets, and GPU placement.

The wiki tools (generated concept pages) are in the separate `lilbee-mcp-wiki` skill. Load
it only when the user asks about wiki pages.
