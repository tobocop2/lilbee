---
name: lilbee-mcp-wiki
description: Experimental wiki layer for lilbee. Use only when the user explicitly asks about wiki / concept / entity / synthesis pages, or when `lilbee_status` shows a built wiki. Requires the lilbee-mcp skill to be active for the underlying MCP connection.
---

# lilbee-mcp-wiki

The wiki layer generates per-concept and per-entity pages with citations from
the indexed library, then lets you query and lint them. It is still rough; treat
it as opt-in, not part of normal answer flow. Skip everything here unless the
user explicitly asks about wiki / synthesis pages, or `lilbee_status` already
shows a wiki built.

The build / read tools (`lilbee_wiki_list`, `lilbee_wiki_read`,
`lilbee_wiki_build`, `lilbee_wiki_update`, `lilbee_wiki_synthesize`) return
`{"error": "wiki not enabled"}` until the user enables it with
`lilbee_settings_set({"wiki": true})`. The remaining wiki tools operate on the
on-disk wiki directory regardless of the flag and report empty results when
there's nothing to read.

The shared-embedder rule from the parent `lilbee-mcp` skill still applies: any
LLM-bound wiki op (`build`, `update`, `synthesize`) goes through the
`lilbee-worker` subagent.

## Build / refresh (long, must go through `lilbee-worker`)

| Tool | Use |
|---|---|
| `lilbee_wiki_build()` | Generate the full topic / entity wiki from the indexed library. LLM-bound; minutes per source. |
| `lilbee_wiki_update()` | Refresh the wiki after a sync. Currently a full rebuild. |
| `lilbee_wiki_synthesize()` | Generate cross-source synthesis pages (concept clusters spanning ≥3 sources). |
| `lilbee_wiki_prune()` | Archive stale wiki pages whose sources were deleted, and flag pages with mostly-stale citations. |

## Read / inspect (inline)

| Tool | Use |
|---|---|
| `lilbee_wiki_status()` | Page counts, generator settings, last build, and the `wiki_enabled` flag. |
| `lilbee_wiki_list()` | Every wiki page with slug, title, type, source count. |
| `lilbee_wiki_read(slug)` | Read one wiki page's body + frontmatter. |
| `lilbee_wiki_lint(wiki_source)` | Find orphan pages, stale citations, pending drafts. Pass empty `wiki_source` to lint all. |
| `lilbee_wiki_citations(wiki_source)` | Per-section citation coverage for one wiki page. |
| `lilbee_wiki_drafts_list()` | Pending drafts with drift, faithfulness, and pairing info. |
| `lilbee_wiki_drafts_diff(slug)` | Unified diff between a pending draft and the live page. |

## Typical flow

```
lilbee_wiki_status                       # check whether one already exists
lilbee-worker:  lilbee_wiki_build()      # LLM-bound, minutes per source
lilbee_wiki_list                         # see what was generated
lilbee_wiki_drafts_list                  # check what landed in drafts/ for review
```

Query a built wiki via `lilbee_search(..., scope="wiki")`. Run
`lilbee_wiki_lint(wiki_source="")` afterward to surface orphans and stale citations.
