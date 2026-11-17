---
name: lilbee-mcp-wiki
description: Builds, reads and lints the generated lilbee wiki (concept, entity and synthesis pages with citations). Use only when the user asks about wiki, concept, entity or synthesis pages, or when lilbee status shows a built wiki. Needs the lilbee-mcp skill for how to reach lilbee.
---

# lilbee wiki

The wiki layer writes per-concept and per-entity pages, with citations, from the indexed
library. It is opt-in and rough. Skip it unless the user asks about wiki pages or the
library status already shows a built wiki. The `lilbee-mcp` skill says how to reach lilbee
(MCP tools or `lilbee --json`) and how to cite.

## Availability

`lilbee_wiki_status` (CLI `lilbee --json wiki status`) always works. It reports
`wiki_enabled` and the page counts. `lilbee_wiki_wipe` is always present too, because
turning the wiki off leaves the generated pages in place. Every other wiki tool is absent
from the MCP tool list when the server started with the wiki off, and returns
`{"error": "wiki not enabled"}` if the wiki is turned off while it runs. Enabling the wiki
with a settings change takes effect after the user restarts the MCP server. The CLI
commands do not need a restart.

With the wiki off, the CLI commands that write (`build`, `update`, `synthesize`, `prune`,
`index`, `generate`, `drafts accept`, `drafts reject`) exit with "Wiki is disabled (set
wiki = true in settings)". The CLI has no settings command: ask the user to set
`wiki = true` in `config.toml` or on the `/settings` screen of `lilbee chat`.

## Commands

Each MCP tool has a CLI twin: `lilbee_wiki_<name>` is `lilbee --json wiki <name>`.

| Task | Name | Cost |
|---|---|---|
| Build the whole wiki | `build` (`--dry-run` previews candidates) | Long, one LLM call per source |
| Refresh after a sync | `update` | Long, currently a full rebuild |
| Cross-source pages | `synthesize` (concept clusters with three or more sources) | Long |
| Rebuild the browse index | `index` | Short, no LLM call |
| Write one indexed page | `generate <slug>` | One LLM call |
| Archive stale pages | `prune` | Short |
| Delete every generated page | `wipe` (MCP `confirm=true`, CLI `--yes`) | Short, cannot be undone |
| Inspect | `status`, `list`, `read <slug>`, `lint`, `citations`, `drafts list`, `drafts diff <slug>` | Short |

A dry run of `build` makes no LLM call, so it is short.

## Flow

1. Read the wiki status. If a wiki exists, skip the build.
2. Run `build` as a long operation: use a subagent or background mechanism if the host has
   one, otherwise run it in the foreground and wait. Do not search meanwhile. See the
   long-operation rule in the `lilbee-mcp` skill.
3. List the pages, then check the drafts list for pages that await review.
4. Search the wiki with `lilbee_search(..., scope="wiki")` or
   `lilbee --json search "query" --scope wiki`. Cite wiki pages as the search reports them.
5. Run `lint` to find orphan pages and stale citations.

Accepting or rejecting a draft changes the wiki the user sees. Do it only when the user
asks. The MCP tools only list and diff drafts. Accept and reject are CLI only:
`lilbee --json wiki drafts accept <slug>` and `lilbee --json wiki drafts reject <slug>`.
Run `wipe` only when the user asks.
