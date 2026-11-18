# Wiki internals

How lilbee builds, updates and searches the wiki. For the overview, see [Architecture](../architecture.md).

## Wiki Layer

> Generation quality depends on your library and the chat model. Expect some pages to land in `drafts/` for human review rather than publish direct.

The wiki layer is lilbee's second-order index: a set of linked markdown pages auto-generated from your indexed documents so that concepts and entities which show up across many sources get their own page with citations from every source that mentions them.

### Layout

Under `$LILBEE_DATA/$wiki_dir/` (default `wiki/`):

| Directory | Contents |
|-----------|----------|
| `concepts/` | One page per LLM-identified concept (e.g. `braking-systems.md`) |
| `entities/` | One page per proper-noun entity extracted by NER (e.g. `henry-ford.md`) |
| `drafts/` | Low-faithfulness output and PENDING markers for parse failures or slug collisions. Reviewed via `lilbee wiki drafts accept / reject`. |
| `summaries/` | Landing spot for accepted drafts with no published counterpart and no recorded origin type |
| `archive/` | Pages retired by `lilbee wiki prune` |
| `synthesis/` | Cross-source pages produced by `lilbee wiki synthesize` |
| `index.md` | Auto-generated table of contents, grouped by page type |
| `log.md` | Append-only audit trail of every build, synthesize, ingest, lint, and prune, with a per-run line reporting what the quality gates did |

Slugs are lowercase hyphen-separated filenames that double as the `[[link]]` target. `make_slug` lives at `src/lilbee/core/text.py` (re-exported from `lilbee.wiki`).

### Build

`lilbee wiki build` runs a one-time Phase D migration (archives pre-Phase-D noun-chunk concept pages, unwraps stale `[[concept-slug]]` links), then extracts NER entities from the chunk store via `cfg.wiki_entity_mode` (default `ner_entities`, spaCy NER only). Per source, a single batched LLM call identifies 3-5 concepts worth their own page and drafts a section for each concept plus each extracted entity. Sections are split, citation-verified against the source chunk pool, embedding-faithfulness-scored (`wiki/quality.py::check_faithfulness`, cosine of body vs mean source-chunk vector), and written to `concepts/` or `entities/`. Sections that fail to parse become PENDING markers in `drafts/`.

### Wikification and incremental update

Enabling the wiki never starts generation on its own. A sync regenerates touched pages only when `LILBEE_WIKI_AUTO_UPDATE` is on, in which case `lilbee.wiki.ingest.incremental_update` runs after ingest with `extract_concepts=False` so re-ingest never churns concept slugs. The cap is `LILBEE_WIKI_INGEST_UPDATE_CAP` (default 20 touched wiki pages per sync); past it the sync logs a warning and leaves the rebuild to you. Otherwise wikification is explicit: `lilbee wiki build` / `wiki update`, the `b` (Wikify) binding on the TUI wiki screen, or the HTTP and MCP build endpoints.

A build spends one LLM call per source document, each carrying that document's chunks up to the context budget, so cost scales with library size rather than with how many pages you read. On a laptop a few dozen documents is minutes; a few thousand is hours of sustained GPU. Run large builds on a machine you are not also using.


### Chunk selection inside wiki generation

Wiki generation does not search. NER assigns each extracted entity to the source that mentions it most (its `chunk_refs`), and the batched call for that source is handed that source's own chunks straight from `store.get_chunks_by_source`. Synthesis pages take the chunks of every source in the cluster. In both paths `wiki/page.py::truncate_chunks_to_budget` drops trailing chunks until the prompt plus the output cap (`LILBEE_WIKI_SUMMARY_MAX_TOKENS`) fits the context window, with a quarter-window fallback when the output cap and prompt overhead alone would exceed it. Hybrid search, the reranker, and the per-source diversity cap play no part in what a page is built from.

### `[[wiki links]]`

After each build, `wiki/links.py::rewrite_wiki_links` rewrites plain-text slug surface forms to `[[slug]]` form in page bodies, skipping YAML frontmatter, code fences, and the auto-generated citation block. `lilbee wiki lint` flags concept or entity pages with zero inbound links.

### Lazy generation

A build spends one LLM call per source document, so its cost tracks library size rather than what anyone reads. `wiki/stubs.py` splits that in half: NER already names every entity in the corpus with its chunk refs and spends no call, so the index of pages-that-could-exist is free and refreshes on every sync while the wiki is enabled (not gated on `wiki_auto_update`, which governs generation). `wiki/lazy.py::generate_stub_page` then writes one page for one call.

Only entities are enumerable this way; LLM-curated concepts come from the per-source batched call and still arrive published from an explicit build.

Per-page generation also widens the evidence. `group_entities_by_primary_source` assigns an entity to the source mentioning it most and generates from that source alone, dropping what every other document says. Reading the index gathers the subject's chunks across all of them and resolves each citation against whichever source holds the excerpt. Chunk refs are capped per entity (`wiki_stub_max_chunk_refs`, default 50, already more than one page's context budget admits) so the index stays small.

### Removing a wiki

Disabling `wiki` stops new pages being written; it deletes nothing. `wiki/wipe.py::wipe_wiki` removes the whole wiki directory and then every `chunk_type=wiki` row and every citation row, in that order: rows outliving their page are what the next prune reconciles away, while a page outliving its rows reads as "nothing to do" and is never retried. It is reachable with the wiki disabled from all four surfaces: `lilbee wiki wipe`, `DELETE /api/wiki`, the `wiki_wipe` MCP tool (which needs `confirm=true`), and in the TUI the **Delete wiki** command-palette entry. The wiki screen's `W` binding is the quick route while the wiki is on, but the whole wiki view drops out of the nav once the setting is off, so the palette entry is what carries that state. The TUI also offers the wipe whenever the setting is switched off. A wipe reports whether the row delete actually landed, so a failure is never presented as a completed wipe.

### Search scope

`search()` accepts a `scope` argument (`raw`, `wiki`, `both`) that filters the hybrid search pool to source chunks, wiki chunks, or the union. Used by the TUI scope toggle and the MCP tool.

While `wiki` is off, generated pages are excluded from retrieval regardless of scope: an unscoped search narrows to `raw` (which covers table chunks), and a `wiki`-scoped search returns nothing rather than widening to the whole pool. Disabling the setting deletes no rows, so without this a library wikified once keeps surfacing generated pages in ordinary search.
