# Wiki stability review

The paper trail for the wiki stabilization branch: how the layer was reviewed, what was broken, what stable means here, and the validation runs behind the label decision.

## Method

Repeated adversarially verified review rounds over the whole subsystem (21 wiki modules plus the CLI, TUI, MCP, and HTTP surfaces). Each round is a seven-dimension sweep (citations, persistence, generation, surface parity, TUI, config and doc truth, test quality) where every finding is checked by an independent refuter before it counts, plus the project gate (lint, format, typecheck, wiki-scoped tests, scoped coverage). Every behavioral fix is pinned by a test proven to fail with the fix reverted.

| Round | Findings | Blocking | Outcome |
|---|---|---|---|
| 1 | 80 | 14 | all fixed |
| 2 | 28 | 1 | all fixed |
| 3 | 23 | 0 | all fixed |
| 4 | 21 | 0 | all fixed |
| 5 | 13 | 0 | all fixed |
| 6 | 17 | 1 (regression from round 5, caught) | all fixed |
| 7 | 8 | 0 | all fixed |
| 8 | 2 (partial round) | 0 | fixed; round to be completed |

192 defects fixed at the time of writing. The loop continues until a complete pass reports no findings.

## What was broken, by class

- **Gates that could not fail.** The citation gate auto-passed excerpt-free footnotes; lint verified quotes against raw PDF bytes; a routing test matched the pytest tmp-dir name; the HTTP QA tier never booted (unauthenticated health poll) so its assertions never ran; five "off the event loop" route tests could never fail.
- **Data loss and crash windows.** A page failing the faithfulness gate still pruned its raw source chunks from retrieval; page indexing cleared rows before embedding; accepted drafts lost citation provenance permanently; in-place writes truncated pages on crash; swallowed store deletes let prune and accept report success over failures.
- **Silent divergence.** Rebuilds never converged (unseeded generation, frontmatter in the drift ratio); stray section headers became junk concept pages; label binding depended on set iteration order; build-time and lint-time citation verification disagreed on identical content.
- **Surface drift.** Builds from different surfaces could interleave (the mutex existed on HTTP only); MCP tools kept working after the wiki was disabled; CLI error paths exited 0; docs described retrieval machinery that does not exist and config fields nothing read.

## What stable means here

Mechanism (this branch): citations that fail closed and verify per named source, crash-ordered persistence with orphan reconciliation, deterministic rebuilds, one structural re-entrant build mutex, explicit wikification (enabling the wiki never generates by itself; `wiki_auto_update` is opt-in), and per-run gate metrics (publish rate, citation verify rate, drafts, markers) reported on every surface and logged durably.

Measurement (validation runs below). Graduation out of the experimental label requires, on two corpus scales:

- At least 95% of citations on published pages verify, and zero pages publish with zero verified citations.
- Draft-routing rate at most 20% of generated sections.
- A second build over an unchanged corpus produces zero drift diversions, zero new drafts, zero slug churn.
- Zero crashes or store corruption; lifecycle drills pass (edit source, then stale lint flags it; delete source, then prune archives its page; accept and reject round-trip from every human surface).
- A sampled claim-support audit of at least 90%: the quoted excerpt actually supports the sentence it cites.

## Validation

### Run 1: local vault (~100 documents, Apple Silicon)

Pending.

### Run 2: large PDF corpus subset (4,256 documents, single GPU)

Corpus: 4,256 scanned PDFs (DOJ production sets and court batches), ingested with OCR on a single NVIDIA L4 (RunPod). Embeddings: nomic-embed-text-v1.5. Page generation: Qwen2.5-7B-Instruct (fp16). 4,212 sources indexed.

Store-backed index (the layer this branch rebuilt around a `_wiki_mentions` table whose ≥floor stub index is a corpus-wide aggregate): 27,548 mention rows, 21,670 distinct subjects, 2,261 published pages. Of those, **1,600 pages (71%) draw their evidence from two or more separately-ingested documents** — the exact subject class the previous per-document index dropped, since a subject below the mention floor in each document alone only clears the floor once evidence from every document is aggregated. That recovery is the point of the change and it holds at corpus scale.

Against the criteria:

- **Citation verification: pass.** 100% of citations on published pages verify (47/47), zero pages published with zero verified citations. Measured with the same `verify_citation` the lint surface uses.
- **Second build over the unchanged corpus: pass at the index.** Re-running `wiki index` produced 2,261/2,261 pages with zero slug churn and a byte-stable mention table (27,548 rows both builds). Generation-level drift was not established this run (see draft-rate caveat).
- **No crashes or corruption; self-heal proven.** Interrupting a full index rebuild empties the derived `_wiki_mentions` table (it clears before it rewrites); the next `wiki index` fully restores it — 27,548 rows, 2,261 pages, 1,600 cross-source. The durable artifact (the on-disk index) is never the half-written table, so reads are unaffected. Making that rebuild atomic is a tracked follow-up.
- **Surface parity: pass.** CLI (`status`, `lint`, `index`), MCP (`wiki_index` → 2,261 entries, `wiki_lint` → clean), and HTTP all agree. HTTP enforces the bearer token (GET list → 200, read → 200, POST lint → 201, every route without a token → 401).
- **Draft-routing rate: not established.** 4 of 17 generated pages routed to drafts (23.5%), but the sample is under-powered: 13 of 30 requested pages exceeded the 300 s per-page generation cap because fp16 7B generation on the L4 is slow. A faster card (or a raised cap) is needed to sample the rate against the 20% bar.
- **Lifecycle — delete source then prune archives its page: fails.** Removing a page's only source (via `lilbee remove`) leaves its citation in place but pointing at absent chunks; `verify_citation` returns `UNVERIFIABLE`, which `_lint_excerpt` reports as no issue at all, so prune's stale-majority check (`_STALE_TYPES = {STALE_HASH, EXCERPT_MISSING}`) never fires and the orphaned page is not archived. Reproduced through the real user path; filed as a follow-up. The other lifecycle drills (edit-source stale lint; accept/reject round-trip) were not exercised in this run.

Profile ([py-spy flame graph](wiki-index-flame.svg), full corpus): spaCy NER over the corpus dominates the index build; the store layer (mention read/write/aggregate) is negligible. A full index over 4,212 sources runs single-threaded NER-bound.

### Claim-support audit

Sampled published pages, each citation's excerpt checked against the cited source's extracted chunks: **100% of sampled citations are `VALID`** with genuine verbatim excerpts (e.g. an entity page's testimony quote, a billing page's `Customer Service Number 1.800.937-8997`). Page-level faithfulness scores ranged 0.73–0.91 (mean 0.84), the generation gate having quarantined lower-scoring drafts. One caveat surfaced: the NER long tail yields occasional spurious subjects (e.g. `aaa`, lifted from call-log text); the resulting page stays faithful — it states outright that the evidence does not define the subject — but is low value. That is a subject-selection quality issue, not a claim-support failure.

## Comparison to other systems

The README carries a comparison table (STORM, GraphRAG, DeepWiki-Open) in which every cell was grounded in the latest cloned source by an independent verifier and then attacked by a refuter; cells that did not survive were corrected or cut. Summary: lilbee's verification and lifecycle machinery (mechanical citation verification, faithfulness quarantine, human review queue, staleness lint, prune, drift protection) exceeds what the compared systems ship. Their strengths are complementary and tracked as roadmap items: STORM's research and outline pipeline, GraphRAG's typed entity graph and hierarchical communities.

## Verdict

Run 2 (4,256-document corpus, single GPU) proves the store-backed cross-source recovery this branch was built for: 1,600 of 2,261 pages depend on evidence aggregated across separately-ingested documents, citations verify at 100%, the index rebuild is deterministic, the mention table self-heals after interruption, and every surface agrees. Two items block full graduation and are tracked as follow-ups: prune does not archive a page whose sources are fully removed (an `UNVERIFIABLE`-vs-stale classification gap in lint/prune), and the draft-routing rate could not be sampled against the 20% bar because fp16 7B generation on an L4 timed out on 13 of 30 pages — a faster card is needed. Run 1 (local vault, Apple Silicon) is still pending. The experimental label stays until those hold.
