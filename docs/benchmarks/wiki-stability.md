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

### Run 2: large PDF corpus subset (~2,000 documents, single GPU)

Pending.

### Claim-support audit

Pending.

## Comparison to other systems

The README carries a comparison table (STORM, GraphRAG, DeepWiki-Open) in which every cell was grounded in the latest cloned source by an independent verifier and then attacked by a refuter; cells that did not survive were corrected or cut. Summary: lilbee's verification and lifecycle machinery (mechanical citation verification, faithfulness quarantine, human review queue, staleness lint, prune, drift protection) exceeds what the compared systems ship. Their strengths are complementary and tracked as roadmap items: STORM's research and outline pipeline, GraphRAG's typed entity graph and hierarchical communities.

## Verdict

Pending validation. The experimental label stays until the criteria above hold on both validation runs.
