# Retrieval pipeline review: feat/retrieval-consolidated (PR #557)

Full-branch review of the retrieval consolidation at 1791c20f against merge-base 6fa73d74, run as a 34-agent pass: 8 subsystem reviewers (fusion, query pipeline, neighbors, structural filter, store/indexes, concepts/entities, ingest/chunking, rerank/embed), one state-of-the-art gap analyst grounded in current IR research and open-source RAG systems, adversarial verification of every finding (empirical probes against the pinned LanceDB where possible), and a completeness critic. Prior audits (the 21-agent subsystem review and the 11-claim product-correctness audit) were excluded from re-reporting; everything below is new.

## Verdict

Better: the default configuration is verified rank-identical to main, so nothing on this branch regresses retrieval for existing users. The fusion core is more rigorous than typical open-source stacks (fixed weight-budget normalization, deterministic ties, documented anti-decisions), the pipeline stage ordering (retrieve, filter, diversify, rerank, select, budget-fit, widen) is textbook-correct where several popular frameworks get it wrong, and intent routing, guardrailed expansion, concept-graph boost, and neighbor expansion put the query side at or above parity with LlamaIndex/Haystack/RAGFlow defaults.

Worse: the opt-in adaptive fusion mechanism has two genuine defects that would invalidate the eval meant to judge it; the title arm is fed low-quality titles (no junk-stem guard, frontmatter and EXIF derivation broken) and its overfetch starves on long documents; the structural filter operates at the wrong layer (query-time text regex instead of ingest-time layout classification) and has more false-positive modes than the one already filed; and the largest available precision lever, cross-encoder reranking, ships effectively off. None of these affect defaults; all are fixable.

## Confirmed findings

24 findings survived adversarial verification; none were refuted. Four are major, all in opt-in or feature-gated paths.

### Major

1. **Adaptive fusion at scale zero hard-drops BM25-supported rows.** `fuse_arms` skips merging the FTS arm entirely when the effective lexical weight is exactly 0, which the adaptive scaler produces across its whole saturated region. Rows BM25 matched then carry no `bm25_score`, so the max-distance cut and dedup filter treat them as vector-only and delete them. A down-weight becomes a hard drop, discontinuously (weight 0.001 keeps the row, 0.0 deletes it), and it deletes exactly the exact-identifier hits hybrid retrieval exists to preserve. `src/lilbee/data/store/fusion.py:156`, `core.py:89-104`, `dedup.py:118-127`.

2. **The adaptive confidence margin is a function of candidate-pool depth, not query confidence.** The margin compares top-1 against the mean of the entire retrieved tail; deeper pools have worse tails, so the same query gets a different lexical weight at different retrieval depths (probe: scale 0.74 at depth 5, 0.0 at depth 60). Configuring a reranker deepens every pool to `rerank_candidates`, so adaptive fusion plus a reranker silences the lexical arms almost always, and combined with finding 1 the BM25-unique hits never reach the reranker at all. `fusion.py:73`, `searcher.py:911`. Both fusion findings must be fixed before the MS MARCO eval that decides this feature's default, or the eval measures a depth artifact.

3. **The title arm's bounded overfetch is starved by long documents.** The per-document collapse overfetches a fixed bound of chunk rows; a long PDF whose chunks saturate that bound crowds every other title-matching document out of the arm, silently. `core.py:790`.

4. **Image EXIF title derivation runs a hidden full OCR pass and never yields a title.** On the pinned kreuzberg the metadata read triggers full Tesseract extraction per image and the title field still comes back empty, so ingest pays a large cost for nothing. `extract.py:511`.

### Minor (grouped)

- **Score-scale distortions.** Adaptive scaling caps fused scores at 1/weight_total so `min_relevance_score` over-filters precisely the confident queries; enabling `title_search` deflates every fused score ~20 percent, silently re-tuning the same threshold. Incidental title-word matches also grant the max-distance exemption meant for real lexical support.
- **Neighbor merge heuristics.** The verbatim suffix-prefix overlap dedup fails on heading-context markdown chunks (lilbee's primary content type): merged passages duplicate the chunker overlap and splice heading breadcrumbs mid-text. The unbounded match can also delete real text on coincidental seams, and the shed loop re-merges the whole span per iteration (quadratic at high expansion settings). The durable fix is storing window/parent info at ingest, where the exact contiguous text is known.
- **Structural filter.** Beyond the filed banner false positive: dotted-leader data pages (price lists, financial tables, logs) classify as TOCs; acronym-dense mixed-case prose trips the caps gate; the most common real TOC renderings (spaced dots, ellipsis, leaderless) slip through; and neighbor expansion re-imports the very text the filter dropped. The realized precision win is a thin slice of the risk. The right layer is ingest-time layout classification persisted as chunk metadata.
- **Store lifecycle.** The first search in a fresh process takes the cross-process write lock to build scalar indexes and can raise under a long concurrent ingest; enabling `title_search` at runtime is a silent no-op until restart or next ingest.
- **Concepts/entities.** Two `rebuild_clusters` early-return paths still leave a stale graph; one deterministically failing LLM batch makes every sync redo the entire corpus entity extraction; the concept boost can promote a vector-only TOC into the structural filter's exempt top rank.
- **Title quality.** No junk-stem guard, so filenames like `IMG 1234` are indexed at full title weight; markdown H1 derivation fails on YAML frontmatter and BOM and ignores the frontmatter `title:` field, which is the dominant Obsidian shape.
- **Docs.** architecture.md still documents adaptive fusion as on by default.

## Composition angles still open

The critic found three interactions no reviewer traced, unverified but plausible: (1) with adaptive fusion on, per-variant score scales are incomparable across the multi-query/HyDE merge, systematically demoting hits from the confident variant; (2) the known-item router still resolves only against filename stems and never reads the titles this branch ingests, so the flagship "summarize [Title]" case does not benefit; (3) stores upgraded from main carry NULL titles for pre-existing rows, leaving the title arm permanently lopsided toward newly ingested documents until a full re-ingest.

## Standing vs open-source state of the art

At or above parity: hybrid dense+BM25+title fusion with weighted RRF and an evidence-backed rank-vs-score choice, intent routing (known-item, aggregate), guardrailed multi-query expansion and HyDE, concept-graph boost with community structure, neighbor expansion, set-cover context selection, grounded refusal. Correctly absent for a laptop-scale system: SPLADE, ColBERT/PLAID late interaction, full agentic retrieval loops, RAPTOR hierarchies (wiki pages already cover most of that surface).

Highest-value gaps, ranked by impact against effort under local-first constraints:

1. **Reranking as the realized default.** The whole mechanism exists but is off unless the user configures a model. A small CPU cross-encoder offered at setup (never silently pulled) is the single largest precision lever in current practice. Gate the default flip on the PR #581 harness.
2. **Embedding prefix profiles for nomic and bge-v1.5 families.** nomic-embed requires task prefixes and measurably degrades without them; profiles cover qwen3/instruct/e5 today. Small change; needs index metadata recording which profile embedded the corpus.
3. **Title/metadata injection into the chunk embedding and BM25 input.** Near-free recall win once the junk-title guard exists; cheap sibling of contextual retrieval.
4. **Contextual chunk enrichment at ingest** (LLM-generated situating sentence per chunk, Anthropic-style): the largest reported recall gain that fits local-first, as an opt-in high-quality ingest mode gated on the eval harness.
5. **Language-aware FTS analyzer.** Both FTS indexes use English stemming defaults; language detection already exists and LanceDB exposes analyzer configuration.
6. **Query-time near-duplicate suppression** (normalized-text hash) and an **absolute rerank-score floor** so the reranker can abstain, not just reorder.
7. **Fusion calibration on the harness:** re-run RRF vs properly normalized convex fusion (the in-repo experiment used the known-weak max-normalized variant), sweep RRF k and the title weight.

## Course of action

1. Fix the two adaptive-fusion mechanism defects (findings 1 and 2) before running the MS MARCO eval; they bias it directly.
2. Land the small high-yield fixes: junk-title guard plus frontmatter parsing, EXIF path cost, prefix profiles, language-aware analyzer, doc correction.
3. Decide posture on the reranker (offered small CPU model) and title-into-embedding, both gated on the PR #581 harness.
4. Re-architect the two heuristic layers at ingest time when next touched: structural classification as persisted chunk metadata, neighbor windows stored at chunk time.
5. Merge posture: nothing found regresses the default configuration vs main. After the routine main re-sync, the branch is safe to merge with the above tracked as follow-ups.
