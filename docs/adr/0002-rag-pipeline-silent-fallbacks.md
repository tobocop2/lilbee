# 2. Multi-stage RAG pipeline with silent fallbacks

## Status
Accepted

## Context
The search pipeline grew from simple vector search to include BM25 confidence skip, HyDE, cross-encoder reranking, temporal filtering, concept graph boosting, and query expansion. Each enhancement adds a failure mode. A broken reranker should not prevent basic search from working.

## Decision
Every retrieval enhancement follows the same pattern:

1. Guarded by a config flag (`concept_graph`, `hyde`, `reranker_model`, `temporal_filtering`)
2. Wrapped in try/except that returns original results on failure
3. Runs after initial retrieval (does not block the search path)
4. Uses conditional lazy imports with config guards

BM25 confidence skip probes full-text search before LLM expansion. If the top BM25 score >= 0.8 and the gap to the second result >= 0.15, the LLM call is skipped entirely (saves ~50-100ms per query).

## Consequences
- Any single enhancement can fail without breaking search
- Features can be enabled/disabled independently via config
- Pipeline failures are invisible to the user (baseline search still works)
- Trade-off: silent failures can mask real bugs. Logging at DEBUG level helps diagnose.
