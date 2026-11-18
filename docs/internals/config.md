# Configuration internals

The core, retrieval and provider settings. The [settings reference](../settings.md) lists every setting. For the overview, see [Architecture](../architecture.md).

## Configuration Reference

All settings are configurable in `config.toml`. All except `linked_roots` are also configurable via `LILBEE_*` environment variables, and the writable ones via `/set` in chat mode. The `GET /api/config` endpoint exposes all current values for API clients.

### Core Settings

| Setting | Default | Description | Caveats |
|---------|---------|-------------|---------|
| `LILBEE_CHAT_MODEL` | `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` | Model used for `ask`, `chat`, and wiki generation | Native GGUF by default; with `[providers]`, any remote name the SDK backend understands |
| `LILBEE_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf` | Model for computing vector embeddings | Changing this requires a full `lilbee rebuild` |
| `LILBEE_VISION_MODEL` | *(none)* | GGUF vision model for OCR (served with `--mmproj`) | When set, takes precedence over Tesseract for scanned PDFs and images |
| `LILBEE_TOP_K` | `12` | Number of search results returned | Higher values provide more context but increase LLM latency and token cost |
| `LILBEE_MAX_DISTANCE` | `0.75` | Cosine distance cutoff for vector results | Lower values are stricter (fewer but more precise results). Set to 1.0 to disable filtering. |
| `LILBEE_ADAPTIVE_THRESHOLD` | `false` | Enable adaptive threshold widening | When true, automatically widens distance threshold if too few results found. Useful for ensuring minimum result count. |
| `LILBEE_CHUNK_SIZE` | `512` | Target tokens per chunk | Changing requires `lilbee rebuild`. Smaller = more precise retrieval, larger = more context per chunk |
| `LILBEE_CHUNK_OVERLAP` | `100` | Overlap tokens between adjacent chunks | Changing requires `lilbee rebuild`. Prevents information loss at chunk boundaries |
| `LILBEE_SYSTEM_PROMPT` | *(built-in)* | System prompt sent to the LLM | Override per-project for domain-specific behavior |

### Retrieval Quality Settings

| Setting | Default | Description | Caveats |
|---------|---------|-------------|---------|
| `LILBEE_MMR_LAMBDA` | `0.5` | Relevance vs diversity (0.0=diverse, 1.0=relevant) | 0.5 is the standard default from Carbonell & Goldstein 1998. Lower for broad exploratory queries. |
| `LILBEE_DIVERSITY_MAX_PER_SOURCE` | `5` | Max chunks returned per source document | Lower = more diverse sources. Higher = deeper coverage of a single document. |
| `LILBEE_CANDIDATE_MULTIPLIER` | `3` | Vector-only candidate pool multiplier feeding MMR | Higher = better diversity selection but slower; hybrid search ignores it. |
| `LILBEE_QUERY_EXPANSION_COUNT` | `3` | Number of LLM-generated query variants | Each variant requires an embedding call. Set to 0 to disable expansion entirely for fastest search. |
| `LILBEE_ADAPTIVE_THRESHOLD_STEP` | `0.2` | Distance filter widening increment | Only used when `LILBEE_ADAPTIVE_THRESHOLD=true`. Smaller = more granular adaptation but more filter iterations |
| `LILBEE_EXPANSION_SKIP_THRESHOLD` | `0.8` | BM25 confidence above which expansion is skipped | Confidence is s/(s+5) of the raw top BM25 score, unsaturated. 0 disables the skip. |
| `LILBEE_EXPANSION_SKIP_GAP` | `0.15` | Min relative gap between top-1 and top-2 raw BM25 to skip expansion | Fraction of the top score. Ensures the match isn't ambiguous. |
| `LILBEE_EXPANSION_GUARDRAILS` | `true` | Validate expansion variants for drift | Prevents hallucinated variants at the cost of potentially filtering valid creative expansions |
| `LILBEE_EXPANSION_SIMILARITY_THRESHOLD` | `0.5` | Minimum question↔variant cosine similarity for an expansion variant to survive the guardrail | Raise for stricter filtering, lower to keep more variants. Calibrate per embedding model. |
| `LILBEE_MAX_CONTEXT_SOURCES` | `5` | Max chunks included in LLM context | More = more complete answers but higher latency and token cost |
| `LILBEE_HYDE` | `false` | Enable hypothetical document embeddings | Adds ~500ms per query. Best for vague queries where terminology doesn't match docs. |
| `LILBEE_HYDE_WEIGHT` | `0.7` | Weight for HyDE results relative to original | Lower = less trust in hypothetical documents. 0.7 prevents fabricated vectors from dominating. |
| `LILBEE_RERANKER_MODEL` | `""` | Cross-encoder model for reranking (empty = disabled) | Native GGUF (e.g. `bge-reranker-v2-m3`) or a remote name via the SDK backend. Only loaded when configured. |
| `LILBEE_RERANK_CANDIDATES` | `60` | Number of candidates to rerank | More = deeper pool but slower. |
| `LILBEE_RERANK_BLEND` | `true` | Blend reranker scores with retrieval fusion, position-aware | Off = pure cross-encoder ordering, useful when measuring the reranker in isolation. |
| `LILBEE_RERANK_MIN_SCORE` | unset | Absolute floor on raw reranker scores | Candidates below it are dropped, so a uniformly irrelevant pool can trigger grounded refusal. Unset = off; scale is model specific (bge logits can be negative). |
| `LILBEE_FTS_LANGUAGE` | `English` | Stemmer/stop-word language for BM25 indexes | tantivy language name; applied on index build, so change it before a rebuild. |
| `LILBEE_EMBED_TITLES` | `false` | Title-prefixed chunk embeddings | Vector-only change; stored text unchanged. Toggling needs a rebuild (embedding space shifts). |
| `LILBEE_CONTEXTUAL_ENRICHMENT` | `false` | LLM-written situating sentence embedded with each chunk | Contextual retrieval; one generation per chunk, so ingest slows. Toggling needs a rebuild. |
| `LILBEE_TEMPORAL_FILTERING` | `true` | Enable date-based result filtering | Only activates when temporal keywords are detected in the query |
| `LILBEE_SHOW_REASONING` | `false` | Show reasoning model thinking process | For Qwen3/DeepSeek-R1 models that emit `<think>` tags |
| `LILBEE_CONCEPT_GRAPH` | `true` | Enable concept graph (LazyGraphRAG index) | Extracts noun phrases, builds co-occurrence graph, boosts search by concept overlap |
| `LILBEE_CONCEPT_BOOST_WEIGHT` | `0.3` | Concept overlap boost strength (0.0-1.0) | Higher = concept overlap matters more relative to vector similarity |
| `LILBEE_CONCEPT_MAX_PER_CHUNK` | `10` | Max concepts extracted per chunk | Caps extraction to reduce noise from long chunks |
| `LILBEE_CRAWL_MAX_DEPTH` | unset | Optional global ceiling on recursion depth | Set to a positive int to cap calls that pass no explicit depth. Explicit `--depth N` always wins. |
| `LILBEE_CRAWL_MAX_PAGES` | unset | Optional global ceiling on total pages per crawl | Same shape as above. Unset / null = no ceiling. |
| `LILBEE_CRAWL_USER_AGENT` | blank | User agent of every crawl request | Blank = no override: browser mode sends Chrome's own, http mode sends crawlberg's, and the sitemap lookup sends its HTTP client's. |
| `LILBEE_CRAWL_TIMEOUT` | `30` | Per-page fetch timeout in seconds | Applies to the HTTP request and to browser page loads |
| `LILBEE_CRAWL_MAX_CONCURRENT` | CPU count | Max simultaneous crawl operations (top-level) | Limits parallel `crawl_and_save` invocations. Per-crawl concurrency is `crawl_concurrent_requests`. |
| `LILBEE_CRAWL_SYNC_INTERVAL` | `30` | Seconds between periodic syncs during crawl | 0 = sync only after crawl completes. Lower = documents searchable sooner |
| `LILBEE_CRAWL_MEAN_DELAY` | `0.5` | Seconds each concurrent request waits before its next request to the same domain | With `crawl_concurrent_requests` at N, the domain gets about N requests per delay. Tune up for rate-sensitive sites. |
| `LILBEE_CRAWL_MAX_DELAY_RANGE` | `0.5` | Random additional delay on top of mean_delay | Each wait falls between mean_delay and mean_delay + max_delay_range. |
| `LILBEE_CRAWL_CONCURRENT_REQUESTS` | `3` | Concurrent in-flight URLs within one crawl |  |
| `LILBEE_CRAWL_RETRY_ON_RATE_LIMIT` | `true` | Retry pages that answer HTTP 429 or 503 | Other server errors and timeouts are not retried. Set to `false` to disable. |
| `LILBEE_CRAWL_RETRY_BASE_DELAY_MIN` / `MAX` | `1.0` / `3.0` | Base delay range for retries (seconds) | The first retry waits the middle of the range, and each later retry doubles the wait. |
| `LILBEE_CRAWL_RETRY_MAX_BACKOFF` | `30.0` | Upper bound on any single backoff wait (seconds) |  |
| `LILBEE_CRAWL_RETRY_MAX_ATTEMPTS` | `3` | Retries per page | A crawl makes at most 20, whatever the value. |

### Provider Settings

| Setting | Default | Description | Caveats |
|---------|---------|-------------|---------|
| `LILBEE_LLM_PROVIDER` | `auto` | Backend selection: auto, remote | auto = SDK backend for remote refs, the local `llama-server` engine for native GGUF refs |
| `LILBEE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (blank uses the default) | |
| `LILBEE_LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL (blank uses the default) | |
