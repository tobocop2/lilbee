# Competitive Landscape

Competitors to lilbee, narrowed to projects that are closest in spirit (local, CLI/developer-oriented) or have the strongest RAG retrieval pipelines.

Last updated: 2026-03-26 (reflects lilbee 0.6.0 release features)

## Feature Comparison

| Feature | lilbee | qmd | llm-search | aichat | txtai | kotaemon | LightRAG | RAGFlow |
|---|---|---|---|---|---|---|---|---|
| **Language** | Python | TypeScript | Python | Rust | Python | Python | Python | Python |
| **Stars** | -- | 17k | 649 | 9.6k | 12.3k | 25.2k | 30.6k | 76.3k |
| **Interface** | CLI + TUI + MCP + REST API | CLI + MCP | MCP + Web | CLI | API + MCP | Web UI | API + Web | API + Web |
| **Vector search** | LanceDB | GGUF embeddings | ChromaDB | Any provider | HuggingFace | ChromaDB/LanceDB/Milvus/Qdrant | Custom | Elasticsearch/Infinity |
| **BM25/keyword** | LanceDB FTS (tantivy) | SQLite FTS5 | SPLADE sparse | Built-in FTS | Sparse vectors | Elasticsearch/LanceDB FTS | No | Elasticsearch |
| **Hybrid search** | FTS + vector + RRF | BM25 + vector + RRF | Dense + SPLADE | Vector + keyword + RRF | Dense + sparse | FTS + vector | KG + vector (mix mode) | FTS + vector |
| **Reranking** | Optional cross-encoder (`pip install lilbee[reranker]`) | Dedicated GGUF reranker, position-aware blending | 3 cross-encoder options (ms-marco, bge, zerank) | Optional via provider | No | Cohere, TEI, VoyageAI | bge-reranker-v2-m3, Jina | Fused re-ranking |
| **Query expansion** | LLM-generated (3 variants) with drift guardrails | Fine-tuned 1.7B GGUF model (2 variants) | Multi-query / RAG Fusion (3 variants) | No | No | No | No | No |
| **HyDE** | Yes (weighted, off by default) | Yes | Yes | No | No | No | No | No |
| **MMR diversity** | Yes (lambda 0.5) | No | No | No | No | No | No | No |
| **Adaptive thresholds** | Yes (step 0.2) | No | No | No | No | No | No | No |
| **Per-source diversity cap** | Yes (max 3/source) | No | No | No | No | No | No | No |
| **Confidence skip** | Yes (skips LLM when keyword match is strong) | No | No | No | No | No | No | No |
| **Smart context selection** | Yes (maximizes query term coverage) | No | No | No | No | No | No | No |
| **Date-aware filtering** | Yes ("recent changes", "last week") | No | No | No | No | No | No | No |
| **Direct search modes** | Yes (`term:`, `vec:`, `hyde:` prefixes) | Yes (`hyde:` sub-query) | No | No | No | No | 5 query modes (local/global/hybrid/naive/mix) | No |
| **Graph RAG** | No | No | No | No | Yes (semantic graphs) | Yes (MS GraphRAG, NanoGraphRAG, LightRAG) | Core feature (entity-relationship extraction) | No |
| **Code-aware chunking** | Tree-sitter AST (170+ languages, 55 extensions) | No | No | No | No | No | No | No |
| **Heading-aware chunking** | Yes (prepends heading path, splits at section boundaries) | No | No | No | No | No | No | Yes (layout-aware) |
| **Document parsing** | kreuzberg (PDF, DOCX, XML, CSV, JSON, images + vision OCR, 150+ formats) | Markdown only | MuPDF (PDF), DOCX, Markdown | External loaders (pdftotext, pandoc) | Multimodal (text, audio, image, video) | PDF with page-level citations, HTML, XLSX | Text only (plugin for PDF/Office) | DeepDoc (OCR, layout, tables, figures) |
| **Per-project isolation** | `.lilbee/` directory (like `.git/`) | Collections in global SQLite | No | Named RAGs in shared dir | No | No | No | No |
| **Fully offline** | Yes (llama-cpp, auto-downloads from HuggingFace) | Yes (GGUF auto-download) | Yes (llama.cpp, Ollama) | Only with Ollama | Yes (HuggingFace, llama.cpp) | Yes (Ollama, llama-cpp) | Partially (needs 32B+ model) | No (Docker + MySQL + Redis + ES) |
| **LLM providers** | 3 (llama-cpp, Ollama, OpenAI/litellm) | 1 (node-llama-cpp GGUF) | Local + LiteLLM + Ollama | 20+ (OpenAI, Claude, Gemini, Ollama, etc.) | HuggingFace, llama.cpp, sentence-transformers | Ollama, llama-cpp, OpenAI, Azure, Cohere | Ollama, OpenAI, others | Various via config |
| **MCP server** | Yes (stdio) | Yes (stdio + HTTP daemon) | Yes (SSE) | No (MCP client only) | Yes (FastAPI) | No | No | Yes |
| **Reasoning model support** | Yes (Qwen3/DeepSeek-R1 thinking tags) | No | No | No | No | No | No | No |
| **Test coverage** | 100% (1279+ tests) | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |

## Tier 1 -- Direct Competitors

### qmd
https://github.com/tobi/qmd

The single closest competitor. TypeScript CLI that indexes local documents into a SQLite database with BM25 full-text search + GGUF vector embeddings, fused via reciprocal rank fusion. Ships a dedicated fine-tuned 1.7B query expansion model and a GGUF reranker with position-aware score blending (top results weighted 75/25 retrieval/reranker, decaying to 40/60). Supports HyDE via `hyde:` sub-queries. MCP server runs as an HTTP daemon that keeps models warm in VRAM. Has an `intent` parameter that steers expansion, search, and ranking behavior. Key limitation: Markdown-only ingestion (no PDF, DOCX, code), no per-project isolation (global SQLite with logical collections), no AST-aware chunking, locked to GGUF models only, no MMR diversity or per-source caps.

### llm-search (pyLLMSearch)
https://github.com/snexus/llm-search

Closest in RAG sophistication. Hybrid retrieval combining dense embeddings (ChromaDB) with SPLADE sparse vectors, plus three cross-encoder reranking options (ms-marco-MiniLM, bge-reranker-v2-m3, zerank-2). Has HyDE and multi-query/RAG Fusion with 3 generated variants. Format-aware chunking for Markdown (heading-based), PDF (MuPDF), and DOCX (nested tables). Exposes an MCP server via SSE and a FastAPI web frontend. Key limitation: no MMR diversity, no adaptive thresholds, no AST chunking, no per-project DBs, no confidence skip or date-aware filtering, YAML-heavy configuration.

### aichat
https://github.com/sigoden/aichat

Rust-based all-in-one LLM CLI with the broadest provider support (20+). Built-in RAG with hybrid vector + keyword search and RRF fusion. Can crawl URLs and websites as RAG sources (via Jina). Doubles as a shell assistant and agent framework. Key limitation: no query expansion, no HyDE, no MMR, no code-aware chunking, no per-project isolation, no MCP server (only MCP client), no heading-aware chunking. RAG is functional but not its primary focus.

## Tier 2 -- Best RAG Engines

### txtai
https://github.com/neuml/txtai

The most capable framework overall. Unified embeddings database with hybrid dense + sparse vectors, semantic graphs (GraphRAG), and SQL queries over embeddings. Truly multimodal -- embeds text, audio (Whisper), images (CLIP/BLIP), and video into the same vector space. Composable pipelines for QA, summarization, translation, transcription. MCP server and SDKs in JS/Java/Rust/Go. Key limitation: it is a toolkit, not an opinionated tool -- requires assembly. No HyDE, no query expansion, no MMR, no adaptive thresholds, no AST chunking, no per-project DBs.

### kotaemon
https://github.com/Cinnamon/kotaemon

Hybrid full-text + vector search with the deepest GraphRAG support: integrates MS GraphRAG, NanoGraphRAG, and LightRAG simultaneously. Reranking via Cohere, TEI, or VoyageAI. Agent reasoning with question decomposition (ReAct/ReWOO). PDF viewer with page-level citation highlighting. Key limitation: Gradio web UI only (no CLI, no MCP), Docker-oriented, no query expansion, no HyDE, no MMR, no date-aware filtering.

### LightRAG
https://github.com/HKUDS/LightRAG

Graph-first RAG that builds entity-relationship knowledge graphs via LLM extraction. Five query modes: local (entity lookup), global (relationship traversal), hybrid, naive (vector only), and mix (KG + vector). Reranking with bge-reranker-v2-m3 or Jina. Entirely different paradigm from vector-first tools -- excels at relationship and entity queries. Key limitation: requires 32B+ parameter LLMs for quality entity extraction, no CLI, no MCP, no per-project DBs, text-only ingestion without plugins, token-based chunking only.

### RAGFlow
https://github.com/infiniflow/ragflow

Best-in-class document understanding via its DeepDoc engine: custom OCR, layout recognition (10 component types including headers, footers, captions), table structure recognition with auto-rotation, and figure extraction. Template-based chunking that respects document layout. Hybrid full-text + vector search with fused re-ranking. MCP support. Key limitation: server-grade platform requiring Docker with MySQL, MinIO, Redis, and Elasticsearch (min 4 cores, 16GB RAM, 50GB disk). No CLI, no per-project DBs, no query expansion, no HyDE, no MMR.

## lilbee's Unique Position

As of 0.6.0, lilbee has the most complete retrieval pipeline of any CLI-first local knowledge base tool. Features that no single competitor matches:

- **Tree-sitter AST code chunking** (170+ languages, 55 extensions) -- 0/7 competitors
- **Heading-aware chunking** with heading path prepending (Anthropic's contextual retrieval pattern) -- only RAGFlow has layout-aware chunking
- **Hybrid search + RRF + MMR diversity** in one pipeline -- no competitor combines all three
- **Adaptive distance thresholds** for recall tuning -- 0/7
- **Per-source diversity caps** (max chunks per document) -- 0/7
- **Confidence skip** (bypasses LLM when keyword match is strong) -- 0/7
- **Smart context selection** (maximizes query term coverage, not just top-k) -- 0/7
- **Date-aware filtering** (natural language temporal queries) -- 0/7
- **Direct search modes** (`term:`, `vec:`, `hyde:` prefixes) -- only qmd has `hyde:`
- **Per-project `.lilbee/` isolation** (physical, walk-up discovery like `.git/`) -- 0/7
- **CLI + TUI + MCP + REST API** in a single lightweight tool -- no competitor offers all four interfaces
- **150+ document formats** via kreuzberg with vision OCR -- only RAGFlow and txtai approach this breadth
- **Reasoning model support** (Qwen3/DeepSeek-R1 thinking tags) -- 0/7
- **No Ollama dependency** -- works standalone with llama-cpp, auto-downloads from HuggingFace

## Remaining Gaps

| Gap | Competitors with it | Worth pursuing? |
|---|---|---|
| **Graph RAG (full)** | txtai, kotaemon (3 impls), LightRAG | No -- 8B models can't reliably extract entities, indexing takes hours, doubles codebase |
| **SPLADE sparse retrieval** | llm-search | No -- only +1-2 NDCG points over BM25 in hybrid pipelines, adds neural model overhead |
| **URL/website crawling** | aichat | Yes -- see roadmap |
| **Warm model daemon** | qmd (HTTP MCP daemon keeps models in VRAM) | Yes -- see roadmap |

**Not a gap (already implemented):** Intent-steered search -- lilbee's `term:`, `vec:`, `hyde:` prefixes + confidence-based expansion skipping + temporal keyword detection cover this.

## Roadmap

Easy wins and pragmatic features informed by the competitive landscape.

### Warm model pre-loading -- trivial (~30 min)

The provider singleton (`providers/factory.py`) lazy-loads on first request. The REST API (Litestar) already keeps models in memory between requests. Only missing: call `get_provider()` during server/MCP startup so the first query isn't slow. For Ollama, models already stay warm in the Ollama daemon.

### URL/website crawling -- moderate (~2-3 hours)

`lilbee add <url>` for single pages, `lilbee add --crawl <root-url> --depth N` for recursive crawling via crawl4ai. Optional dependency: `pip install lilbee[crawler]`. httpx is already available for simple fetches; kreuzberg handles HTML extraction. Follows the existing optional-dep pattern (`lilbee[reranker]`).

### Concept graph (LazyGraphRAG index side) -- moderate (~5 days)

Build a concept co-occurrence graph at index time using NLP extraction (spaCy or regex), with Leiden community detection for topic clustering. Zero LLM calls at index or query time. Optional dependency: `pip install lilbee[graph]` (spaCy, networkx, graspologic-native).

Three lightweight wins from the graph with no query-time LLM overhead:

1. **Concept-boosted search** -- boost chunks from the same concept community when query matches graph nodes, as an additional signal fed into existing RRF fusion
2. **Graph-aware query expansion** -- use concept co-occurrences to expand queries with related terms, supplementing or replacing LLM-generated expansion (saves an LLM call)
3. **`lilbee topics`** -- show top concept communities as a knowledge base overview, giving users a map of what's indexed

This gets ~60% of LazyGraphRAG's value at ~30% of the cost. The full LazyGraphRAG query pipeline (10-30 LLM calls per query, no open-source reference implementation, marginal 8B model support) is deferred until justified by user demand.
