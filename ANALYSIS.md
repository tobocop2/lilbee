# Project Nomad vs lilbee: Engine Analysis

## Simple Comparison (No Programming Knowledge Required)

| What You Want To Do | Nomad | lilbee |
|---|---|---|
| **Search your Word documents (.docx)** | Broken — files are read as corrupted binary, search results will be garbage | Works — extracts text, tables, and formatting correctly |
| **Search your spreadsheets (.xlsx)** | Not supported | Works |
| **Search your PowerPoint files (.pptx)** | Not supported | Works |
| **Search your ebooks (.epub)** | Not supported | Works |
| **Search your PDFs** | Works | Works |
| **Search your images (OCR)** | Works — outputs plain text, loses tables and layout | Works — outputs structured markdown preserving tables and headings |
| **Search plain text and markdown** | Works | Works |
| **Install the software** | Requires Docker, MySQL, Redis, Qdrant, and 6+ running containers. Minimum ~4GB RAM before AI models. | `pip install lilbee`. No Docker, no database servers. |
| **Keep different topics separate** | Not possible — all documents go into one shared search pool | Built-in — each project gets its own isolated search database |
| **Change settings (search sensitivity, AI model, chunk size)** | Requires editing source code and rebuilding the Docker image | Change via environment variables, config file, or command-line flags |
| **Know when something breaks** | No automated tests exist. Broken features (like DOCX) ship without detection. | 100% automated test coverage. Broken code is caught before release. |
| **Use with AI assistants (Claude Code, etc.)** | Not possible — no agent integration | Built-in MCP server lets AI assistants search your documents directly |
| **Browse offline Wikipedia, maps, Khan Academy** | Yes — pre-packaged content bundles available | Not a feature — you bring your own files |
| **Use a visual interface** | Yes — full web dashboard with chat, file upload, monitoring | [obsidian-lilbee](https://github.com/tobocop2/obsidian-lilbee) — Obsidian plugin providing GUI for search, chat, and document management inside an existing knowledge management tool. Also CLI + REST API. |
| **Run on a low-power device (Raspberry Pi, old laptop)** | Unlikely — Docker + 6 containers + MySQL + Redis + Qdrant is resource-heavy | Yes — single lightweight process with embedded database |

---

## Advanced Technical Comparison

### Architecture

| Aspect | Nomad | lilbee |
|---|---|---|
| **Language** | TypeScript (non-strict) | Python 3.11+ (mypy strict, `disallow_untyped_defs`) |
| **Vector store** | Qdrant (external server, requires Docker) | LanceDB (embedded, no server, by design) |
| **Infrastructure** | Docker + MySQL + Redis + Qdrant + 6 containers | Single process + Ollama |
| **Largest source file** | `rag_service.ts` — 1,212 lines, 20+ responsibilities | `ingest.py` — 669 lines, single pipeline |
| **Module count** | 16 service files | 27 focused modules (each single-responsibility) |
| **Direct dependencies** | 62 (includes native binaries: sharp, tesseract.js, better-sqlite3) | 14 (pure Python + Ollama) |
| **Dev dependencies** | 22 | 8 |

### Text Extraction

| Format | Nomad | lilbee |
|---|---|---|
| **PDF (text-based)** | pdf-parse library | Kreuzberg library |
| **PDF (scanned/image)** | tesseract.js with basic preprocessing (grayscale, sharpen). Plain text output. | Kreuzberg → Tesseract fallback → Vision model OCR fallback. Markdown output preserving tables/headings. |
| **DOCX** | Classified as `'text'` type in `fs.ts:155-169`. Processed by `extractTXTText()` which calls `filebuffer.toString('utf-8')`. DOCX is a ZIP archive — this reads binary ZIP data as UTF-8, producing corrupted output that gets embedded into the vector store. | Kreuzberg — proper ZIP/XML extraction |
| **XLSX** | Not supported (returns `'unknown'` file type) | Kreuzberg extraction |
| **PPTX** | Not supported | Kreuzberg extraction |
| **EPUB** | Not supported | Kreuzberg extraction |
| **CSV/TSV** | Not supported | Custom preprocessor — converts to "Header: Value" prose for embedding |
| **JSON/JSONL** | Not supported | Custom preprocessor — flattened "dotted.path: value" format |
| **XML** | Not supported | Custom preprocessor — tree walker with indentation |
| **YAML** | Not supported | Kreuzberg as text |
| **HTML/RST** | Not supported | Kreuzberg extraction |
| **Code (150+ languages)** | Not supported — only `.txt` and `.md` are "text" type | tree-sitter AST-aware chunking. Extracts functions, classes, methods as semantic units. 28 languages have definition-level extraction metadata. |
| **Images** | tesseract.js OCR | Kreuzberg → Tesseract → Vision model fallback chain |
| **ZIM** | ZIMExtractionService (separate module) | Not supported |

### Chunking

| Aspect | Nomad | lilbee |
|---|---|---|
| **Method** | `@chonkiejs/core` TokenChunker | Custom recursive splitter (`chunker.py`) |
| **Token estimation** | `Math.ceil(text.length / 3)` — fixed 3 chars = 1 token | tiktoken `cl100k_base` — actual tokenizer |
| **Chunk size** | 5,100 chars (~1,700 estimated tokens), hardcoded | 512 tokens default, configurable via `LILBEE_CHUNK_SIZE` |
| **Overlap** | 450 chars (~150 estimated tokens), hardcoded | 100 tokens default, configurable via `LILBEE_CHUNK_OVERLAP` |
| **Boundary awareness** | None — pure character count splitting | Recursive: paragraph (`\n\n`) → sentence (`. `) → word (` `) boundaries |
| **Code awareness** | None — code treated as plain text | tree-sitter AST parsing. Functions/classes kept as semantic units. Line numbers preserved for citations. |

### Search & Retrieval

| Aspect | Nomad | lilbee |
|---|---|---|
| **Search type** | Hybrid — semantic + keyword matching | Pure vector cosine similarity |
| **Reranking** | Yes — semantic score + keyword boost (10%) + direct match boost (7.5%) | No reranking |
| **Source diversity** | Yes — 0.85 penalty for same-source results | Deduplication by source file |
| **Query preprocessing** | Abbreviation expansion (26 domain terms), keyword extraction | Direct embedding of query |
| **Similarity threshold** | 0.3 (hardcoded in `rag_service.ts:673`) | 0.7 default (configurable via `LILBEE_MAX_DISTANCE`) |
| **Embedding model** | `nomic-embed-text:v1.5` (hardcoded) | `nomic-embed-text` default (configurable via `LILBEE_EMBEDDING_MODEL`) |
| **Embedding dimensions** | 768 (hardcoded) | 768 default (configurable via `LILBEE_EMBEDDING_DIM`) |

### Configuration

| Setting | Nomad | lilbee |
|---|---|---|
| **Embedding model** | Hardcoded: `'nomic-embed-text:v1.5'` | `LILBEE_EMBEDDING_MODEL` env var / config.toml / `--model` flag |
| **Chunk size** | Hardcoded: `1700` tokens | `LILBEE_CHUNK_SIZE` env var / config.toml |
| **Chunk overlap** | Hardcoded: `150` tokens | `LILBEE_CHUNK_OVERLAP` env var / config.toml |
| **Similarity threshold** | Hardcoded: `0.3` | `LILBEE_MAX_DISTANCE` env var / config.toml |
| **Batch size** | Hardcoded: `8` | Adaptive based on content length (`MAX_BATCH_CHARS=6000`) |
| **Result count** | Hardcoded in search method | `LILBEE_TOP_K` env var / `--top-k` flag |
| **System prompt** | Not configurable | `LILBEE_SYSTEM_PROMPT` env var / config.toml |
| **Temperature** | Not configurable | `LILBEE_TEMPERATURE` env var / `--temperature` flag |
| **Vision model** | Not applicable (no vision OCR) | `LILBEE_VISION_MODEL` env var / `--vision` flag |
| **Vision timeout** | Not applicable | `LILBEE_VISION_TIMEOUT` env var (default 120s) |
| **Server host/port** | Not configurable for RAG | `LILBEE_SERVER_HOST` / `LILBEE_SERVER_PORT` |
| **Ignore directories** | Not configurable | `LILBEE_IGNORE` env var (CSV list) |
| **Context window** | Not configurable | `LILBEE_NUM_CTX` env var |

### Knowledge Base Isolation

| Aspect | Nomad | lilbee |
|---|---|---|
| **Collection model** | Single global: `'nomad_knowledge_base'` (hardcoded, line 27 of `rag_service.ts`) | Per-project `.lilbee/` directory containing its own LanceDB store |
| **Cross-contamination** | All documents from all contexts share one vector space | Each project is fully isolated — separate embeddings, separate search |
| **Init mechanism** | Automatic on first use | `lilbee init` creates `.lilbee/` in current directory |
| **Discovery** | N/A (single global) | Walks up directory tree looking for `.lilbee/` (like git finds `.git/`) |
| **Fallback** | N/A | Platform-specific default: `~/Library/Application Support/lilbee` (macOS), `~/.local/share/lilbee` (Linux), `%LOCALAPPDATA%/lilbee` (Windows) |

### Error Handling

| Aspect | Nomad | lilbee |
|---|---|---|
| **`embedAndStoreText` failure** | Returns `null` — caller cannot distinguish failure from no-op | Typed exceptions with context messages propagated to caller |
| **`processAndEmbedFile` failure** | Returns `{ success: false, message: 'Error processing and embedding file.' }` — generic | Returns `_IngestResult` with error field containing the original exception |
| **Search failure** | Returns `[]` — indistinguishable from "no results" | Exceptions propagate with context |
| **OCR failure** | Single attempt, logs error | Three-stage fallback: Kreuzberg → Tesseract → Vision model, each with logging |
| **Console.log in production** | 5 instances in service files (should use logger) | Zero — all output through Rich console or `logging.getLogger()` |
| **Catch-all exception count** | Multiple `catch (error)` blocks that log and return null/empty | 6 `except Exception` blocks, all with documented fallback behavior |

---

## Slop & Vibe Coding Analysis

Signs of AI-generated code shipped without review, or features built by prompting without understanding the output.

| Indicator | Nomad | lilbee |
|---|---|---|
| **Broken features shipped as working** | DOCX listed as supported file type but reads binary ZIP as UTF-8 text. Produces corrupted embeddings with no error raised. | No broken features identified. |
| **Test coverage** | 0%. Zero test files. CI runs no tests. | 100% enforced (`fail_under=100`). 25 test files. CI runs lint + typecheck + tests across 3 OS × 3 Python versions. |
| **Commented-out code** | 51-line AMD GPU discovery block in `docker_service.ts:743-793`. 9-line container check block at `docker_service.ts:221-229`. | None. |
| **TODO/FIXME comments** | 2 TODOs (both thoughtful with context). | Zero. |
| **Console.log in production** | 5 instances across `collection_manifest_service.ts`, `rag_service.ts`, `map_service.ts`. | Zero. |
| **Type safety escapes (`as any` / `type: ignore`)** | 3 `as any` casts. | 8 `# type: ignore` comments, each with justification. |
| **God objects** | `RagService`: 1,212 lines, 20+ distinct responsibilities in one class. | None. 27 modules, each under 670 lines with a single focus. |
| **Hardcoded magic numbers** | 18+ in `rag_service.ts` with no configuration override. | All defaults extracted to config model with env var / file / CLI overrides. |
| **Dead code** | AMD GPU detection runs and broadcasts "AMD GPU detected" message but the discovered devices are never used. `_discoverAMDDevices()` is commented out. | None. |
| **Error swallowing** | 3 critical methods (`embedAndStoreText`, `processAndEmbedFile`, `searchSimilarDocuments`) catch errors and return null/empty — callers get silent failure. | All exceptions include context and are either propagated or handled with documented fallback chains. |
| **Copy-paste duplication** | Minimal. | Minimal (one justified UI duplication in model picker display). |
| **Naming consistency** | Consistent throughout (camelCase/PascalCase/UPPER_SNAKE_CASE). | Consistent throughout (PEP 8). |
| **Dependency bloat** | 62 direct production dependencies including native binaries requiring compilation. | 14 direct dependencies. No native compilation. |
| **Unused imports** | None found. | None found. |
| **Over-engineering** | `IMapService` interface defined but used by only one class with one property. Dynamic service instantiation as circular dependency workaround. | None. No unnecessary abstractions. |
| **Docstring/comment quality** | Substantive — explains "why" not just "what." | Substantive — explains intent and behavior. |
| **README accuracy** | Accurate — described features are implemented (except DOCX works as claimed). | Accurate — no feature claims without working implementation. |
| **Half-implemented features** | AMD GPU support: detection runs, user sees "AMD GPU detected" broadcast, but acceleration is not functional. | None. |
| **Generated boilerplate feel** | No — shows domain knowledge and intentional design choices. | No — shows deliberate engineering decisions. |

---

## What Nomad Has That lilbee Does Not

| Feature | Details |
|---|---|
| **Standalone web UI** | Full React dashboard with chat interface, file upload, service management, system monitoring, and setup wizard. |
| **Pre-packaged offline content** | Kiwix ZIM catalog (Wikipedia in 100+ languages, WikiHow, StackOverflow, Project Gutenberg), Khan Academy via Kolibri, offline maps via ProtoMaps. |
| **Hybrid search** | Semantic search + keyword matching + reranking with source diversity scoring (0.85 penalty for same-source, 10% keyword boost, 7.5% direct match boost). |
| **Container orchestration** | Docker lifecycle management with health checks, dependency resolution, auto-updates with rollback, NVIDIA GPU passthrough. |
| **Chat session persistence** | MySQL-backed CRUD — create, list, delete sessions with AI-generated titles. |
| **System monitoring** | Disk usage tracking (sidecar collector), service health checks, benchmark suite. |
| **Offline maps** | ProtoMaps integration with downloadable regional tile sets. |

## What lilbee Has That Nomad Does Not

| Feature | Details |
|---|---|
| **Working DOCX/XLSX/PPTX/EPUB extraction** | Kreuzberg library handles Office formats, ebooks, and structured data correctly. |
| **Code understanding** | tree-sitter AST parsing for 150+ languages. Functions and classes extracted as semantic units with line numbers for citations. |
| **MCP server** | 6 tools (`lilbee_search`, `lilbee_status`, `lilbee_sync`, `lilbee_add`, `lilbee_init`, `lilbee_reset`) for AI assistant integration. |
| **Per-project isolation** | `.lilbee/` directories with independent vector stores. No cross-contamination between projects. |
| **Configuration system** | 20+ settings configurable via environment variables (`LILBEE_*` prefix), `config.toml` persistence, and CLI flags. |
| **Boundary-aware chunking** | Recursive splitting on paragraph → sentence → word boundaries. AST-level splitting for code. |
| **Vision model OCR** | Ollama vision models (e.g., LightOnOCR-2) that output structured markdown preserving tables and headings. Three-stage fallback chain. |
| **Test suite** | 100% coverage enforced. 25 test files (~18,000 lines of tests for ~5,100 lines of source). CI matrix: 3 OS × 3 Python versions. |
| **Strict typing** | mypy with `disallow_untyped_defs=true`. Pydantic models for config validation and request/response schemas. |
| **Embedded database (by design)** | LanceDB requires no external server. The architecture intentionally avoids Docker, database servers, and container orchestration. Single `pip install` is the entire stack. |
| **Obsidian GUI** | [obsidian-lilbee](https://github.com/tobocop2/obsidian-lilbee) plugin — search, chat, and document management inside Obsidian. Integrates with an existing knowledge management tool rather than requiring a separate web app. |
| **Structured data preprocessing** | CSV → "Header: Value" prose. JSON → "dotted.path: value" flattening. XML → indented tree walking. All converted to embeddable text. |
