# lilbee — Development Guide

## Project
Local knowledge base. Python 3.11+, pluggable LLM providers (llama-cpp default, Ollama/OpenAI via litellm), LanceDB for vectors. Managed with `uv`. Task tracking with `beads` (`bd`). Learned behaviors with `floop`.

**Framing:** Lead with "knowledge base" — not "RAG" or "local-first" (those are properties, not the identity). lilbee is both a standalone multipurpose tool AND an AI agent backend.

## Task Tracking (beads)
```bash
bd ready                      # See what's ready to work on
bd update <id> -s in_progress # Claim a task
bd close <id>                 # Mark done
bd list                       # All issues
```
Every code change MUST be tracked as a beads task. Create tasks before starting work, close them when done.

## Commands
```bash
uv sync                       # Install dependencies
make check                    # Run all checks: lint, format, typecheck, test (same as CI)
make test                     # Tests with coverage
make lint                     # Ruff linting
make typecheck                # Mypy
make format                   # Auto-format code
uv run lilbee init            # Create .lilbee/ in current dir (per-project DB)
uv run lilbee sync            # Sync documents to vector DB
uv run lilbee ask "question"
uv run lilbee chat
uv run lilbee status
uv run lilbee rebuild
```

## Architecture
- `src/lilbee/` — All source code
- Per-project DB: `lilbee init` creates `.lilbee/` in cwd (like `.git/`)
- Data location precedence: `--data-dir`/`LILBEE_DATA` > `.lilbee/` (walk up from cwd) > global platform default
  - macOS: `~/Library/Application Support/lilbee/`
  - Linux: `~/.local/share/lilbee/`
  - Windows: `%LOCALAPPDATA%/lilbee/`
- All settings configurable via `LILBEE_*` env vars or CLI flags
- Auto-sync: documents/ is source of truth, data/ is rebuilt from it

## Configuration
All settings override via environment variables:
- `LILBEE_DATA` — data directory path
- `LILBEE_CHAT_MODEL` — LLM model (default: `qwen3:8b`)
- `LILBEE_EMBEDDING_MODEL` — embedding model (default: `nomic-embed-text`)
- `LILBEE_EMBEDDING_DIM` — embedding dimensions (default: `768`)
- `LILBEE_CHUNK_SIZE` — tokens per chunk (default: `512`)
- `LILBEE_CHUNK_OVERLAP` — overlap tokens (default: `100`)
- `LILBEE_TOP_K` — retrieval result count (default: `5`)
- `LILBEE_MAX_DISTANCE` — cosine distance threshold, 0-1 (default: `0.9`). Higher = more results, lower = stricter filtering
- `LILBEE_ADAPTIVE_THRESHOLD` — enable adaptive threshold widening (default: `false`). When true, widens distance threshold if too few results found
- `LILBEE_VISION_MODEL` — vision OCR model (default: none)
- `LILBEE_VISION_TIMEOUT` — per-page vision OCR timeout in seconds (default: `120`, `0` = no limit)
- `LILBEE_TESSERACT_TIMEOUT`: wall-clock timeout in seconds for the Tesseract OCR fallback (default: `60`, `0` = no limit). Only runs when no vision model is available.
- `LILBEE_LLM_PROVIDER` — backend: `auto` (default), `llama-cpp`, `litellm` (requires `pip install lilbee[litellm]`)
- `LILBEE_LITELLM_BASE_URL` — litellm backend endpoint (default: `http://localhost:11434`, also reads `OLLAMA_HOST` for backwards compat)
- `LILBEE_DIVERSITY_MAX_PER_SOURCE` — max chunks per source in results (default: `3`)
- `LILBEE_MMR_LAMBDA` — MMR relevance/diversity tradeoff, 0-1 (default: `0.5`)
- `LILBEE_CANDIDATE_MULTIPLIER` — extra candidates for MMR reranking (default: `3`)
- `LILBEE_QUERY_EXPANSION_COUNT` — LLM-generated query variants, 0=disabled (default: `3`)
- `LILBEE_ADAPTIVE_THRESHOLD_STEP` — distance threshold widening step (default: `0.2`). Only used when adaptive_threshold is true.
- `LILBEE_LOG_LEVEL` — logging level: DEBUG, INFO, WARNING, ERROR (default: `WARNING`)
- `LILBEE_NO_SPLASH` — set to any non-empty value to suppress the startup bee animation

CLI also accepts `--model` / `-m` for chat model, `--data-dir` / `-d`, `--vision-timeout`, and `--log-level`.

## Code Quality Rules

### Test-Driven Development
- **100% test coverage required** — enforced by `pytest-cov` with `fail_under = 100`
- Write tests BEFORE or alongside implementation, not after
- Every public function MUST have at least one test
- **Every test must assert something meaningful** — no zero-assertion coverage padding
- Mock all external dependencies (LLM providers, filesystem I/O where needed) — tests must run without a live server
- Use `pytest.mark.skipif` only for integration tests that genuinely require live services
- Use `tmp_path` fixtures for filesystem tests — never write to real paths
- Test edge cases and error paths, not just the happy path
- Tests are documentation — name them descriptively (`test_add_nonexistent_fails`, not `test_add_3`)
- **Never run tests in background** — user must see output
- **Kill tests that hang past 2-3 minutes** — investigate the hang, don't wait
- **Integration tests must test real multi-system flows**, not mocked unit test duplicates
- Target only affected test files during development; full suite once at the end

### DRY & Modularity
- **Don't Repeat Yourself** — extract shared logic into helpers when duplicated
- Single Responsibility — each function does one thing well
- Small functions — max ~20 lines, max 2 levels of nesting
- Low cyclomatic complexity — extract helpers when branches exceed 3
- **Use maps for classification/dispatch** — if-chains that map values to categories belong in a dict, not repeated `if`/`elif` blocks
- Compose small functions rather than writing monolithic ones
- If you need to copy-paste code, refactor into a shared function instead

### Code Style
- No LangChain — provider abstraction (no raw SDK calls)
- Type hints on all public functions
- Dataclasses for structured return types (not raw dicts)
- **Named constants for magic numbers AND magic strings** — never use raw string literals when a constant exists. Grep globally for the raw value of any new constant to ensure no duplicates remain.
- **No positional array indexes** — use named constants or membership checks, not `array[0]`
- Descriptive variable names — `pending_segments` not `current`, `chunk_size` not `n`
- Logging with `logging.getLogger(__name__)` — no bare `except: pass`
- No hardcoded values — all configurable through `config.py` with env var overrides
- Imports: stdlib first, then third-party, then local — no star imports. See **Import Discipline** below for the rules on function-local imports.
- Linting: `ruff check` + `ruff format` (line length 100)
- Type checking: `mypy` with strict settings
- **No em dashes (—)** — use periods or commas instead
- **No divider comments** (`# ------`) for grouping — use modules or classes instead
- **Annotate `isinstance` guards** with a comment explaining why the check is needed (e.g. untyped frontmatter, test compatibility)
- Use literal unicode chars (`★`) not escapes (`\u2605`); extract to named constants
- **No filterwarnings** without explicit user approval; fix warnings at the source

### Configuration & State
- **No mutable module-level globals** — all config lives in the `Config` dataclass singleton (`from lilbee.config import cfg`)
- Never duplicate state across modules (e.g. no `store_mod.LANCEDB_DIR` mirroring `cfg.lancedb_dir`)
- Prefer dependency injection (pass values as parameters) over reading globals inside functions
- Access config via `cfg.attribute` (late-bound), never `from lilbee.config import SOME_CONSTANT` (early-bound copy)

### Import Discipline

Default: **every import lives at module top**, ordered stdlib, third-party, local. Function-local imports are the exception and must be justified by one of the three cases below. "Feels like it might be heavy" is not a justification.

**Permitted reasons for a function-local import:**

1. **Circular import.** Module A's top-level already imports module B, and module B needs a symbol from A. Put A's import of B inside the one function in B that needs it, with a comment `# circular: B -> A via <symbol>`.
2. **Heavy third-party lib.** Module-top import takes **>50 ms measured by `python -X importtime`** or loads native libraries. Known-heavy libs in this project: `llama_cpp` (native dylibs, Metal/CUDA), `litellm` (provider SDK fanout), `lancedb` (arrow + datafusion), `kreuzberg` (OCR stack), `sentence_transformers`, `spacy`, `crawl4ai`, `textual` when imported outside the TUI screen modules. Use importtime before declaring a lib heavy.
3. **CLI startup path.** The import lives inside a Typer command body so `lilbee --help` stays fast. Treat this as a sub-case of (2): the CLI loader stays lean so unused subcommands don't pay for their dependencies.

**Never lazy-import the following:**

- Stdlib modules (`os`, `struct`, `enum`, `io`, `fnmatch`, …) — zero cost.
- Local lilbee modules that don't pull in heavy third-party deps at their own module top (`lilbee.config`, `lilbee.services`, `lilbee.catalog`, `lilbee.models`, `lilbee.registry`, …).
- Third-party libs already dragged in transitively by the module's top-level imports — re-importing them later is pure noise.
- Project dependencies added explicitly to `pyproject.toml` that measure under 50 ms (`httpx`, `pydantic`, `tiktoken`, `numpy`, `pillow`, `gguf`, …).

**How to measure before arguing "it's heavy":**

```bash
uv run python -X importtime -c "import <lib>" 2>&1 | tail -5
# the last "self" column in microseconds is the module-top cost
```

If the result is under 50 000 µs, it belongs at the module top.

Other rules:

- **Never use `importlib.reload`** — it's a sign of bad design. If you need different config in tests, mutate the `cfg` singleton.
- **`TYPE_CHECKING` guards** are the right way to avoid circular imports *only* for type annotations. For runtime use, apply the circular-import rule above.

### Test Fixtures
- **Snapshot/restore pattern** for config isolation:
  ```python
  from dataclasses import fields, replace
  from lilbee.config import cfg

  @pytest.fixture(autouse=True)
  def isolated_env(tmp_path):
      snapshot = replace(cfg)
      cfg.documents_dir = tmp_path / "documents"
      # ... set test values ...
      yield
      for f in fields(cfg):
          setattr(cfg, f.name, getattr(snapshot, f.name))
  ```
- Never save/restore individual fields manually — snapshot the whole object
- Never touch internal module state (e.g. `store_mod.LANCEDB_DIR`) — only mutate `cfg`

### TUI Rules (Textual)
- **No inline CSS** — screens and modals use `.tcss` files via `CSS_PATH`, not `DEFAULT_CSS` strings. Widgets that don't support `CSS_PATH` may use `DEFAULT_CSS` or load from a file.
- **CSS class/ID sync** — every CSS class or ID in a `.tcss` file must have a matching widget in the `.py` file and vice versa
- **TUI tests use `pilot.press()`** — not `action_*()` method calls. `Button.press()` is acceptable since it's a public widget API, not an action bypass.
- **All TUI design must reference ~/projects/toad** as the canonical implementation for patterns
- **All user-facing text in `messages.py`** — inline strings in screens and widgets are forbidden

### Tests Before Deletion
- When removing code, first make existing tests pass with the new implementation, then delete redundant tests
- Never delete tests and implementation in the same step

### YAGNI & Simplicity
- Don't add features, abstractions, or config that isn't needed yet
- Three similar lines are better than a premature abstraction
- Only validate at system boundaries (user input, external APIs) — trust internal code
- No backwards-compatibility shims — if something is unused, delete it

### Git & Workflow
- Every change tracked as a beads task (`bd create` → `bd close`)
- Run `make check` before closing any task — it mirrors CI exactly
- Tests, lint, and type checks must pass before closing a task
- CI runs on every push and PR
- **Never git push without explicit user approval** — ask before pushing
- No Co-Authored-By lines in commits
- No "Phase N" numbering in commit messages
- Don't commit specs to repo; `docs/superpowers/` is gitignored
- **Pre-commit checklist**: `make format && make lint && make format-check` before every commit
- **Pre-push checklist**: run full review workflow, verify code compiles, run affected tests. Never push optimistically.
- PR descriptions: short, human-readable, no implementation details or internal names, no test plan sections
- Don't rename branches with open PRs; use `gh pr edit` instead
- Use worktrees for parallel work, separate branches for distinct features
- Never stop to ask "should I keep going?" during multi-phase work. Just finish.
- Fix root causes individually, not downstream choke-point bandaids

### QA Protocol
- **QA must run in a dedicated worktree** — branch drift corrupts runs
- QA sessions: **file bugs as beads issues, don't fix inline**
- Use tmux send-keys for automated QA of TUI features
- Always ask before creating issues, PRs, or anything externally visible

### Cross-Cutting Changes
Changes that add an enum variant, a new role/task slot, or anything else whose
correctness depends on updating multiple parallel dispatch / validation sites.

- **Write-boundary canonicalization beats read-path fallbacks.** When the same
  ref, identifier, or config field can be set through multiple entry points
  (HTTP, CLI, TUI, env vars, TOML, direct `cfg.X = ...`), canonicalize it
  ONCE at the write boundary (pydantic `field_validator(mode="after")`, a
  setter, whatever is the single narrow gate) and return the canonical form.
  Alias fallbacks in resolvers are a code smell: they mean canonicalization
  landed in some places but not others. A non-canonical value works until it
  shows up in a log, UI, or diff and confuses the next reader.
- **Front-load the match-arm and entry-point audit.** Before writing the
  first line, grep every dispatch site for the existing variants (`grep -rn
  "ModelTask\.\|ModelRole\.\|SomeEnum\." src/ tests/`). List every
  switch/if-chain/dict. Decide per-site whether it needs the new variant.
  Update every site in the SAME commit that introduces the variant. Leaving
  dispatch sites for reviewers to catch in round N means round N+1 is
  cleaning up regressions from round N's partial fix.
- **Capability / supports_X functions must be role-aware from commit 1.** A
  `get_capabilities(model)` that returns the same default for every input
  (`["completion"]`, `True`) leaks cross-role models into the wrong surfaces.
  Start with role-specific output derived from the catalog entry's task.
  Don't ship the non-role-aware version and wait for a reviewer to flag it.
- **Defaults must round-trip through the tightened validator.** When a
  validator rejects previously-accepted inputs, update the config defaults
  to values that pass under the new rules in the same commit. Otherwise the
  first `Config()` construction raises. Test: `assert Config().X == expected`
  under the new validator.

### Code Review Standards
- **Low complexity** — max ~3 branches per function, extract helpers when exceeded
- **DRY** — reusable shared logic, no copy-paste
- **No private API leaks** — underscore-prefixed functions/attrs stay internal to their module
- **Pythonic idioms** — comprehensions, context managers, dataclasses, protocols over inheritance
- **Named types over inline dicts** — any repeated dict shape should be a dataclass or TypedDict
- **Minimal changes** — make smallest possible edit, don't rewrite large blocks for small fixes
- **Exhaustive review** — multiple review passes until no new findings emerge
- **Compile before test** — verify code compiles before running tests
- **Separate architecture changes from hygiene sweeps.** Mixing a docstring /
  comment cleanup with a semantic refactor makes review hard: reviewers can't
  tell "trimmed docstring" from "removed important rationale." Keep hygiene
  sweeps to their own commit at the end of a series, after architecture is
  frozen.

### Self-Review Checklist (before every push)
Run this mentally or explicitly before claiming work is done:
1. **Trace consumers** — for every changed file, grep for who imports it. Go 2-3 levels deep for shared modules (types, constants, config).
2. **Constants vs raw strings** — grep globally for the raw value of any new constant. Every occurrence must use the constant.
3. **CSS sync** — every class/ID in `.tcss` has a matching widget in `.py` and vice versa.
4. **Mock signatures** — after any refactor, verify test patches target the correct module path (patch where imported, not where defined). When hoisting an import between module-top and function-local, grep `@patch("...{Symbol}")` decorators for that symbol and update them in the same commit. Stale patch targets mean the mock never fires and tests pass without exercising the path — the single worst review-cycle-burner in a refactor.
5. **Dead code** — removed/renamed functions have no remaining references. No orphaned imports.
6. **Test coverage** — every new code path has a test with meaningful assertions. No untested branches.
7. **Type changes** — field added/removed? All constructors, factories, serializers updated?
8. **Export changes** — new export actually needed? Removed export has no dangling imports?

### Behavior Learning (floop)
- `floop` captures corrections and learned behaviors across sessions
- Hooks run automatically via `~/.claude/settings.json` (session-start, dynamic-context, detect-correction)
- `floop active` — show behaviors active in current context
- `floop learn` — manually capture a correction/behavior
- `floop list` — list all learned behaviors
- `floop prompt` — generate prompt section from active behaviors

## Agent Integration

lilbee has a local knowledge base you can query. Use it for domain-specific questions about the user's documents.

### MCP Server (recommended)

An MCP server is configured in `.claude/settings.json` for this project. Tools available:

| Tool | Description |
|------|-------------|
| `search(query, top_k)` | Search for relevant chunks |
| `status()` | Show indexed docs and config |
| `sync()` | Sync documents to vector store |
| `add(paths, force, vision_model)` | Add files/dirs and sync |
| `init(path)` | Initialize a local `.lilbee/` knowledge base |
| `reset(confirm)` | Delete all documents and data (factory reset, requires confirm=true) |

Prefer `search` -- it returns pre-embedded chunks without calling the LLM at query time.

### JSON CLI (fallback)

All commands accept `--json` (before the subcommand) for structured output:

```bash
lilbee --json search "query" --top-k 5
lilbee --json status
lilbee --json sync
```

Every command returns a single JSON object on stdout. Errors return non-zero exit + `{"error": "message"}`.

See [docs/agent-integration.md](docs/agent-integration.md) for full reference.

## Key Files
- `config.py` — All settings (env-var configurable)
- `ingest.py` — Document sync engine (hash-based change detection)
- `query.py` — RAG pipeline (embed → search → generate)
- `store.py` — LanceDB operations
- `chunker.py` — Text chunking (token-based recursive)
- `code_chunker.py` — Code chunking (tree-sitter AST)
- `providers/` — LLM provider abstraction (base protocol, llama-cpp, litellm, factory)
- `catalog.py` — Model discovery from HuggingFace
- `model_manager.py` — Model lifecycle (install, remove, list)
- `embedder.py` — Embedding wrapper (uses provider abstraction)
- `platform.py` — OS helpers, `find_local_root()` for `.lilbee/` discovery
- `cli.py` — Typer CLI with --model, --data-dir, --version, and --json flags
- `mcp.py` — MCP server exposing search, ask, status, sync, init as tools

## Wiki Conventions

lilbee's wiki layer is modelled after Karpathy's LLM Wiki: concept and
entity pages that compound across sources rather than one page per
document. The layout under `$data_root/$wiki_dir/` is:

```
concepts/    one page per noun-phrase topic (e.g. braking-systems.md)
entities/    one page per proper-noun entity (e.g. henry-ford.md)
summaries/   legacy per-source pages (still supported, not the default)
synthesis/   cross-source pages produced by `wiki synthesize`
drafts/      low-faithfulness drafts that missed the threshold
archive/     pages retired by `wiki prune`
index.md     auto-generated table of contents, grouped by page type
log.md       append-only audit trail (## [YYYY-MM-DD HH:MM] op | details)
```

**Slugs** are lowercase, hyphen-separated filenames that also double as
the `[[link]]` target (`braking-systems`, not `Braking Systems`). The
slug generator lives at `wiki/shared.py:make_slug`.

**Page lifecycle**. `lilbee wiki build` extracts entities from the
chunk store using `cfg.wiki_entity_mode` (default `ner_concepts`:
spaCy NER + noun phrases), generates a page per record via
`wiki/gen.py:build_wiki`, refreshes `index.md`, and appends a `build`
entry to `log.md`. `lilbee sync` runs `_incremental_wiki_update` after
a successful sync: pages whose chunk trail cites a changed source, or
pages that don't exist yet, are regenerated. The cap is
`cfg.wiki_ingest_update_cap` (default 20); above that the hook logs a
manual-update hint and exits.

**Retrieval inside wiki/gen.py**. Each page is grounded in the top
`cfg.wiki_concept_max_chunks_per_page` chunks returned by the store's
hybrid search, optionally reordered by the native GGUF reranker when
`cfg.reranker_model` is set. Every path respects
`cfg.diversity_max_per_source` so one loud document can't monopolize a
topic page.

**`[[wiki links]]`**. After each build, `wiki/links.py:rewrite_wiki_links`
rewrites plain-text slug surface forms to Obsidian `[[slug]]` links
in the page body, skipping YAML frontmatter, code fences, and the
auto-generated citation block. The graph view in Obsidian is the point;
orphan detection in `lilbee wiki lint` flags concept or entity pages
with zero inbound `[[links]]`.

**Adding a new entity-extraction strategy**. Implement
`EntityExtractor` from `wiki/entity_extractor/base.py`, add a new
`WikiEntityMode` enum value in `config.py`, wire it into
`wiki/entity_extractor/factory.py:_EXTRACTOR_BY_MODE`, and extend the
`wiki_entity_mode` entry in `cli/settings_map.py` (its `choices` field
reads from the enum, so no update needed there).

**Adding a new config knob**. All wiki settings live under the `Wiki`
group in `config.py` and `cli/settings_map.py`. Prefix with `wiki_`
(for module-level behaviour) or `wiki_concept_` / `wiki_entity_`
(for strategy-scoped knobs). Use `ConfigField(writable=True)` so the
setting appears in `/settings`, the HTTP `/set` route, and
`LILBEE_*` env vars.

**Writing to log.md**. Use `append_wiki_log(action, details)` from
`wiki/index.py` with one of the `WIKI_LOG_ACTION_*` constants in
`wiki/shared.py` (`BUILD`, `INGEST`, `LINT`, `GENERATED`).
Don't hand-roll timestamps; the helper writes
`## [YYYY-MM-DD HH:MM] action | details` so `grep '## \['` still
surfaces every entry.
