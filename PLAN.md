# Plan: TUI bugs round 2 — Ctrl+C, stdout, task queues, navigation

## Session status (last updated: 2026-04-01)

**Closed**: lilbee-ttx (fd error), lilbee-k46 (download crash), lilbee-cxl (token mask)
**In progress**: lilbee-6a0 + lilbee-o97 + lilbee-qc3 (vision handler + stdout -- agent may have stalled, check `.claude/worktrees/fix-vision`)
**Pushed**: `feature/tui-improvements` has fd fix + download crash fix + token masking
**Next P0**: lilbee-blq (Ctrl+C hang), then finish vision issues
**Remaining**: 20 open issues. Run `bd ready` for full list.

## Cold-start instructions for any agent

1. **Start dolt**: `cd /tmp/lilbee-dolt && dolt sql-server --port 3307 &` (beads needs this)
2. **See all work**: `bd ready` lists issues by priority. P0 first.
3. **Claim work**: `bd update <id> -s in_progress`
4. **Use worktrees**: `git worktree add .claude/worktrees/<name> feature/tui-improvements` — never work directly on the branch
5. **Branch**: `feature/tui-improvements` (PR #43 targeting `feature/tui-fixes-and-models`)
6. **Rules**: Read CLAUDE.md. No emojis. No Co-Authored-By. 100% test coverage. Small functions. `make check` before commit.
7. **When done**: `bd close <id>`, commit in worktree, push, clean up worktree
8. **Phase 1 (P0) items are independent** — can be parallelized across worktrees
9. **Phase 2 (P1) has dependencies**: navigation fixes (2a) before mode indicator (2b), task queues (2e) before progress migration (2d)

## New issues from testing

| # | Issue | Priority |
|---|-------|----------|
| 1 | Ctrl+C doesn't quit during download/vision — TUI hangs | P0 |
| 2 | Vision model stdout still corrupts TUI (subprocess not active?) | P0 |
| 3 | Download fails with NoActiveAppError on complete | P0 |
| 4 | Tasks block each other — downloads, adds, crawls need separate queues | P1 |
| 5 | hjkl doesn't navigate in normal mode | P1 |
| 6 | No visual indicator of normal mode vs chat mode | P1 |
| 7 | NavBar should move to top (vim-airline style) | P2 |
| 8 | Mirror task queue changes in Obsidian plugin | P1 |

| 9 | Model dropdowns show wrong model types (chat models in embed dropdown, etc.) | P1 |
| 10 | Model dropdowns have no label (chat/embed/vision) | P1 |
| 11 | Model dropdowns can't be reached via keyboard in normal mode | P1 |
| 12 | "bad value(s) in fds_to_keep" error when chatting (subprocess fork issue) | P0 |
| 13 | Catalog needs family grouping, sort/filter controls not visible | P2 |
| 14 | Up arrow in normal mode navigates views instead of UI elements on chat screen | P1 |

## Analysis

**#12 is critical** — "bad value(s) in fds_to_keep" is a Python multiprocessing error. The subprocess worker likely has file descriptor issues on spawn. This breaks chat entirely.

**#1 + #2 + #3 + #12 suggest the subprocess worker isn't working correctly.** Vision stdout still leaks, Ctrl+C hangs during native calls, downloads crash with NoActiveAppError, and chat errors with fd issues. These may all stem from the subprocess implementation not being properly integrated.

**#5 + #11 + #14 are the same root cause** — normal mode navigation doesn't work for moving between UI elements within a screen (model dropdowns, chat log). hjkl/arrows only cycle between views, not within a view.

| 15 | Download/crawl progress bars still show in chat view — move to task center | P1 |
| 16 | Tasks view should highlight/flash when there are active tasks | P2 |
| 17 | Vision models broken with llama-cpp (work with Ollama) | P0 |
| 18 | Settings screen doesn't show GGUF model defaults (temp, etc.) | P1 |
| 19 | All views should be clickable (mouse support for non-vim users) | P2 |
| 20 | Download progress bar is broken — downloads work but no incremental progress | P1 |

| 21 | Settings are read-only — no way to edit values inline | P1 |
| 22 | Some settings should be read-only (data dir, etc.) — need writable vs readonly distinction | P1 |
| 23 | HuggingFace API token visible in plain text — must be masked and non-editable from settings | P0 |

| 24 | Vision model regression — worked with Ollama, broken with native llama-cpp management | P0 |
| 25 | Vision mmproj file shows in dropdown — should be hidden, paired with main model | P1 |

## Prioritized phases (self-contained — any agent can execute)

### Phase 1: Critical bugs (P0) — must fix first

**1a. "bad value(s) in fds_to_keep" on chat (#12)**
- Root cause: `WorkerProcess` uses `multiprocessing.Process(start_method="spawn")`. On macOS, spawning a subprocess while file descriptors are open causes this error. The subprocess worker starts lazily on first embed call, but if chat triggers it (or if the worker is started during app init), fd inheritance breaks.
- Investigate: `src/lilbee/providers/worker_process.py` — check when worker starts, what fds are open. May need `close_fds=True` or `multiprocessing.set_start_method("spawn", force=True)` at module level.
- Test: `lilbee chat` should work without errors.

**1b. Ctrl+C doesn't quit during downloads/vision (#1)**
- Root cause: Textual's `action_quit` checks `_streaming` flag and calls `self.exit()`. But during a `@work(thread=True)` download or vision call, the native C code holds the GIL and Ctrl+C signal is deferred. The quit action fires but `self.exit()` can't run until the GIL is released.
- Fix: `action_quit` in `app.py` should `SIGTERM` the worker subprocess if running, cancel all Textual workers, then exit. Use `os._exit(1)` as last resort after timeout.
- Files: `src/lilbee/cli/tui/app.py` (action_quit)

**1c. Vision stdout corrupts TUI (#2)**
- Root cause: The subprocess worker was implemented but vision may not be routing through it. Check if `vision.py:extract_page_text()` actually uses `provider.vision_ocr()` (subprocess path) or falls back to `provider.chat()` (in-process, leaks stdout).
- Investigate: `src/lilbee/vision.py`, `src/lilbee/providers/llama_cpp_provider.py` — is `vision_ocr` method present? Is `subprocess_embed` config True?
- Also: redirect fd 1 and fd 2 in `_worker_main` so child process stdout/stderr go to a log file, not the terminal.

**1d. Vision model regression + wrong chat handler (#17, #24, new)**
- Root cause FOUND: `_load_vision_llama()` hardcodes `Llava15ChatHandler` for ALL vision models. LightOnOCR-2 is NOT a Llava 1.5 model — it needs a different handler. This causes the CLIP projection to run on CPU (15s per image slice instead of <1s on GPU).
- Available handlers in llama-cpp-python: `Llava15ChatHandler`, `Llava16ChatHandler`, `MiniCPMv26ChatHandler`, `MoondreamChatHandler`, `NanoLlavaChatHandler`, `ObsidianChatHandler`, `Qwen25VLChatHandler`, `Llama3VisionAlphaChatHandler`
- Fix: Read model architecture from GGUF metadata (`general.architecture` or `projector` field in mmproj). Map to correct handler. Fallback to Llava15 if unknown.
- File: `src/lilbee/providers/llama_cpp_provider.py` line 457-485 (`_load_vision_llama`)
- Also: mmproj pairing — `_find_mmproj_for_model()` needs to correctly match mmproj files to their parent model.
- Files: `src/lilbee/providers/llama_cpp_provider.py`, `src/lilbee/providers/worker_process.py`

**1e. Download crashes with NoActiveAppError (#3)**
- Root cause: `_run_download` in `catalog.py` calls `self.app.call_from_thread(bar.complete_task, task_id)` but if the user navigated away from the catalog screen, the app context is lost.
- Fix: Guard `call_from_thread` calls with try/except for `NoActiveAppError`. Or store the task_bar reference before starting the worker.
- File: `src/lilbee/cli/tui/screens/catalog.py`

**1f. HF token visible in settings (#23)**
- Fix: Mask the token value in the settings display (show `****...` or `[set]`). Mark as non-editable.
- File: `src/lilbee/cli/tui/screens/settings.py`

### Phase 2: Navigation and UI (P1)

**2a. hjkl doesn't work in normal mode (#5, #11, #14)**
- Root cause: h/l are bound to `action_nav_prev/next` at app level, which cycles views. In normal mode on chat screen, j/k should move focus between UI elements (model dropdowns, chat log, input) and h/l should still cycle views.
- Fix: In normal mode, j/k moves focus within the current screen's focusable widgets (model bar selects, chat log, input). Escape enters normal mode, any typing or Enter on input returns to insert mode.
- Files: `src/lilbee/cli/tui/screens/chat.py` (key handlers, focus cycling)

**2b. Mode indicator (#6)**
- Add `-- NORMAL --` or `-- INSERT --` text to NavBar or a status line, like vim.
- File: `src/lilbee/cli/tui/widgets/nav_bar.py` or new mode indicator widget

**2c. Model dropdowns: wrong types, no labels, mmproj visible (#9, #10, #25)**
- Chat dropdown should only show chat models, embed dropdown only embedding models, vision dropdown only vision models.
- Each dropdown needs a label: "Chat:", "Embed:", "Vision:"
- mmproj files should be hidden from all dropdowns — they're paired with their parent model.
- Files: `src/lilbee/cli/tui/widgets/model_bar.py` (or wherever dropdowns are), `src/lilbee/model_manager.py` (filter by task)

**2d. Move progress out of chat, into task center (#4, #15, #16)**
- Remove TaskBar widget from chat screen compose. Progress should only show in NavBar (one-line summary) and TaskCenter (full details).
- When tasks are active, highlight "Tasks" in NavBar (bold, color).
- Files: `src/lilbee/cli/tui/screens/chat.py` (remove TaskBar yield), `src/lilbee/cli/tui/widgets/nav_bar.py` (highlight active)

**2e. Separate task queues (#4)**
- Downloads, syncs, crawls should not block each other. Current TaskQueue is single-FIFO.
- Change to: one queue per task type (download, sync, crawl), each can run one task at a time concurrently.
- Files: `src/lilbee/cli/tui/task_queue.py`, `src/lilbee/cli/tui/widgets/task_bar.py`
- Mirror in Obsidian plugin: `/Users/tobias/projects/obsidian-lilbee` (separate repo)

**2f. Settings editable (#21, #22)**
- Press Enter on a writable setting to edit inline. Read-only settings show a lock icon.
- Config fields already have `writable` metadata. Use it.
- File: `src/lilbee/cli/tui/screens/settings.py`

**2g. Settings show model defaults and architecture (#18)**
- When a model has defaults (from GGUF/Ollama), show them in settings as the effective value when user hasn't overridden.
- Show model architecture info: chat model architecture, embedding model architecture, vision model architecture + projector type, chat handler being used. These should be visible (read-only) so the user can verify the right handler/architecture is detected. Values must never be None — show "not loaded" or "unknown" instead.
- File: `src/lilbee/cli/tui/screens/settings.py`, `src/lilbee/config.py`

**2h. Download progress broken (#20)**
- huggingface_hub download may not expose incremental progress to our callback. Investigate `hf_hub_download` progress hooks. May need a custom `tqdm` callback or polling the file size.
- File: `src/lilbee/catalog.py` (download_model)

### Phase 3: Polish (P2) — defer

- #7: NavBar at top (vim-airline style)
- #13: Catalog family grouping and sort/filter UX
- #16: Tasks view highlight animation
- #19: Mouse/click support for all views

### Phase 4: Obsidian plugin (lowest priority — do last)

- Mirror task queue changes (separate queues per type)
- Plugin repo: `/Users/tobias/projects/obsidian-lilbee`
- Only after all TUI work is stable and merged

## Execution notes for delegated agents

- Branch: `feature/tui-improvements` (PR #43 targets `feature/tui-fixes-and-models`)
- Run `make check` before committing (mirrors CI)
- No emojis in code or strings
- No Co-Authored-By in commits
- 100% test coverage on new code
- Small functions (max ~20 lines)
- Use worktrees for parallel work
- Phase 1 items can mostly be parallelized (1a-1f are independent)
- Phase 2 items have some dependencies: 2a before 2b, 2c independent, 2d+2e together
