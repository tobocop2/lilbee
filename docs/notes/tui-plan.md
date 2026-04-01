# lilbee TUI — Remaining Work

Branch: `feature/tui-fixes-and-models`
Last commit: `28ae73c` — fix catalog _current_tab crash, remove Rich Console from /add

## What's working now
- NavBar at bottom: `1:Chat 2:Models 3:Status 4:Settings ? Help ^c Quit`
- Escape blurs input → 1-4/j/k/? work → typing refocuses input
- TaskBar for downloads/syncs/crawls (auto-hides when idle)
- Catalog with family grouping, clean names, quality tiers
- HF token via `huggingface_hub.get_token()` fallback
- `/add` uses `copy_files()` not Rich Console (no TUI conflict)
- Memory-aware model cache (LRU, keep_alive)
- Type-safe progress callbacks (Pydantic models)
- Context-aware Ctrl+C (cancel task → cancel stream → quit)

## What needs implementing (in order)

### 1. Model dropdowns — DONE (commit 39703ee)
Select widgets for chat/embed/vision. Populated via `@work(thread=True)` after mount. `_populating` flag guards against change events during init. Vision allows blank.

**Files:** `src/lilbee/cli/tui/widgets/model_bar.py`

### 2. Editable settings screen — DONE (commit 271b502)
Enter on a row opens inline Input, Enter saves via `set_value()`, Escape cancels. Read-only fields guarded. Still needs `/set key value` slash command in chat.

**Files:** `src/lilbee/cli/tui/screens/settings.py`
**TODO:** Wire `/set` slash command in `chat.py` + `command_registry.py`

### 3. Tab/arrow navigation between widgets
Tab should cycle: ModelBar dropdowns → Chat input → (back to top)
When chat log focused (via Escape): j/k scroll, Tab goes to next widget

**Files:** `src/lilbee/cli/tui/screens/chat.py`, `src/lilbee/cli/tui/theme.tcss`

### 4. Catalog pagination rework
"Load more" at bottom needs better UX — infinite scroll or explicit button.

## Known bugs still present
- Vision models partially working (mmproj download works but end-to-end vision OCR untested)
- No task center view (user wants 5:Tasks in NavBar for monitoring background operations)
- Stale stash at `stash@{0}` contains broken features agent work — can drop

## DONE (recent fixes)
- Catalog `_current_tab` crash fixed (uses TabbedContent.active now)
- Rich Console removed from `/add` (uses copy_files() directly)
- NavBar persistence fixed (app-level, not per-screen)
- Tab key limited to completions only
- Catalog filter less intrusive (hidden by default, shows with `/`)

## Architecture rules
- NO Rich Console in `src/lilbee/cli/tui/` — use Textual widgets only
- Progress callbacks use `ProgressEvent` union type (Pydantic models, not `dict[str, Any]`)
- NavBar lives on ChatScreen (not App — Textual screens own their widgets)
- TaskBar auto-hides when empty (`display=False`)
- `_hf_token()` in `catalog.py` reads env vars then falls back to `huggingface_hub.get_token()`
- App-level Bindings for 1-4 (view switching) with `show=False`
- All `show=True` bindings in Footer/NavBar — Textual auto-prepends the key name, so labels should NOT include the key (e.g., "Back" not "q Back")

## File layout
```
src/lilbee/cli/tui/
  __init__.py          — run_tui(), KeyboardInterrupt handler
  app.py               — LilbeeApp, BINDINGS (1-4, ?, F1-F4, ^c)
  theme.tcss           — all CSS
  command_registry.py  — slash commands
  commands.py          — command palette provider
  task_queue.py        — FIFO queue for background operations
  screens/
    chat.py            — main chat screen, compose, streaming
    catalog.py         — model browser with families
    settings.py        — read-only settings display
    status.py          — knowledge base status
    setup.py           — first-run wizard
    task_center.py     — task history (if exists)
  widgets/
    model_bar.py       — model status (currently Static, needs Select)
    nav_bar.py         — cmus-style 1:Chat 2:Models tabs
    task_bar.py        — download/sync/crawl progress
    message.py         — UserMessage, AssistantMessage (streaming markdown)
    help_modal.py      — ? help overlay
    autocomplete.py    — slash command completion
    suggester.py       — input suggestions
```

## Testing requirements
Every fix/feature MUST have integration or e2e tests before marking complete.
Key test files: `tests/test_tui_widgets.py`, `tests/test_tui_screens.py`, `tests/test_tui.py`
