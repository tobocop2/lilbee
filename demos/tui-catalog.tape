Source demos/_shared.tape
Output demos/_out/tui-catalog.gif

# Model catalog tour: every inner tab, grid/list toggle, infinite scroll,
# info modal on a local row, HF Hub search, real download of a tiny model.
# Measured timings:
#   lilbee cold launch                          ~2.5s
#   inner-tab switch (1..6, priority bindings)  <0.5s
#   `/` filter, type "smol", Enter              ~3.5s HF Hub round-trip
#   SmolLM2 135M cold pull                      ~30s (101 MB, bartowski mirror)
Env LILBEE_DATA "/tmp/lilbee-demo/tui-catalog"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

Escape
Sleep 1s

# Open Catalog (next screen tab from Chat).
Type "]"
Sleep 1500ms

# Tour every inner tab: Discover / Chat / Embed / Vision / Rerank / Library.
# Keys 1..6 are priority bindings -- they win over the search input.
# Slower sleeps (3s per tab) give the viewer time to actually read each
# pane before the next jump rather than feeling like a slot-machine.
Type "1"
Sleep 3s
Type "2"
Sleep 3s
Type "3"
Sleep 3s
Type "4"
Sleep 3s
Type "5"
Sleep 3s
Type "6"
Sleep 3s

# Back to Discover for the rest of the tour.
Type "1"
Sleep 2s

# Grid <-> list toggle.
Type "v"
Sleep 2s
Type "v"
Sleep 2s

# Infinite scroll: load more pages.
Type "n"
Sleep 1500ms
Type "n"
Sleep 1500ms

# Info modal on a local installed row. `i` notifies "frontier no info" on
# HF Hub rows, so we trigger it from Library where Qwen3 0.6B lives.
Type "6"
Sleep 1500ms
Type "i"
Sleep 3s
Escape
Sleep 800ms

# Back to the Chat sub-tab for the HF Hub search.
#
# The `/` filter input is hidden on Discover and Library by design; only
# the task-scoped tabs (Chat / Embed / Vision / Rerank) show it. Search
# for "smol" from Chat -> first visible match installs on Enter (the
# screen's _on_search_submitted handler picks the first card).
Type "2"
Sleep 1500ms

Screenshot demos/_out/tui-catalog.png

Type "/"
Sleep 400ms
Type "smol"
# Wait for the HF Hub round-trip so the SmolLM cards land in the grid
# BEFORE Enter fires; otherwise Enter falls through to "trigger remote
# search" instead of "install first visible card".
Sleep 5s
Enter
Sleep 6s

# Enter on the search input installs the first visible match.
# Switch to Tasks to watch the download. Escape releases the search
# input focus so the screen-nav `]` fires.
Escape
Sleep 400ms
Type "]"
Sleep 400ms
Type "]"
Sleep 400ms
Type "]"
Sleep 800ms
Sleep 32s

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
