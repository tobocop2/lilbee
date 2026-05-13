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
Type "1"
Sleep 1800ms
Type "2"
Sleep 1800ms
Type "3"
Sleep 1800ms
Type "4"
Sleep 1800ms
Type "5"
Sleep 1800ms
Type "6"
Sleep 1800ms

# Back to Discover for the rest of the tour.
Type "1"
Sleep 1200ms

# Grid <-> list toggle.
Type "v"
Sleep 1500ms
Type "v"
Sleep 1500ms

# Infinite scroll: load more pages.
Type "n"
Sleep 1200ms
Type "n"
Sleep 1200ms

# Info modal on a local installed row. `i` notifies "frontier no info" on
# HF Hub rows, so we trigger it from Library where Qwen3 0.6B lives.
Type "6"
Sleep 1500ms
Type "i"
Sleep 3s
Escape
Sleep 800ms

# Back to Discover, search HF Hub for a small model.
Type "1"
Sleep 1200ms

Screenshot demos/_out/tui-catalog.png

Type "/"
Sleep 400ms
Type "smol"
Sleep 1500ms
Enter
Sleep 3500ms

# Focus the SmolLM2 135M card (third of four).
Right
Sleep 300ms
Right
Sleep 300ms

# Pull it. Queues into Task Center; the catalog grid stays focused.
Enter
Sleep 1s

# Switch to Tasks so the download progress is visible on camera. Four `[`
# walks back: Catalog -> Status -> Settings -> Tasks (wrap), but we just
# use `t` from NORMAL mode after Escape.
Escape
Sleep 300ms
Type "t"
Sleep 32s

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
