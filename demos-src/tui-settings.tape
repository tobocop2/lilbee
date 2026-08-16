Source _shared.tape
Output _out/tui-settings.gif

# Settings tour: cycle through every sub-tab via `>` and use `j` to
# scroll the longer panes (Generation, Retrieval, Crawling).
#
# Critical: do NOT Tab into individual fields. Tab focuses an Input
# widget, and once any input has focus, `>` / `q` / `/` all type as
# literal characters into the field. Stay on screen-level bindings:
# `>` (priority, cycle_pane), `j`/`k` (scroll, no focus), `q` (back).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-settings"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Two Escapes: dismiss wizard, drop chat to NORMAL.
Escape
Sleep 500ms
Escape
Sleep 500ms

# Walk to Settings: 3 `]` presses (Chat -> Catalog -> Status -> Settings).
Type "]"
Sleep 700ms
Type "]"
Sleep 700ms
Type "]"
Sleep 1500ms

# Models pane (first, default). Hold a moment.
Sleep 2s

# Ingest pane.
Type ">"
Sleep 2s

# Generation pane: 17 fields, the longest. Scroll through with `j`.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Retrieval pane: 20 fields. Scroll the full pane.
Type ">"
Sleep 2s

Screenshot _out/tui-settings.png

Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Display pane: short, but still scroll a bit so the panel isn't static.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Crawling pane (13 fields). Scroll the full pane.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# API-Keys pane: a couple of scrolls.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# System pane: scroll through diagnostics.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Back to Chat via `q`. No fields are focused (we never Tabbed in),
# so `q` fires the Settings screen's go_back binding cleanly.
Type "q"
Sleep 1000ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
