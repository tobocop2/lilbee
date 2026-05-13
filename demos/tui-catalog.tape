Source demos/_shared.tape
Output demos/_out/tui-catalog.gif

# Model catalog tour: tabs, grid / list view, search, model info modal.
# Measured: each slash-command + Enter renders the destination screen in
# under 1s; the actual work happens during chat / sync, neither of which
# fires here. Generous Sleeps give the viewer time to read.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-catalog"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

Type "/models"
Sleep 500ms
Enter
Sleep 3s

# Toggle to list view, then back to grid.
# Note: `[` and `]` are GLOBAL screen-nav keys (Chat/Catalog/Status/Settings/
# Tasks); they don't filter inside Catalog. Filter tabs in Catalog use the
# left sidebar / mouse / dedicated keys we'd need to discover, so this tape
# stays on the default view.
Type "V"
Sleep 2s
Type "V"
Sleep 2s

Screenshot demos/_out/tui-catalog.png

# Search the grid.
Type "/"
Sleep 500ms
Type "qwen"
Sleep 2500ms
Escape
Sleep 800ms

# Model info on the highlighted row.
Type "i"
Sleep 3s
Escape
Sleep 800ms

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
