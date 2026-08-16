Source "_shared.tape"
Output "_out/tui-unsupported.gif"
Output "_out/tui-unsupported.mp4"

# Architecture-compatibility demo: search HF Hub for "deepseek-v4" (an arch
# that isn't in the bundled llama.cpp build's MODEL_ARCH_NAMES) and
# show the rows surfacing with the `unsupported` pill in both grid
# and list view. Trying to pull one opens the ConfirmDialog gate.
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

# Catalog (next screen tab from Chat).
Type "]"
Sleep 1500ms

# Move to the Chat task tab (numeric 1..6 are priority bindings).
Type "2"
Sleep 1500ms

# Open the filter, type "deepseek-v4", trigger the HF round-trip.
Type "/"
Sleep 400ms
Type "deepseek-v4"
Sleep 1500ms
Enter
Sleep 12s

# Grid view shows the deepseek-v4 rows, each with `unsupported` on the
# secondary pill row.
Screenshot "_out/tui-unsupported-grid.png"

# Switch to list view -- secondary pills become italic tags after the
# row name; "unsupported" still surfaces on every deepseek-v4 row.
Escape
Sleep 400ms
Type "v"
Sleep 2500ms
Screenshot "_out/tui-unsupported-list.png"

# Back to grid for the final frame so the captured GIF closes on the
# stacked-pill layout.
Type "v"
Sleep 1500ms
Screenshot "_out/tui-unsupported.png"

# Quit cleanly.
Escape
Sleep 400ms
Type "]"
Sleep 400ms
Type "]"
Sleep 800ms
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
