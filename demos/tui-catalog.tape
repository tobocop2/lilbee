Source demos/_shared.tape
Output demos/_out/tui-catalog.gif

# Model catalog tour: tabs, grid / list view, search, model info modal.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-catalog"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Open the catalog.
Type "/models"
Sleep 400ms
Enter
Sleep 2s

# Toggle to list view, then back.
Type "V"
Sleep 1500ms
Type "V"
Sleep 1500ms

# Cycle the source filter tabs: Local -> Hugging Face -> Frontier -> Local.
Type "]"
Sleep 1500ms
Type "]"
Sleep 1500ms
Type "]"
Sleep 1500ms

Screenshot demos/_out/tui-catalog.png

# Quick search.
Type "/"
Sleep 400ms
Type "qwen"
Sleep 2s
Escape
Sleep 800ms

# Open model info on the highlighted row.
Type "i"
Sleep 3s
Escape
Sleep 600ms

# Back to chat.
Type "["
Sleep 800ms
Type "/quit"
Sleep 400ms
Enter
Sleep 1s
