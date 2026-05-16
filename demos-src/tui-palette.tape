Source _shared.tape
Output _out/tui-palette.gif

# Command discovery surface: the Textual command palette (Ctrl+P) lists
# every registered action and filters as you type. We show three filters
# in succession so the viewer sees how the palette's a navigator, not a
# one-shot.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-palette"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Two Escapes: dismiss wizard + drop chat into NORMAL so subsequent
# Ctrl+P always reaches the app palette binding.
Escape
Sleep 500ms
Escape
Sleep 500ms

# Open palette. Default view shows every registered action.
Ctrl+P
Sleep 2s

# 1. Filter "task" -> Open Tasks (stays in rose-pine).
Type "task"
Sleep 1500ms
Enter
Sleep 1500ms

# Back to Chat so the next palette action makes sense.
Type "["
Sleep 500ms
Type "["
Sleep 500ms
Type "["
Sleep 500ms
Type "["
Sleep 800ms

# 2. Filter "model" -> Models bar focus (stays in rose-pine).
Ctrl+P
Sleep 800ms
Type "model"
Sleep 1500ms
Escape
Sleep 600ms

# 3. Slash-command catalog (stays in rose-pine).
Type "i"
Sleep 300ms
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms
Escape
Sleep 600ms

# 4. Brief theme cycle, then restore rose-pine so the screenshot lands
# on the canonical theme. The cycle is the last palette demo so the
# alternate-theme frames are minimal.
Escape
Sleep 300ms
Ctrl+P
Sleep 600ms
Type "theme"
Sleep 1200ms
Enter
Sleep 1500ms

# Restore rose-pine inline.
Type "i"
Sleep 300ms
Type "/theme rose-pine"
Sleep 600ms
Enter
Sleep 2s

Screenshot _out/tui-palette.png
Sleep 500ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
