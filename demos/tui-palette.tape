Source demos/_shared.tape
Output demos/_out/tui-palette.gif

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
Sleep 3s

# 1. Filter "theme" -> Cycle Theme. Watch the chrome shift.
Type "theme"
Sleep 2s
Enter
Sleep 2s

# 2. Open palette again, filter "task" -> Open Tasks (jumps to Tasks screen).
Ctrl+P
Sleep 1s
Type "task"
Sleep 2s
Enter
Sleep 2s

# Back to Chat so the next palette action makes sense.
Type "["
Sleep 800ms
Type "["
Sleep 800ms
Type "["
Sleep 800ms
Type "["
Sleep 800ms

# 3. Filter "model" -> Models bar (jumps focus to the model picker bar).
Ctrl+P
Sleep 1s
Type "model"
Sleep 2s
Escape
Sleep 800ms

# 4. Slash-command catalog: same modal F2 opens, reachable via /help.
Type "i"
Sleep 300ms
Type "/help"
Sleep 400ms
Enter
Sleep 3s

Sleep 1s
Escape
Sleep 800ms

# Restore rose-pine explicitly so the GIF starts AND ends in the same
# theme. Without this, "Cycle theme" leaves the demo on whatever the
# next theme in the cycle happens to be (usually a light theme), which
# clashes with the rose-pine palette of every other tape.
#
# Two Escapes here are defensive: closing the /help modal returns chat
# to INSERT in some Textual states, so the leading `i` of "i/theme..."
# would land as content (the chat would send "i/theme rose-pine" as a
# user message). A second Escape forces NORMAL; THEN the explicit `i`
# enters INSERT and `/theme rose-pine` types into the cleared input.
Escape
Sleep 400ms
Type "i"
Sleep 300ms
Type "/theme rose-pine"
Sleep 800ms
Enter
Sleep 2500ms

Screenshot demos/_out/tui-palette.png
Sleep 1s

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
