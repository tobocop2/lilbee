Source demos/_shared.tape
Output demos/_out/tui-palette.gif

# Command discovery surface: Ctrl+P (Textual command palette), /help (the
# slash-command catalog). The palette lists every registered action and
# filters as you type; /help is the per-slash-command reference.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-palette"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

Escape
Sleep 1s

# Open the Textual command palette. Default view shows every registered
# action ("Cycle Theme", "Open Tasks", "Run Sync", "Show Help", ...).
Ctrl+P
Sleep 3s

# Filter the palette by typing a few letters. The action list narrows
# in real time.
Type "theme"
Sleep 2s

# Pick the highlighted "Cycle Theme" action.
Enter
Sleep 2s

# Cycle a second time so the viewer can see the palette is doing real
# work, not a one-shot.
Ctrl+t
Sleep 2s

# Switch to the slash-command catalog. Same modal F2 opens; we reach
# it via the /help slash command since VHS can't send F-keys.
Type "/help"
Sleep 400ms
Enter
Sleep 3s

Screenshot demos/_out/tui-palette.png

Sleep 1s
Escape
Sleep 800ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
