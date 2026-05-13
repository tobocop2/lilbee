Source demos/_shared.tape
Output demos/_out/tui-palette.gif

# Command surface tour: Textual command palette (Ctrl+P), the help panel
# (`?`), and the F2 slash-command catalog.
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

# Textual command palette: filterable action list.
Ctrl+P
Sleep 2s
Type "theme"
Sleep 1500ms
Escape
Sleep 800ms

# Help panel: keybinding cheat sheet for the current screen.
Type "?"
Sleep 2s
Escape
Sleep 800ms

# Slash-command catalog. Same modal F2 opens, but reachable via the /help
# slash command (VHS doesn't support F-keys).
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms

Screenshot demos/_out/tui-palette.png

Escape
Sleep 800ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
