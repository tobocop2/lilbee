Source demos/_shared.tape
Output demos/_out/tui-add.gif

# Empty library, add a document from inside the TUI, watch sync, check status.
# Measured timings:
#   /add ~/.../cv-manual.pdf -> sync done   ~22s  -> Sleep 28s after Enter
#   /status -> screen rendered             ~0.5s -> Sleep 3s
#   chat (warm)                            ~10s  -> Sleep 12s
Env LILBEE_DATA "/tmp/lilbee-demo/tui-add"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Add the manual. Tab-completion in the lilbee TUI keeps focus on the
# autocomplete dropdown and Enter selects rather than submits; type the full
# path and Enter submits cleanly.
Type "/add ~/Downloads/cv-manual.pdf"
Sleep 600ms
Enter
Sleep 28s

# Status screen shows the new source and chunk count.
Type "/status"
Sleep 500ms
Enter
Sleep 4s

Screenshot demos/_out/tui-add.png

# Back to chat (Escape to NORMAL mode, `[` for prev tab, `i` to re-enter INSERT).
Escape
Sleep 400ms
Type "["
Sleep 600ms
Type "i"
Sleep 400ms

# Ask a question to confirm the corpus is live.
Type "what is my oil capacity?"
Sleep 500ms
Enter
Sleep 12s

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
