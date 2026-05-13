Source demos/_shared.tape
Output demos/_out/tui-add.gif

# Empty library, add a document from inside the TUI, watch it sync, check status.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-add"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Add the manual with tab-completion on the path.
Type "/add ~/Downloads/cv-man"
Sleep 600ms
Tab
Sleep 800ms
Enter
Sleep 18s

# Status screen shows the new source and chunk count.
Type "/status"
Sleep 400ms
Enter
Sleep 3s

Screenshot demos/_out/tui-add.png

# Ask a question to confirm the corpus is live.
Type "["
Sleep 500ms
Type "what is my oil capacity?"
Sleep 400ms
Enter
Sleep 12s

Type "/quit"
Sleep 400ms
Enter
Sleep 1s
