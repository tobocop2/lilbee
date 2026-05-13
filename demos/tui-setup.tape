Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run experience: no models installed, splash + setup wizard appear.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 4s

# The Setup Wizard opens automatically when no chat / embedding model is set.
# Arrow to the Qwen3 0.6B card (it should be pre-selected as the RAM-fit recommendation),
# Enter to queue the download.
Enter
Sleep 1500ms

# Move to the embedding grid and select Nomic v1.5.
Tab
Sleep 800ms
Enter
Sleep 1500ms

# Open the Task Center to show the two downloads ticking.
Escape
Sleep 800ms
Type "t"
Sleep 5s

Screenshot demos/_out/tui-setup.png

# Back to chat.
Type "q"
Sleep 800ms
Ctrl+C
Sleep 1s
