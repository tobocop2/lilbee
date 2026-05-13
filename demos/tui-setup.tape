Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# Fresh-slate first run. Wizard appears with no installed models; we pick a
# tiny chat model (SmolLM2 135M, ~30s cold pull) and the default embed model
# (Nomic v1.5, ~25s cold pull) and watch the Task Center fill before landing
# on Chat. No Hide/Show -- both pulls are small enough to show live.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Discover tab focused; navigate right to SmolLM2 135M and pick it.
Right
Sleep 300ms
Right
Sleep 300ms
Enter
Sleep 800ms

# Tab into the Embedding row, pick Nomic v1.5 (first card).
Tab
Sleep 300ms
Enter
Sleep 800ms

# Watch the two downloads in Task Center (~30s window covers both).
Type "t"
Sleep 30s

Screenshot demos/_out/tui-setup.png

# Back to the wizard so the user sees the two cards now installed.
Type "q"
Sleep 1500ms

# Dismiss into Chat.
Escape
Sleep 2000ms

Type "/quit"
Enter
Sleep 500ms
