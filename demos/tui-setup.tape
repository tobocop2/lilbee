Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# Show the Setup Wizard. The wizard auto-opens on first launch only when no
# model is configured; with pre-staged models the TUI lands on Chat. Use
# `/setup` to open the wizard explicitly, so the tape is reproducible.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

Type "/setup"
Sleep 500ms
Enter
Sleep 3s

Screenshot demos/_out/tui-setup.png

# Pick a chat card (Enter on the pre-selected Qwen3 0.6B).
Enter
Sleep 2s

# Tab to the embedding grid and pick Nomic.
Tab
Sleep 1s
Enter
Sleep 2s

# Dismiss the wizard.
Escape
Sleep 1s

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
