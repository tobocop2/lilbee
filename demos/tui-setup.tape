Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run experience: no models installed, splash + setup wizard appear.
# Note: with /tmp/lilbee-demo/models pre-populated by demos/_prep.sh, the
# wizard's `download` will report "already installed" instantly. To show a
# real download, unset LILBEE_MODELS_DIR and let lilbee pull into the
# data-dir's models/ subdir (Qwen3 0.6B pulls in a few seconds).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 4s

# The Setup Wizard opens automatically when no chat/embedding model is set.
# Enter on the pre-selected Qwen3 0.6B chat card.
Enter
Sleep 2s

# Move to the embedding grid and select Nomic v1.5.
Tab
Sleep 1s
Enter
Sleep 2s

# Open the Task Center to show progress (downloads finish fast with cache).
Escape
Sleep 600ms
Type "t"
Sleep 4s

Screenshot demos/_out/tui-setup.png

# Back out.
Type "q"
Sleep 600ms
Type "/quit"
Sleep 500ms
Enter
Sleep 1s
