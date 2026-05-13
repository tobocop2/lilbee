Source demos/_shared.tape
Output demos/_out/tui-settings.gif

# Settings screen still. Keep the in-screen navigation minimal because the
# Settings screen has its own (non-`]`) keybindings for tabs/fields and we
# don't have those mapped out for VHS reproducibly.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-settings"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

Type "/settings"
Sleep 500ms
Enter
Sleep 3s

Screenshot demos/_out/tui-settings.png

Sleep 2s

Type "q"
Sleep 800ms

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
