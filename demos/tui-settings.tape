Source demos/_shared.tape
Output demos/_out/tui-settings.gif

# Settings tour: cycle tabs, edit a field, reset.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-settings"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

Type "/settings"
Sleep 400ms
Enter
Sleep 2s

# Cycle a few tabs.
Type "]"
Sleep 1500ms
Type "]"
Sleep 1500ms
Type "]"
Sleep 1500ms

Screenshot demos/_out/tui-settings.png

# Tab into the first editable field, type a new value, confirm.
Tab
Sleep 600ms
Ctrl+a
Sleep 200ms
Type "8"
Sleep 800ms
Enter
Sleep 1500ms

# Reset that one field.
Ctrl+r
Sleep 1500ms

# Back to chat.
Type "q"
Sleep 800ms
Type "/quit"
Sleep 400ms
Enter
Sleep 1s
