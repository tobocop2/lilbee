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
Sleep 500ms
Enter
Sleep 3s

# Cycle a few tabs (each tab swap renders in <1s; Sleep 2s gives the viewer time).
Type "]"
Sleep 2s
Type "]"
Sleep 2s
Type "]"
Sleep 2s

Screenshot demos/_out/tui-settings.png

# Tab into the first editable field, edit, confirm.
Tab
Sleep 800ms
Ctrl+a
Sleep 400ms
Type "8"
Sleep 1s
Enter
Sleep 1500ms

# Reset that field.
Ctrl+r
Sleep 1500ms

Type "q"
Sleep 800ms
Type "/quit"
Sleep 500ms
Enter
Sleep 1s
