Source demos/_shared.tape
Output demos/_out/tui-tour.gif

# 60-90s montage: a quick sweep through chat, catalog, settings, task center,
# slash command catalog. Source for the hero clip in the README.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-tour"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Chat with citations.
Type "what does the oil pressure warning mean?"
Sleep 400ms
Enter
Sleep 14s

Screenshot demos/_out/tui-tour.png

# Slash-command catalog (covers the full command surface in one frame).
F2
Sleep 3s
Escape
Sleep 600ms

# Catalog.
Type "/models"
Sleep 400ms
Enter
Sleep 3s
Type "V"
Sleep 1500ms

# Settings.
Type "["
Sleep 600ms
Type "/settings"
Sleep 400ms
Enter
Sleep 3s

# Task center.
Type "q"
Sleep 600ms
Type "t"
Sleep 3s
Type "q"
Sleep 600ms

# Theme switch on screen for flair.
Ctrl+t
Sleep 1500ms
Ctrl+t
Sleep 1500ms

# Quit.
Type "/quit"
Sleep 400ms
Enter
Sleep 1s
