Source demos/_shared.tape
Output demos/_out/tui-tour.gif

# ~60-90s montage: chat with citations, slash-command catalog, catalog,
# settings, theme switch. Source for the hero clip in the README.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-tour"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Chat with citations (first answer; folds model load into the 18s).
Type "what does the oil pressure warning mean?"
Sleep 500ms
Enter
Sleep 18s

Screenshot demos/_out/tui-tour.png

# Help panel (covers a lot of the command surface in one frame).
# (F-keys aren't a top-level VHS command; `?` toggles help in lilbee.)
Escape
Sleep 400ms
Type "?"
Sleep 4s
Type "?"
Sleep 600ms
Type "i"
Sleep 400ms

# Catalog.
Type "/models"
Sleep 500ms
Enter
Sleep 3s
Type "V"
Sleep 2s

# Settings.
Escape
Sleep 400ms
Type "["
Sleep 800ms
Type "i"
Sleep 400ms
Type "/settings"
Sleep 500ms
Enter
Sleep 3s

# Task Center.
Escape
Sleep 400ms
Type "["
Sleep 800ms
Type "t"
Sleep 3s
Type "q"
Sleep 800ms

# Theme switch.
Ctrl+t
Sleep 2s
Ctrl+t
Sleep 2s

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
