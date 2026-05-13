Source demos/_shared.tape
Output demos/_out/tui-chat.gif

# Chat with the Crown Victoria owner's manual. Models + index are pre-staged
# by demos/_prep.sh; LILBEE_DATA points at the prepared corpus.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-chat"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# The signature question (matches the README copy).
Type "what does the oil pressure warning mean?"
Sleep 500ms
Enter
Sleep 14s

Screenshot demos/_out/tui-chat.png

# Crisp factual follow-ups Qwen3 0.6B nails: each one is a single-fact lookup.
Type "what's my oil capacity?"
Sleep 500ms
Enter
Sleep 12s

Type "what tire pressure does it recommend?"
Sleep 500ms
Enter
Sleep 12s

Type "what fuel does it take?"
Sleep 500ms
Enter
Sleep 10s

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
