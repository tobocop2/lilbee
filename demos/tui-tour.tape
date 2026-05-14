Source demos/_shared.tape
Output demos/_out/tui-tour.gif

# ~85s composite tour: chat with citations, catalog inner tabs, settings
# sub-tabs, task center, help panel, F2 slash catalog. Even coverage of every
# surface (per user feedback: cover all evenly, don't lean on just two).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-tour"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Off-camera warm-up so the first on-camera chat turn is warm.
Hide
Type "ping"
Sleep 400ms
Enter
Sleep 45s
Type "/clear"
Sleep 400ms
Enter
Sleep 1s
Show

# 1. Chat with citations. Qwen3 4B's full answer typically streams for
# 18-22s including the Sources line; sleep budget covers the worst case.
Type "what's my oil capacity?"
Sleep 400ms
Enter
Sleep 25s

Screenshot demos/_out/tui-tour.png

# 2. Catalog: two inner tabs.
Type "/models"
Sleep 400ms
Enter
Sleep 2500ms
Type "1"
Sleep 1500ms
Type "4"
Sleep 1500ms

# 3. Settings: cycle through a few sub-tabs.
Type "/settings"
Sleep 400ms
Enter
Sleep 2s
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms

# 4. Task Center.
Escape
Sleep 300ms
Type "t"
Sleep 1800ms

# 5. Help panel.
Type "?"
Sleep 2500ms
Escape
Sleep 500ms

# 6. Slash-command catalog (same modal F2 opens; VHS can't send F-keys, so
# we reach it via the /help slash command).
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms
Escape
Sleep 500ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
