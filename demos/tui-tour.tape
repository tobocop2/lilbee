Source demos/_shared.tape
Output demos/_out/tui-tour.gif

# ~85s composite tour: chat with citations, catalog inner tabs, settings
# sub-tabs, task center, help panel, /help slash catalog. Even coverage.
#
# Navigation rules learned the hard way:
#   * Slash commands work only on Chat. From Catalog, `/` opens the
#     catalog filter input. From Settings, `/` opens the settings
#     search input. Both swallow the rest of the line.
#   * To navigate AWAY from any screen, use `]` (priority binding).
#   * After ANY chat submit, the chat input keeps focus and is in
#     INSERT; printable keys (`>`, `]`, `t`, etc.) type as literals.
#     Escape sets the chat input unfocusable so the bindings fire.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-tour"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
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

# 1. Chat with citations.
Type "what's my oil capacity?"
Sleep 400ms
Enter
Sleep 25s

Screenshot demos/_out/tui-tour.png

# Drop into NORMAL so the screen-nav bindings work.
Escape
Sleep 400ms

# 2. Catalog (one `]` from Chat).
Type "]"
Sleep 1500ms
Type "1"
Sleep 1500ms
Type "4"
Sleep 1500ms

# 3. Settings (two `]` from Catalog: Status -> Settings).
Type "]"
Sleep 800ms
Type "]"
Sleep 1500ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms

# 4. Task Center (one `]` from Settings).
Type "]"
Sleep 1800ms

# 5. Help panel.
Type "?"
Sleep 2500ms
Escape
Sleep 500ms

# 6. Slash-command catalog. Get back to Chat first (one `]` wraps to
# Chat from Tasks, or three `[` walks back), then /help opens the
# catalog modal.
Type "]"
Sleep 800ms
Type "i"
Sleep 300ms
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
