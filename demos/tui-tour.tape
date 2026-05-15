Source demos/_shared.tape
Output demos/_out/tui-tour.gif

# ~90s composite tour. The chat answer streams in the background while we
# walk every screen and every sub-tab: Catalog (6 inner tabs), Status,
# Settings (8 panes), Tasks, command palette, slash-command catalog --
# then back to Chat to read the final cited answer. This is the demo for
# lilbee's async story: a local model is generating, you're free to keep
# using the UI.
#
# Navigation rules:
#   * Slash commands only work on Chat. From Catalog, `/` opens the
#     catalog filter input; from Settings, the settings search.
#   * To leave any screen, use `]` (priority binding, fires regardless
#     of focus).
#   * After a chat submit, the chat input stays in INSERT. Escape drops
#     to NORMAL (does NOT cancel the stream); only Ctrl+C cancels.
#   * `>` cycles the inner tab on Catalog / Settings (priority binding,
#     unprefixed key works in NORMAL only -- but on those screens chat
#     input isn't focused so it works there too).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-tour"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
Env OLLAMA_HOST "http://127.0.0.1:65535"

Type "lilbee"
Sleep 300ms
Enter
Sleep 3s

# Two Escapes: dismiss wizard, drop chat input to NORMAL. Then `i` to
# enter INSERT for the chat prompt.
Escape
Sleep 700ms
Escape
Sleep 700ms

# Ask the question -- answer streams for ~90s on a cold Qwen3 4B. We
# walk the whole UI during the stream.
Type "i"
Sleep 300ms
Type "what's my oil capacity?"
Sleep 500ms
Enter
Sleep 3s

# Drop to NORMAL so `]` and `>` fire as bindings instead of typing.
# Streaming continues in the background.
Escape
Sleep 800ms

# Catalog: 6 inner tabs (Discover / Chat / Embed / Vision / Rerank /
# Library). `>` cycles them.
Type "]"
Sleep 1800ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms

# Status screen.
Type "]"
Sleep 2500ms

# Settings: 8 sub-tabs (Models / Ingest / Generation / Retrieval / Display /
# Crawling / API-Keys / System).
Type "]"
Sleep 1500ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms

# Tasks.
Type "]"
Sleep 2500ms

# Help cheat sheet (App-level binding, fires from anywhere).
Type "?"
Sleep 2500ms
Escape
Sleep 500ms

# Back to Chat (one `]` from Tasks wraps to Chat). Re-enter INSERT just
# long enough to open the slash catalog, then dismiss.
Type "]"
Sleep 1500ms
Type "i"
Sleep 300ms
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms
Escape
Sleep 500ms

# By now the chat answer has streamed to completion. Drop to NORMAL one
# last time to capture the final state.
Escape
Sleep 800ms

Screenshot demos/_out/tui-tour.png
Sleep 3s

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
