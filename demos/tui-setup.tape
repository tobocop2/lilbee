Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run wizard. Qwen3 4B + Nomic v1.5 are pre-cached in setup-models
# by demos/_prep.sh so the wizard reports "Your existing models are
# ready" -- the chat model is real and works for the answer at the end.
# Then the demo demonstrates the Task Center download flow by pulling
# SmolLM2 135M (~100 MB) live from the catalog. SmolLM is NOT set as
# the chat model; chat continues to use the pre-cached Qwen 4B.
#
# Insert/normal mode flow (empirically verified in tmux):
#   wizard Esc        -> chat INSERT
#   Esc               -> NORMAL (necessary for `]` navigation)
#   `]` `2` `/`       -> Catalog, Chat tab, filter search
#   "smol" Enter      -> HuggingFace search returns SmolLM cards
#   Right Right Enter -> pick SmolLM2 135M, install
#   Esc               -> close filter
#   `t`               -> Tasks Center (download visible)
#   `]`               -> back to Chat in NORMAL
#   `i`               -> INSERT (necessary for typing /add)
#   /add typed+Enter  -> stays INSERT
#   Esc               -> NORMAL (necessary for `t`)
#   `t`               -> Tasks Center (see /add done)
#   `]`               -> back to Chat in NORMAL
#   `i`               -> INSERT (necessary for typing the question)
# Four mode toggles total. Each is necessary.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
# Point Ollama at an unused port so the wizard ignores any locally
# installed Ollama models.
Env OLLAMA_HOST "http://127.0.0.1:65535"

Type "cd /tmp/lilbee-demo/tui-setup"
Sleep 300ms
Enter
Sleep 500ms

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Wizard appears. Models are pre-cached so the footer reads "Your
# existing models are ready - Esc to return". Esc dismisses into chat.
Sleep 1500ms
Escape
Sleep 1000ms

# Drop chat from INSERT to NORMAL so `]` works as screen navigation.
Escape
Sleep 600ms

# `]` Catalog. `2` selects the Chat tab. `/` opens the filter input.
Type "]"
Sleep 1200ms
Type "2"
Sleep 1200ms
Type "/"
Sleep 400ms
Type "smol"
Sleep 800ms
Enter
Sleep 5s

# Two Rights focus the SmolLM2 135M card (the smallest, fastest pull).
# Enter kicks off the install.
Right
Sleep 500ms
Right
Sleep 600ms
Enter
Sleep 1500ms

# Close the filter, drop to NORMAL.
Escape
Sleep 500ms

# Tasks Center to watch the SmolLM2 135M download. Hide the slow
# middle of the ~60s pull; Show resumes at "done".
Type "t"
Sleep 8s
Hide
Sleep 55s
Show
Sleep 4s

# Back to Chat. `]` from Tasks wraps to Chat, lands in NORMAL.
Type "]"
Sleep 1200ms

# INSERT for typing the /add command.
Type "i"
Sleep 400ms
Type "/add ./README.md"
Sleep 1500ms
Enter
Sleep 1500ms

# Drop to NORMAL for `t`.
Escape
Sleep 400ms
Type "t"
Sleep 4s

# Back to Chat via `]`.
Type "]"
Sleep 1000ms

# INSERT for typing the question.
Type "i"
Sleep 400ms
Type "What is lilbee?"
Sleep 1500ms
Enter
Sleep 1500ms

# Hide the cold first-token wait for Qwen3 4B (~30-60s on M1 fresh
# process). Show resumes for the visible streamed answer.
Hide
Sleep 80s
Show
Sleep 10s

Screenshot demos/_out/tui-setup.png
Sleep 400ms

Type "/quit"
Enter
Sleep 500ms
