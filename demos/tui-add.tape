Source demos/_shared.tape
Output demos/_out/tui-add.gif

# Empty library; add a document, watch ingest in the Task Center, then ask
# a question. /add takes ~39s cold for cv-manual.pdf (parse + chunk + embed
# + LanceDB write, measured).
#
# Per user feedback: the screenshot does NOT land on Status (we go to Tasks
# instead, the better-looking screen).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-add"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Dismiss the wizard. Pre-staged models are already installed.
Escape
Sleep 1s

# Submit the path. Autocomplete dropdown will be visible but Enter still
# fires the slash command (the dropdown only intercepts when an arg-hint is
# focused, not when we've typed a full path).
Type "/add ~/projects/lilbee-demos/demos/sample-corpus/cv-manual.pdf"
Sleep 600ms
Enter
Sleep 1s

# Drop into NORMAL mode FIRST: while the chat input is focused (the
# default after a slash submission) every printable character including
# `]` and `t` types as a literal into the prompt. Escape sets the chat
# input unfocusable and moves focus to the chat log; `t` then opens
# the Task Center via the priority binding.
Escape
Sleep 400ms
Type "t"
Sleep 38s

Screenshot demos/_out/tui-add.png

# Back to chat. From Tasks, `[` walks back: Tasks -> Settings -> Status
# -> Catalog -> Chat. 4 presses. Each press is a priority binding.
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 800ms

# Re-focus the chat input via `i` (NORMAL -> INSERT). `i` is bound only
# when chat is in NORMAL mode; it's a no-op in INSERT.
Type "i"
Sleep 300ms

# Confirm the corpus is live with a chat question.
Type "what's my oil capacity?"
Sleep 400ms
Enter

# Cold first turn after the wait above; Hide the bulk of the answer.
Sleep 4s
Hide
Sleep 35s
Show
Sleep 4s

Sleep 2s
Escape
Sleep 200ms
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
