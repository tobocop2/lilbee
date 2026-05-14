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

# Jump to the Task Center to watch the ingest run.
Escape
Sleep 300ms
Type "t"
Sleep 38s

Screenshot demos/_out/tui-add.png

# Back to chat to confirm the corpus is live. Model load + warm answer
# takes ~30-40s on the first question after /add.
Type "["
Sleep 1s
Type "what's my oil capacity?"
Sleep 400ms
Enter
Sleep 35s

Sleep 2s
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
