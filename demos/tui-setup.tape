Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run wizard. Models are pre-pulled into setup-models by
# demos/_prep.sh so the wizard reliably shows "Your existing models
# are ready" -- a real cold pull of Qwen3 4B Q8_0 (2.5 GB) over a
# 240s Hide previously failed on slower connections and the demo
# captured a "Model not found in registry" error.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
# Point Ollama at an unused port so the wizard ignores any locally
# installed Ollama models.
Env OLLAMA_HOST "http://127.0.0.1:65535"

# cd into the data dir so `/add ./README.md` after the wizard resolves
# to the staged README (lilbee's own, pre-copied by demos/_prep.sh).
Type "cd /tmp/lilbee-demo/tui-setup"
Sleep 300ms
Enter
Sleep 500ms

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Wizard appears with the curated catalog. The footer reads
# "Your existing models are ready - Esc to return". Esc drops into chat.
Sleep 1500ms
Escape
Sleep 1500ms

# First chat: index lilbee's own README and ask a starter question.
# Hide the embedder warmup and the cold first-token wait so the chat
# prompt isn't on camera doing nothing.
Type "/add ./README.md"
Sleep 600ms
Enter
Sleep 800ms
Hide
Sleep 22s
Show
Sleep 400ms

Type "What is lilbee and how do I use it?"
Sleep 1500ms
Enter
Sleep 800ms
Hide
Sleep 28s
Show
Sleep 14s

# Overwrite the wizard still with the cited-answer screen -- that's the
# moment the demo lands on.
Screenshot demos/_out/tui-setup.png
Sleep 500ms

Type "/quit"
Enter
Sleep 500ms
