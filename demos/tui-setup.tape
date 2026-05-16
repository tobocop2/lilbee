Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run wizard with a real cold pull on camera. setup-models is wiped
# by demos/_prep.sh so the wizard kicks off two parallel downloads
# (Qwen3 0.6B chat + Nomic v1.5 embedder, ~480 MB total) in the Task
# Center. Hide elides the slow middle; Show resumes once both downloads
# are at 100% (~170s at 2.4 MB/s, the empirical measurement).
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
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

# Wizard appears with the curated catalog, Qwen3 0.6B auto-focused on
# the chat row. Enter picks it; Tab into the Embedding row and Enter
# picks Nomic v1.5 (first card). Both kick off parallel downloads.
Sleep 1500ms
Enter
Sleep 1200ms
Tab
Sleep 300ms
Enter
Sleep 800ms

# Jump to Task Center; show the two parallel downloads kick off briefly,
# then elide the slow middle, then come back at "done" state.
Type "t"
Sleep 8s

Hide
Sleep 200s
Show
Sleep 3s

# Back to the wizard so the user sees the cards now marked installed.
Type "q"
Sleep 1500ms

# Dismiss into Chat.
Escape
Sleep 1500ms

# First chat: index lilbee's own README and ask a starter question.
# Hide the embedder warmup so the chat prompt isn't on camera idle.
Type "/add ./README.md"
Sleep 600ms
Enter
Sleep 800ms
Hide
Sleep 18s
Show
Sleep 400ms

Type "What is lilbee?"
Sleep 1500ms
Enter
Sleep 800ms
Hide
Sleep 12s
Show
Sleep 6s

# Overwrite any wizard still with the cited-answer screen.
Screenshot demos/_out/tui-setup.png
Sleep 400ms

Type "/quit"
Enter
Sleep 500ms
