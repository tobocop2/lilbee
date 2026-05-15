Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# Fresh-slate first run. Wizard appears with no installed models. The
# user steps right from the auto-focused Qwen3 0.6B to pick Qwen3 4B
# (2.5 GB, Q8_0) for the chat model and Nomic v1.5 (~84 MB) for the
# embedding, then watches the Task Center fill. 4B gives a coherent
# cited answer on the first question after the wizard.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
# Point Ollama at an unused port so the wizard can't detect locally
# installed Ollama models. Without this, picking Qwen3 0.6B is a no-op
# (the model is "already available via Ollama") and the Task Center
# only shows the embedding download instead of two parallel pulls.
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

# Wizard is fully painted: chat cards on top, embedding cards below,
# Qwen3 0.6B auto-focused. Sleep first so the GridSelect has the
# keyboard focus, then Right walks the cursor one card over to
# Qwen3 4B (Q8_0, 2.5 GB) for a chat model that produces a coherent
# answer on the first question.
Sleep 800ms
Right
Sleep 1500ms
Enter
Sleep 1500ms

# Tab into the Embedding row and pick Nomic v1.5 (first card).
Tab
Sleep 300ms
Enter
Sleep 800ms

# Jump to Task Center; show the two parallel downloads kick off, then
# elide the slow middle (Qwen3 0.6B is ~4min on a slow connection),
# then show the final state with both at 100%.
Type "t"
Sleep 6s

Hide
Sleep 240s
Show
Sleep 4s

# Back to the wizard so the user sees the two cards now marked installed.
Type "q"
Sleep 1500ms

# Dismiss into Chat.
Escape
Sleep 2000ms

# First chat: index lilbee's own README and ask a starter question.
# Qwen3 4B returns a structured cited answer in ~35s. The post-/add
# sleep is long because the embedder mmaps the GGUF and warms up on
# first use; without that wait, the next sync starts on a not-ready
# worker and the question loses its retrieval context.
Type "/add ./README.md"
Sleep 600ms
Enter
Sleep 25s

Type "What is lilbee and how do I use it?"
Sleep 1500ms
Enter
Sleep 55s

# Overwrite the wizard still with the cited-answer screen -- that's the
# moment the demo lands on.
Screenshot demos/_out/tui-setup.png
Sleep 3s

Type "/quit"
Enter
Sleep 500ms
