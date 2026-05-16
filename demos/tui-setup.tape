Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# First-run wizard with a REAL cold pull on camera. setup-models is wiped
# by demos/_prep.sh so the wizard kicks off two parallel downloads in the
# Task Center. Hide elides the slow middle; Show resumes ONLY after both
# downloads are guaranteed to be at "done".
#
# Empirical baseline today: Qwen3 0.6B Q8_0 (~500 MB) + Nomic v1.5
# (~85 MB) reached "done" at t+419s. Bandwidth fluctuates run-to-run
# (a prior render captured Qwen at 93% when chat fired), so Hide is
# sized at a hard 1000s -- ~2.4x the measured time. The question
# absolutely never fires before both downloads complete.
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

# Wizard appears with Qwen3 0.6B auto-focused on the chat row. Enter
# picks it; Tab into the Embedding row and Enter picks Nomic v1.5.
Sleep 1500ms
Enter
Sleep 1200ms
Tab
Sleep 300ms
Enter
Sleep 800ms

# Jump to Task Center; show the two parallel downloads kick off briefly,
# then Hide elides the entire ~7 minute cold-pull middle, then resume
# at "done" state. Hide is sized for worst-case bandwidth so the demo
# NEVER captures a mid-download chat.
Type "t"
Sleep 8s

Hide
Sleep 1000s
Show
Sleep 4s

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
Sleep 10s
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
